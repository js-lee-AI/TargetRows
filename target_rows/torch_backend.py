"""Static-cache Llama and Qwen3 inference with explicit tree attention masks.

Install the gpu extra before importing this module. CUDA graphs are optional;
CPU execution is supported for small correctness tests.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FastLM:
    """One fixed computation with caller-managed cache positions.

    Attention and position semantics match the supported architectures. Floating-
    point results can differ across batch sizes and from Hugging Face kernels.
    """
    def __init__(self, path, device, max_len=2048, buckets=(), dtype=torch.bfloat16,
                 revision=None, local_files_only=False):
        hf = AutoModelForCausalLM.from_pretrained(
            path, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
            local_files_only=local_files_only, attn_implementation="eager",
        ).eval()
        self._load(hf, device, max_len, buckets, dtype)

    @classmethod
    def from_model(cls, model, device="cpu", max_len=128, buckets=(), dtype=torch.float32):
        """Construct from an already loaded model, including tiny test models."""
        lm = cls.__new__(cls)
        lm._load(model, device, max_len, buckets, dtype)
        return lm

    def _load(self, hf, device, max_len, buckets, dtype):
        cfg = hf.config
        device = torch.device(device)
        if cfg.model_type not in ("llama", "qwen3"):
            raise ValueError("Supported model types are llama and qwen3.")
        if (getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False)
                or getattr(cfg, "use_sliding_window", False)
                or getattr(cfg, "hidden_act", "silu") != "silu"
                or getattr(cfg, "quantization_config", None)):
            raise ValueError("Use unquantized, bias-free models with full attention and SiLU.")
        rope_scaling = cfg.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type not in ("default", "llama3"):
            raise ValueError("This backend supports default and llama3 rotary embeddings.")
        if max_len < 4 or max_len > cfg.max_position_embeddings:
            raise ValueError("Cache length must be within the model's position limit and at least 4.")
        if buckets and device.type != "cuda":
            raise ValueError("CUDA graph buckets require a CUDA device.")
        hf = hf.to(dtype=dtype)
        self.cfg, self.dev, self.dtype, self.max_len = cfg, device, dtype, max_len
        self.nh, self.kvh = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self.g = self.nh // self.kvh
        self.eps = cfg.rms_norm_eps
        self.qk_norm = cfg.model_type == "qwen3"
        self.scale = self.hd ** -0.5
        m = hf.model
        with torch.no_grad():
            self.embed = m.embed_tokens.weight.to(device)
            tied = hf.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()
            self.lm_head = self.embed if tied else hf.lm_head.weight.to(device)
            self.layers = []
            for layer in m.layers:
                a, mlp = layer.self_attn, layer.mlp
                d = {"wqkv": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).to(device),
                     "wo": a.o_proj.weight.to(device),
                     "wgu": torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).to(device),
                     "wd": mlp.down_proj.weight.to(device),
                     "ln1": layer.input_layernorm.weight.to(device),
                     "ln2": layer.post_attention_layernorm.weight.to(device)}
                if self.qk_norm:
                    d["qn"] = a.q_norm.weight.to(device)
                    d["kn"] = a.k_norm.weight.to(device)
                self.layers.append(d)
            self.norm = m.norm.weight.to(device)
        self.tied = tied
        del hf, m
        rs = cfg.rope_scaling
        rope_type = (rs.get("rope_type", rs.get("type")) if rs else None) or "default"
        inv_freq, att_scale = ROPE_INIT_FUNCTIONS[rope_type](cfg, device)
        pos = torch.arange(max_len + 64, dtype=torch.float32, device=device)
        freqs = pos[:, None] * inv_freq.float()[None, :]
        emb = torch.cat([freqs, freqs], -1)
        self.cos = (emb.cos() * att_scale).to(dtype)
        self.sin = (emb.sin() * att_scale).to(dtype)
        nl = len(self.layers)
        self.k = torch.zeros(nl, self.kvh, max_len, self.hd, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.neg = torch.finfo(dtype).min
        self.graphs = {}
        self.pool = None
        for q in sorted(set(buckets)):
            self.capture(q)

    @torch.no_grad()
    def _forward(self, ids, pos, slots, mask):
        """ids, pos, slots [Q] long and mask [Q, max_len] bool; returns logits [Q, vocab] in the model dtype."""
        Q = ids.shape[0]
        nh, kvh, hd, g = self.nh, self.kvh, self.hd, self.g
        x = self.embed[ids]
        cos = self.cos[pos][:, None, :]
        sin = self.sin[pos][:, None, :]
        madd = torch.where(mask, 0.0, self.neg).to(self.dtype)
        madd = madd[None].expand(g, Q, self.max_len).reshape(g * Q, self.max_len)
        for li, d in enumerate(self.layers):
            h = _rms(x, d["ln1"], self.eps)
            qkv = h @ d["wqkv"].T
            q = qkv[:, : nh * hd].view(Q, nh, hd)
            k = qkv[:, nh * hd: (nh + kvh) * hd].view(Q, kvh, hd)
            v = qkv[:, (nh + kvh) * hd:].view(Q, kvh, hd)
            if self.qk_norm:
                q = _rms(q, d["qn"], self.eps)
                k = _rms(k, d["kn"], self.eps)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            self.k[li].index_copy_(1, slots, k.transpose(0, 1))
            self.v[li].index_copy_(1, slots, v.transpose(0, 1))
            qr = q.transpose(0, 1).reshape(kvh, g * Q, hd)
            s = torch.matmul(qr, self.k[li].transpose(1, 2)) * self.scale + madd
            p = torch.softmax(s, dim=-1, dtype=torch.float32).to(self.dtype)
            o = torch.matmul(p, self.v[li]).view(kvh, g, Q, hd).reshape(nh, Q, hd).transpose(0, 1).reshape(Q, nh * hd)
            x = x + o @ d["wo"].T
            h = _rms(x, d["ln2"], self.eps)
            gate, up = (h @ d["wgu"].T).chunk(2, dim=-1)
            x = x + (F.silu(gate) * up) @ d["wd"].T
        x = _rms(x, self.norm, self.eps)
        return x @ self.lm_head.T

    def capture(self, Q):
        if not 1 <= Q < self.max_len:
            raise ValueError("Invalid CUDA graph size.")
        dev = self.dev
        st = {"ids": torch.zeros(Q, dtype=torch.long, device=dev), "pos": torch.zeros(Q, dtype=torch.long, device=dev),
              "slots": torch.full((Q,), self.max_len - 1, dtype=torch.long, device=dev),
              "mask": torch.zeros(Q, self.max_len, dtype=torch.bool, device=dev)}
        st["mask"][:, 0] = True
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(2):
                self._forward(st["ids"], st["pos"], st["slots"], st["mask"])
        torch.cuda.current_stream(dev).wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.no_grad(), torch.cuda.device(dev), torch.cuda.graph(graph, pool=self.pool):
            out = self._forward(st["ids"], st["pos"], st["slots"], st["mask"])
        self.graphs[Q] = (graph, st, out)

    def bucket(self, n):
        for q in sorted(self.graphs):
            if q >= n:
                return q
        return None

    @torch.no_grad()
    def run(self, ids, pos, slots, fill_mask):
        """ids, pos, slots: long tensors on the device of length n; fill_mask(mask_rows) sets the n real mask rows,
        which arrive zeroed. Returns logits [n, vocab], copied out of graph memory when a graph ran, because the
        graphs share one memory pool and a replay of another size may overwrite this one's output."""
        n = ids.shape[0]
        Q = self.bucket(n)
        if Q is None:
            mask = torch.zeros(n, self.max_len, dtype=torch.bool, device=self.dev)
            fill_mask(mask)
            return self._forward(ids, pos, slots, mask)
        graph, st, out = self.graphs[Q]
        st["ids"][:n].copy_(ids)
        st["pos"][:n].copy_(pos)
        st["slots"][:n].copy_(slots)
        if Q > n:
            st["ids"][n:].zero_()
            st["pos"][n:].zero_()
            st["slots"][n:].fill_(self.max_len - 1)
        st["mask"].zero_()
        st["mask"][n:, 0] = True
        fill_mask(st["mask"][:n])
        graph.replay()
        return out[:n].clone()

    def prefill(self, ids):
        """Feed a prompt at positions and slots 0..P-1 with a causal mask; returns logits of the last token."""
        P = len(ids)
        dev = self.dev
        if not 0 < P < self.max_len - 1:
            raise ValueError("Prompt must be nonempty and fit in the cache.")
        t = torch.tensor(ids, dtype=torch.long, device=dev)
        ar = torch.arange(P, device=dev)
        cols = torch.arange(self.max_len, device=dev)
        out = None
        # chunks of at most 512 rows bound the score tensor
        for s0 in range(0, P, 512):
            sub = ar[s0:min(P, s0 + 512)]
            with torch.no_grad():
                out = self._forward(t[s0:s0 + len(sub)], sub, sub, cols[None, :] <= sub[:, None])
        return out[-1]
