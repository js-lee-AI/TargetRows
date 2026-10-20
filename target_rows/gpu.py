"""Command-line generation and matched-policy comparisons on CUDA GPUs."""
import argparse
import json
import math
from pathlib import Path
import sys


PRESETS = {
    "qwen3-4b": {
        "target": "Qwen/Qwen3-4B", "drafter": "Qwen/Qwen3-0.6B",
        "target_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "drafter_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    "qwen3-8b": {
        "target": "Qwen/Qwen3-8B", "drafter": "Qwen/Qwen3-0.6B",
        "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "drafter_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    "llama3-8b": {
        "target": "meta-llama/Llama-3.1-8B-Instruct",
        "drafter": "meta-llama/Llama-3.2-1B-Instruct",
        "target_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "drafter_revision": "9213176726f574b556790deb65791e0c5aa438b6",
    },
}
POLICIES = ("ar", "draft", "root", "root-draw", "preselected")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "compare"))
    p.add_argument("--preset", choices=PRESETS, default="qwen3-4b")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="User message, formatted with the target chat template.")
    group.add_argument("--prompts-file", type=Path, help="JSONL with a text field and optional id per line.")
    p.add_argument("--policy", choices=POLICIES, default="root-draw", help="Policy for generate.")
    p.add_argument("--policies", default=",".join(POLICIES), help="Comma-separated policies for compare.")
    p.add_argument("--budget", type=int, default=59, help="Non-root tree nodes (paper settings: 8 or 59).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-context", type=int, default=2048, help="KV cache including prompt, output and scratch slots.")
    p.add_argument("--target-device", default="cuda:0")
    p.add_argument("--draft-device", default="cuda:0", help="Use cuda:1 for a separate drafter process.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--graphs", choices=("compact", "full", "off"), default="compact")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--warmup", type=int, default=1, help="Unmeasured short decodes per selected policy.")
    p.add_argument("--round-limit", action="store_true", help="Stop after a full round; may exceed max-new-tokens.")
    p.add_argument("--output", type=Path, help="Save configuration, outputs, timings and comparison summary as JSON.")
    p.add_argument("--target-model", help="Override with a Hugging Face model ID or local directory.")
    p.add_argument("--draft-model", help="Override with a Hugging Face model ID or local directory.")
    p.add_argument("--target-revision")
    p.add_argument("--draft-revision")
    p.add_argument("--local-files-only", action="store_true")
    return p


def read_prompts(args):
    if args.prompt is not None:
        records = [{"id": "prompt-0", "text": args.prompt}]
    else:
        records = []
        for line_no, line in enumerate(args.prompts_file.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Prompt line {line_no} must be a JSON object.")
                records.append({"id": str(item.get("id", f"prompt-{line_no}")), "text": item.get("text")})
    if not records or any(not isinstance(x["text"], str) or not x["text"].strip() for x in records):
        raise ValueError("Provide at least one nonempty text prompt.")
    if len({x["id"] for x in records}) != len(records):
        raise ValueError("Prompt IDs must be unique.")
    return records


def graph_buckets(mode, budget):
    if mode == "off":
        return (), ()
    if mode == "full":
        return tuple(range(1, budget + 2)), tuple(range(1, 11))
    target = {1, budget, budget + 1}
    target.update(x for x in (8, 16, 32, 64) if x < budget + 1)
    return tuple(sorted(target)), (1, 10)


def run(args):
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, GenerationConfig
        from .torch_backend import FastLM
        from .drafter import LocalDrafter, ProcessDrafter
        from .decoding import ar_decode, decode, validate_decode
    except ImportError as exc:
        raise RuntimeError('GPU dependencies are missing. Install PyTorch, then pip install ".[gpu]".') from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA-enabled PyTorch build and an NVIDIA GPU.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if not 1 <= args.budget <= 710 or args.max_new_tokens < 1 or args.warmup < 0:
        raise ValueError("Budget must be in [1, 710], token limit positive, and warmup nonnegative.")
    policies = [args.policy] if args.command == "generate" else args.policies.split(",")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError(f"Choose distinct policies from {', '.join(POLICIES)}.")
    if args.command == "generate" and args.prompts_file:
        raise ValueError("Use compare with --prompts-file, or generate with --prompt.")
    if args.output and args.output.exists():
        raise ValueError("Output file already exists; choose a new path.")
    prompts = read_prompts(args)
    need_drafter = any(p != "ar" for p in policies)
    target_dev, draft_dev = torch.device(args.target_device), torch.device(args.draft_device)
    active_devices = [target_dev, draft_dev] if need_drafter else [target_dev]
    for dev in active_devices:
        if dev.type != "cuda" or dev.index is None or dev.index >= torch.cuda.device_count():
            raise ValueError("Use explicit visible CUDA devices, such as cuda:0 and cuda:1.")
        with torch.cuda.device(dev):
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU lacks bfloat16 support; use --dtype float16.")
    torch.cuda.set_device(target_dev)
    torch.set_num_threads(2)
    preset = PRESETS[args.preset]
    target_name = args.target_model or preset["target"]
    draft_name = args.draft_model or preset["drafter"]
    target_rev = args.target_revision or (None if args.target_model else preset["target_revision"])
    draft_rev = args.draft_revision or (None if args.draft_model else preset["drafter_revision"])
    common = {"local_files_only": args.local_files_only}
    tok = AutoTokenizer.from_pretrained(target_name, revision=target_rev, **common)
    if need_drafter:
        draft_tok = AutoTokenizer.from_pretrained(draft_name, revision=draft_rev, **common)
        if tok.get_vocab() != draft_tok.get_vocab():
            raise ValueError("Target and drafter must use the same token-to-ID vocabulary.")
    target_buckets, draft_buckets = graph_buckets(args.graphs, args.budget)
    dtype = getattr(torch, args.dtype)
    print(f"Loading {target_name} on {target_dev}", file=sys.stderr, flush=True)
    target = FastLM(target_name, target_dev, args.max_context, target_buckets, dtype,
                    revision=target_rev, **common)
    if target.embed.shape[0] < len(tok):
        raise ValueError("Tokenizer vocabulary exceeds the target embedding table.")
    cfg = GenerationConfig.from_pretrained(target_name, revision=target_rev, **common)
    end = cfg.eos_token_id
    eos = set(end if isinstance(end, list) else ([] if end is None else [end]))
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    for item in prompts:
        options = {"enable_thinking": False} if target.cfg.model_type == "qwen3" else {}
        item["ids"] = tok.apply_chat_template(
            [{"role": "user", "content": item["text"]}], tokenize=True, add_generation_prompt=True, **options,
        )
        validate_decode(target, item["ids"], args.temperature, args.budget, args.max_new_tokens)
    drafter = None
    try:
        if need_drafter:
            print(f"Loading {draft_name} on {draft_dev}", file=sys.stderr, flush=True)
            kwargs = dict(path=draft_name, device=draft_dev, max_len=args.max_context,
                          buckets=draft_buckets, dtype=dtype, revision=draft_rev, **common)
            if draft_dev == target_dev:
                drafter = LocalDrafter(FastLM(**kwargs), len(tok))
            else:
                drafter = ProcessDrafter(kwargs, len(tok))

        def generate(ids, policy, seed, limit):
            if policy == "ar":
                tokens, seconds = ar_decode(target, ids, eos, limit, args.temperature, seed)
                return {"emitted": tokens, "tokens": len(tokens), "seconds": seconds}
            return decode(target, drafter, ids, args.temperature, args.budget, policy,
                          seed, eos, limit, round_limit=args.round_limit)

        for w in range(args.warmup):
            for policy in policies:
                generate(prompts[w % len(prompts)]["ids"], policy, args.seed + w, min(16, args.max_new_tokens))
        records = []
        for i, item in enumerate(prompts):
            order = policies[i % len(policies):] + policies[:i % len(policies)]
            current = []
            for policy in order:
                row = generate(item["ids"], policy, args.seed + i, args.max_new_tokens)
                row.update(prompt_id=item["id"], policy=policy, prompt_tokens=len(item["ids"]), seed=args.seed + i)
                row["text"] = tok.decode(row["emitted"], skip_special_tokens=True)
                row["tokens_per_second"] = row["tokens"] / row["seconds"]
                current.append(row)
                print(f"{item['id']}  {policy:11s}  {row['tokens']:4d} tokens  "
                      f"{row['seconds']:.3f} s  {row['tokens_per_second']:.2f} tokens/s", file=sys.stderr, flush=True)
            if args.temperature == 0 and "ar" in policies:
                reference = next(x["emitted"] for x in current if x["policy"] == "ar")
                for row in current:
                    # A round limit may produce a longer speculative output than AR.
                    row["greedy_matches_ar"] = (row["emitted"][:len(reference)] == reference)
                    if not row["greedy_matches_ar"]:
                        print(f"Greedy mismatch: {item['id']} / {row['policy']}; inspect the saved tokens.",
                              file=sys.stderr, flush=True)
            records.extend(current)
        summary = {}
        for policy in policies:
            selected = [x for x in records if x["policy"] == policy]
            tokens, seconds = sum(x["tokens"] for x in selected), sum(x["seconds"] for x in selected)
            entry = {"tokens": tokens, "seconds": seconds, "tokens_per_second": tokens / seconds}
            if policy != "ar":
                rounds = sum(x["rounds"] for x in selected)
                entry["mean_accepted_per_round"] = sum(x["sum_A"] for x in selected) / rounds if rounds else 0
                entry["mean_verified_nodes"] = sum(x["sum_size"] for x in selected) / rounds if rounds else 0
            if args.temperature == 0 and "ar" in policies:
                entry["greedy_matches_ar"] = all(x["greedy_matches_ar"] for x in selected)
            summary[policy] = entry
        if "draft" in summary:
            baseline = summary["draft"]["tokens_per_second"]
            for entry in summary.values():
                entry["throughput_vs_draft"] = entry["tokens_per_second"] / baseline
        report = {
            "config": {
                "preset": args.preset, "target": target_name, "drafter": draft_name if need_drafter else None,
                "target_revision": target_rev, "drafter_revision": draft_rev if need_drafter else None,
                "target_device": str(target_dev), "draft_device": str(draft_dev) if need_drafter else None,
                "overlap": need_drafter and target_dev != draft_dev,
                "dtype": args.dtype, "temperature": args.temperature, "budget": args.budget,
                "max_new_tokens": args.max_new_tokens, "max_context": args.max_context,
                "token_limit": "after_round" if args.round_limit else "strict",
                "graphs": args.graphs, "target_buckets": target_buckets, "draft_buckets": draft_buckets,
                "warmup": args.warmup, "seed": args.seed, "policies": policies,
            },
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "cuda": torch.version.cuda, "target_gpu": torch.cuda.get_device_name(target_dev),
                            "drafter_gpu": torch.cuda.get_device_name(draft_dev) if need_drafter else None},
            "memory": {"target_process_peak_gib": torch.cuda.max_memory_allocated(target_dev) / 2**30,
                       "drafter_process_peak_gib": drafter.memory_gib() if need_drafter and draft_dev != target_dev else None},
            "summary": summary, "records": records,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {args.output}", file=sys.stderr)
        if args.command == "generate":
            print(records[0]["text"])
        else:
            print(json.dumps(summary, indent=2))
        return report
    finally:
        if drafter is not None:
            drafter.close()


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
