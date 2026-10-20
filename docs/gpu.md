# GPU generation and comparisons

The GPU runner implements the standalone-drafter deployment from the paper: a fixed pool of 710 candidates, target-root selection, root-draw selection, draft ranking, and preselected-tree pruning. It loads public Llama or Qwen3 weights, applies an explicit ancestor mask, and commits the accepted path in a static KV cache. No trained TargetRows checkpoint is needed.

## Installation

Use Linux, Python 3.10–3.12, and an NVIDIA GPU. Create a separate environment:

```bash
git clone https://github.com/js-lee-AI/TargetRows.git
cd TargetRows
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[gpu]"
python -m target_rows.gpu --help
```

The PyTorch command uses CUDA 12.8 wheels; select a different supported wheel from the [official installation matrix](https://pytorch.org/get-started/previous-versions/) if needed for your driver. The GPU extra pins Transformers to 4.53.1, the version used by the standalone implementation. FlashAttention, custom CUDA compilation, and EAGLE weights are not required. Both models use bfloat16 by default; `--dtype float16` supports GPUs without bfloat16.

## Choose a model pair

| Preset | Target | Drafter | Suggested starting hardware |
| --- | --- | --- | --- |
| `qwen3-4b` | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | One 24 GB GPU for both models |
| `qwen3-8b` | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Qwen3-0.6B | 24 GB or more for the target; 8 GB or more for the drafter |
| `llama3-8b` | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 24 GB or more for the target; 8 GB or more for the drafter |

These are capacity recommendations for short prompts, a 2,048-slot cache, and compact CUDA graphs, not measured minimum requirements. Leave additional memory for other processes and the CUDA runtime. A machine with at least 32 GB of host RAM is a practical starting point; checkpoints alone need roughly 10 GB of disk space for the smaller Qwen pair and 20 GB for an 8B pair. Larger caches and full graph capture increase VRAM use.

The Qwen 8B and Llama pairs are the paper's standalone-drafter pairs. Qwen 4B is a smaller runnable example. The paper's two-GPU timings used RTX A6000 GPUs; a one-GPU run executes drafting and target passes sequentially. Its timings should be interpreted as measurements of that deployment.

Llama access must be granted by the model provider. Accept the conditions for both models on Hugging Face and authenticate locally with `huggingface-cli login`. The repository downloads weights from their original providers; it does not redistribute them. No data or model weights are included in this release.

## 1. Generate on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 python -m target_rows.gpu generate \
  --preset qwen3-4b --policy root-draw \
  --prompt "Explain speculative decoding in two sentences." \
  --max-new-tokens 128 --budget 59 \
  --output results/qwen4b-generation.json
```

Generated text is printed to standard output; loading and timing information goes to standard error. Downloads are cached after the first call. Each invocation loads the weights and captures CUDA graphs, which can take a few minutes. These setup costs are excluded from the recorded generation time. Use `--policy root`, `--policy draft`, or `--policy ar` to select another decoder. AR loads only the target.

## 2. Compare policies on two GPUs

Provide your own prompts as JSONL, one object per line:

```json
{"id": "explanation", "text": "Explain speculative decoding in two sentences."}
{"id": "coding", "text": "Write a Python function that checks whether a string is a palindrome."}
```

Save this as `prompts.jsonl`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m target_rows.gpu compare \
  --preset qwen3-8b --target-device cuda:0 --draft-device cuda:1 \
  --prompts-file prompts.jsonl --budget 59 --temperature 0 \
  --max-new-tokens 256 --output results/qwen8b-greedy.json
```

This runs AR, draft selection, root selection, root-draw selection, and preselected-tree pruning on the same prompts. Drafting runs in a separate process on the second GPU. Root-based policies overlap the target's root pass with drafting; draft selection waits for the pool before verification. The preselected policy implements the pruning comparison, including its root pass, but not the full PEARL scheduling system or its post-verification stage.

For the other paper settings, rerun with `--budget 8` or `--temperature 1`, using a different output filename. At temperature zero, every record includes `greedy_matches_ar`; inspect it before interpreting timings. At positive temperature the decoder preserves target sampling in exact arithmetic, but equal seeds need not produce identical text across different execution paths.

## 3. Run the Llama pair

After obtaining access to both upstream models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m target_rows.gpu compare \
  --preset llama3-8b --target-device cuda:0 --draft-device cuda:1 \
  --prompt "Explain why speculative decoding can preserve the target model distribution." \
  --budget 59 --max-new-tokens 128 \
  --output results/llama8b-greedy.json
```

The same CLI accepts `--target-model` and `--draft-model` for local model directories or other compatible checkpoints, with optional `--target-revision` and `--draft-revision`. An override removes that model's preset revision. Token-to-ID vocabularies must match. Use `--local-files-only` to forbid network downloads.

## What the report measures

The JSON report contains the model IDs and requested revisions, software versions, GPU names, configuration, generated token IDs and text, per-prompt elapsed time, accepted-token counts, verified tree sizes, and process peak allocated GPU memory. On one GPU, the target-process peak includes both models. Output files are never silently overwritten.

Timing starts before prompt prefill and ends after the final token commit and target synchronization. It includes drafting, root passes, selection, verification, and cache updates. It excludes model downloads, loading, CUDA graph capture, tokenization, and warmup. Aggregate throughput is total emitted tokens divided by total elapsed time. `throughput_vs_draft` uses draft selection as the baseline. Short examples check functionality; use enough prompts, repeat measurements, and avoid competing GPU jobs when measuring performance.

Accepted-token counts exclude the additional continuation token. Statistics describe complete executed rounds. By default, returned text is clipped to `--max-new-tokens`; `--round-limit` instead retains the paper's stopping rule after a completed round, which can exceed that limit. Both stop the returned text at the first EOS. Use `--graphs full --round-limit --max-new-tokens 256` for the paper's graph sizing and stopping convention. Reproducing a reported table additionally requires its prompt set, seeds, hardware, and settings; these small examples do not reproduce the whole experimental study.

`--graphs compact` captures a few target call sizes and pads smaller passes; `--graphs full` captures every size from one through budget plus one; `--graphs off` runs eagerly. The draft pool always starts with ten first tokens and expands ten frontier nodes with ten children over seven further levels, giving 710 candidates. Budgets count non-root nodes. Qwen chat templates disable thinking, as in the standalone experiment. Sampling applies temperature and no top-k or top-p filter; draft pool scores use untempered draft probabilities.

## Model revisions

The presets pin the following upstream revisions:

| Model | Revision |
| --- | --- |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Llama-3.2-1B-Instruct | `9213176726f574b556790deb65791e0c5aa438b6` |

## Execution checks

Short greedy checks ran on A100 80 GB GPUs with Python 3.11, PyTorch 2.10.0 + CUDA 12.8, and Transformers 4.53.1. All four tree policies matched the AR output prefix for each pair below. These are functional checks, not speed benchmarks or minimum-memory measurements. CPU CI also checks the backend with PyTorch 2.7.1.

| Pair | GPUs | Budget / cache slots | Peak allocated VRAM, GiB |
| --- | --- | --- | --- |
| Qwen3-4B / 0.6B | 1 | 8 / 512 | 8.86, both models |
| Qwen3-8B / 0.6B | 2 | 59 / 2,048 | 15.68 target; 1.38 drafter |
| Llama-3.1-8B / 3.2-1B | 2 | 8 / 512 | 15.14 target; 2.44 drafter |

## Troubleshooting

| Symptom | Action |
| --- | --- |
| CUDA unavailable | Check `nvidia-smi` and install a CUDA-enabled PyTorch wheel. |
| Model access denied | Confirm access to both upstream checkpoints, then authenticate locally. |
| Out of GPU memory | Use the 4B preset, reduce `--max-context`, use `--graphs off`, or place the drafter on another GPU. |
| Context exceeds cache | Increase `--max-context` or shorten the prompt/output. The cache also needs tree scratch slots. |
| Tokenizer mismatch | Use a preset pair or checkpoints with identical token-to-ID mappings. |
| Greedy output mismatch | Inspect the saved tokens. Packed passes can differ numerically from single-token passes; do not treat a mismatched run as a successful fidelity check. |
| Newer Transformers fails | Use the pinned GPU extra in a fresh environment. The custom backend supports the listed architectures and rotary variants. |

The backend uses the model's RMSNorm, rotary embeddings, and Qwen3 query/key normalization. Small random-model tests compare every packed tree row and committed cache state against Hugging Face eager forwards. Quantized weights, sliding-window attention, and other model architectures are outside this runner.
