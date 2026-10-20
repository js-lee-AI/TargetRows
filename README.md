<p align="center">
  <img src="assets/banner.png" width="100%" alt="TargetRows — target-aware tree selection and exact-context probability reuse" />
</p>

<h3 align="center">Understanding Target Probability Reuse<br>in Speculative Decoding</h3>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![GPU](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](docs/gpu.md) [![NumPy](https://img.shields.io/badge/NumPy-CPU-4D77CF?logo=numpy&logoColor=white)](https://numpy.org/) [![CI](https://github.com/js-lee-AI/TargetRows/actions/workflows/ci.yml/badge.svg)](https://github.com/js-lee-AI/TargetRows/actions/workflows/ci.yml)

<b><a href="#overview">Overview</a> · <a href="#quick-start">Quick start</a> · <a href="#gpu-examples">GPU examples</a> · <a href="#usage">Core API</a> · <a href="docs/gpu.md">GPU guide</a> · <a href="docs/algorithms.md">Method details</a> · <a href="#citation">Citation</a></b>

</div>

<p align="center">
  Jungseob Lee · Giyoon Jang · Chailin Nam · Jiwoo Kum · Sihyeon Lee<br>
  Kyojun Choo · Jihun Song · Eunseo Song · Chanjun Park
</p>

---

## Overview

Speculative decoding verifies a tree of draft tokens with a target model. The same pass also computes next-token probability distributions at the scored prefixes. **When can those distributions be reused, and how can a target distribution improve the next tree's allocation?**

TargetRows provides the selection algorithms and a GPU decoder for studying both questions:

| Component | What it does |
| --- | --- |
| **Root selection** | Reranks a candidate pool by replacing the first draft factor with the exact target probability. |
| **Root-draw selection** | Draws the first token from the target before allocating the tree beneath it. |
| **Exact-context reuse** | Looks up a retained probability row using the full committed token context. |
| **Verification and acceptance** | Follows target draws through a tree and computes expected accepted-token counts. |
| **GPU decoding** | Runs public Qwen3 or Llama target/drafter pairs with tree attention, a static KV cache, and optional CUDA graphs. |

Run both models on one GPU, or place the drafter on a second GPU to overlap drafting with the target's root pass. The core NumPy API remains available without PyTorch. **No separately trained TargetRows checkpoint is required:** the GPU examples load public target and draft models from Hugging Face.

## Quick start

```bash
git clone https://github.com/js-lee-AI/TargetRows.git
cd TargetRows
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[gpu]"

CUDA_VISIBLE_DEVICES=0 python -m target_rows.gpu generate \
  --preset qwen3-4b --policy root-draw \
  --prompt "Explain speculative decoding in two sentences." \
  --max-new-tokens 128 --output results/generation.json
```

Use Linux and Python 3.10–3.12 for the GPU runner. Start with a 24 GB NVIDIA GPU for this smaller Qwen pair. Downloads are cached after the first run; each invocation loads weights and prepares CUDA graphs before decoding. The [GPU guide](docs/gpu.md) covers driver-compatible PyTorch wheels, model access, memory, and measurement conventions.

## GPU examples

| Example | Public target / drafter | Deployment |
| --- | --- | --- |
| **Generate** | Qwen3-4B / Qwen3-0.6B | One GPU; both models run sequentially. |
| **Compare policies** | Qwen3-8B / Qwen3-0.6B | Two GPUs; drafting overlaps the target root pass. |
| **Second model family** | Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct | Two GPUs; upstream Llama access required. |

The first example is the quick start above. To compare all five decoders on a prompt:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m target_rows.gpu compare \
  --preset qwen3-8b --target-device cuda:0 --draft-device cuda:1 \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --budget 59 --temperature 0 --max-new-tokens 256 \
  --output results/qwen8b-comparison.json
```

For the Llama pair, after obtaining access to both models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m target_rows.gpu compare \
  --preset llama3-8b --target-device cuda:0 --draft-device cuda:1 \
  --prompt "Explain speculative decoding in two sentences." \
  --budget 59 --max-new-tokens 128 --output results/llama8b-comparison.json
```

`compare` evaluates AR, draft ranking, root selection, root-draw selection, and preselected-tree pruning. It prints throughput and saves generated text, token IDs, accepted-token counts, hardware, and configuration. At temperature zero it also checks each output against AR. Use `--temperature 1` for sampling, `--budget 8` for a smaller tree, or `--prompts-file prompts.jsonl` for your own prompt set. [Full commands and report definitions](docs/gpu.md) explain the overlap schedule, pinned checkpoints, and the paper's stopping convention.

## CPU example

```bash
python -m pip install .
python examples/quickstart.py
```

This example requires only NumPy and downloads no models or datasets.

<details>
<summary>Example output — synthetic three-token vocabulary</summary>

```text
Synthetic example: expected accepted tokens (continuation excluded)
Draft        0.428  ((0,), (0, 0), (0, 1), (0, 0, 0))
Root         1.824  ((1,), (1, 0), (1, 0, 0), (0,))
Oracle       1.824  ((1,), (1, 0), (1, 0, 0), (0,))
Root-draw    2.257
Preselected  1.228
```

These are illustrative values from the example, not the paper's benchmark results. The budget is four non-root nodes. Root-draw and preselected-tree pruning average over the target first draw and count it as accepted; all counts exclude the additional continuation token.

</details>

## Usage

### Select a tree with the target root row

Represent each candidate as a tuple of token IDs relative to the current context. The pool contains every candidate's ancestors, with parents listed before children. Supply cumulative draft log probabilities, rather than individual edge scores.

```python
import numpy as np
from target_rows import CandidatePool, select_root, select_root_draw

pool = CandidatePool(
    paths=((0,), (1,), (0, 0), (1, 0)),
    draft_log_scores=tuple(np.log([0.7, 0.3, 0.42, 0.24])),
)
root_row = np.array([0.2, 0.8])

# Replace the first draft factor with the target probability.
tree = select_root(pool, root_row, budget=3)
print(tree)  # ((1,), (1, 0), (0,))

# Or sample first, then allocate the tree beneath the sampled token.
rng = np.random.default_rng(7)
first = int(rng.choice(len(root_row), p=root_row))
adaptive_tree = select_root_draw(pool, first, budget=3)
```

For root-draw selection, pass the same `first` token to `verify_tree(..., first_token=first)` after obtaining the target rows for the selected tree. This consumes the draw once. The complete selection-and-verification example is in [`examples/quickstart.py`](examples/quickstart.py).

### Reuse a row at its exact context

```python
from target_rows import ExactRowCache

cache = ExactRowCache()
cache.put((10, 20, 30), [0.2, 0.8])

assert cache.get((10, 20, 30)) is not None  # Same full context.
assert cache.get((11, 20, 30)) is None      # Same last token, different context.
```

Each cache belongs to one fixed target computation. Clear it when model weights, tokenizer, attention or position semantics, numerical execution semantics, temperature, or probability transformations change. The caller is responsible for preserving those conditions.

## API

| Function | Purpose |
| --- | --- |
| `select_draft` | Keep candidates in the original draft order. |
| `select_root` | Replace the first path factor with the target probability. |
| `select_root_draw` | Allocate descendants after the first target draw. |
| `select_preselected` | Prune a preselected draft tree beneath that same draw. |
| `select_oracle` | Rank by target path probabilities when rows throughout the pool are available. |
| `verify_tree` | Accept a target path, emit one continuation, and report address coverage. |
| `expected_acceptance` | Evaluate expected accepted draft tokens for a fixed tree. |
| `expected_root_draw_acceptance` | Average acceptance over the target first draw. |
| `distribution` | Convert logits to a softmax distribution, or an argmax point mass at temperature zero. |
| `ExactRowCache` | Store and retrieve rows by the full token context. |

[Method details](docs/algorithms.md) cover the equations, tie handling, verification inputs, stopping policies, and acceptance conventions.

## Checks

```bash
python -m unittest discover -s tests -v
```

The base tests cover selection, exact output distributions on small trees, acceptance and coverage identities, zero-probability branches, and cache isolation. With the GPU extra installed, additional CPU tests use tiny random Llama and Qwen3 models to check packed tree logits and cache commits against Hugging Face, draft path scores, and greedy equivalence across all selection policies. CI runs these checks without downloading any checkpoints.

## FAQ

**What part of the paper can I run?**

The core algorithms and the standalone-drafter GPU deployment are included. The Qwen 8B and Llama examples use the paper's target/drafter pairs; Qwen 4B is a smaller starting point. The complete benchmark archive, EAGLE feature-drafter experiments, and learned-reranker training are outside this focused release.

**Do I need new model weights?**

No. Root selection and root-draw selection use the public target and drafter unchanged. Presets pin upstream revisions. Models are downloaded from their original providers, subject to their access conditions; this repository distributes code, not checkpoints or benchmark data.

**What happens if the first target draw is absent from the pool?**

Root-draw selection keeps that token alone. The caller must obtain its target row during verification.

**Does the acceptance count include the continuation token?**

No. Root-draw selection counts its first target draw as accepted, but excludes the continuation token, just like fixed-tree acceptance.

**Can I reuse a row because the final token matches?**

The cache requires the complete token context. Different prefixes can end in the same token while having different next-token distributions.

## Repository layout

```text
target_rows/core.py       selection, verification, acceptance, and exact-row cache
target_rows/torch_backend.py  Llama/Qwen3 forward, static KV cache, CUDA graphs
target_rows/decoding.py   draft pool, tree verification, and generation
target_rows/drafter.py    same-GPU drafting and the second-GPU worker
target_rows/gpu.py        generate/compare CLI and pinned model presets
examples/quickstart.py    complete synthetic CPU example
tests/test_core.py        selection and verification checks
tests/test_decoding.py    Hugging Face equivalence and decoder checks
docs/algorithms.md       equations and integration assumptions
docs/gpu.md              installation, three GPU recipes, and measurement guide
```

## Citation

```bibtex
@misc{lee2026targetrows,
  title = {Understanding Target Probability Reuse in Speculative Decoding},
  author = {Lee, Jungseob and Jang, Giyoon and Nam, Chailin and Kum, Jiwoo
            and Lee, Sihyeon and Choo, Kyojun and Song, Jihun and Song, Eunseo
            and Park, Chanjun},
  year = {2026},
  url = {https://github.com/js-lee-AI/TargetRows}
}
```

The preselected-tree comparison implements the PEARL-style pruning rule discussed in the paper, using a supplied candidate pool. See [PEARL](https://arxiv.org/abs/2408.11850) for its full decoding and GPU scheduling system.
