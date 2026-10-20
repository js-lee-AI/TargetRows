# TargetRows

Core reference implementation for **Understanding Target Probability Reuse in Speculative Decoding**.

Jungseob Lee, Giyoon Jang, Chailin Nam, Jiwoo Kum, Sihyeon Lee, Kyojun Choo, Jihun Song, Eunseo Song, and Chanjun Park.

TargetRows implements tree selection using a target model's next-token distribution and illustrates when a previously computed distribution can be reused exactly. It contains a small NumPy library, a CPU example, and tests of the selection and verification identities. Model weights, datasets, generated text, benchmark execution scripts, and GPU serving integrations are outside this release.

## Quick start

Python 3.10 or newer is required. The example needs no GPU, model download, or dataset.

```bash
git clone https://github.com/js-lee-AI/TargetRows.git
cd TargetRows
python -m pip install .
python examples/quickstart.py
python -m unittest discover -s tests -v
```

The example uses a synthetic three-token vocabulary. Its expected acceptance values are:

| Selection rule | Expected accepted tokens |
| --- | ---: |
| Draft ranking | 0.428 |
| Root selection | 1.824 |
| Oracle selection | 1.824 |
| Root-draw selection | 2.257 |
| Pruning a preselected tree | 1.228 |

These numbers explain the algorithms; they are not the paper's experimental results. All rules use a budget of four non-root nodes. Root-draw selection and pruning count the first target draw as an accepted token and average over that draw. The additional continuation token is excluded throughout.

## Selecting a tree

A candidate path is a tuple of token IDs relative to the current context. The empty path is the implicit root. A pool must contain all non-root ancestors of every candidate; parents precede children. Supply cumulative draft log probabilities, rather than individual edge log probabilities.

```python
import numpy as np
from target_rows import CandidatePool, select_root, select_root_draw

pool = CandidatePool(
    paths=((0,), (1,), (0, 0), (1, 0)),
    draft_log_scores=tuple(np.log([0.7, 0.3, 0.42, 0.24])),
)
root_row = np.array([0.2, 0.8])
tree = select_root(pool, root_row, budget=3)

rng = np.random.default_rng(7)
first_token = int(rng.choice(len(root_row), p=root_row))
adaptive_tree = select_root_draw(pool, first_token, budget=3)
```

For a path $u=(x_1,\ldots,x_d)$, root selection replaces only the first draft factor:

$$w_{\mathrm{root}}(u)=p(x_1\mid c)\prod_{i=2}^{d}q(x_i\mid c,x_{<i}).$$

`select_root` ranks candidates by this weight, with ties resolved by depth and then draft rank. `CandidatePool` accepts an optional `draft_rank` permutation to preserve an existing drafter's ordering. Scores must be finite cumulative log probabilities and must not increase along a path.

`select_root_draw` receives the first token **already sampled from the target root row**, retains that token, and fills the remaining budget with descendants in draft order. If the first token is absent from the pool, it returns that token alone. The verification pass must still evaluate the target at that token's prefix.

| Function | Purpose |
| --- | --- |
| `select_draft` | Keep the top candidates in draft order. |
| `select_root` | Replace the first path factor with the exact target probability. |
| `select_root_draw` | Allocate the tree after drawing the first target token. |
| `select_preselected` | Prune a preselected draft tree beneath that same draw, as in the PEARL-style comparison. |
| `select_oracle` | Rank by target path probabilities when rows throughout the pool are available. |

The oracle is a reference for analysis; obtaining all of its rows has a cost. Root selection changes the allocation but does not guarantee an improvement for every pool and target distribution.

## Verification and exact reuse

`verify_tree` follows target draws through a selected tree, accepts the in-tree path, and emits one continuation token. Its input `rows` maps each scored relative prefix to a complete target next-token probability vector, including the root at `()`. Supply rows at all selected nodes. Additional scored prefixes can be included when examining address coverage.

For root-draw selection, pass `first_token` to verification so the draw is consumed exactly once. For ordinary fixed-tree verification, omit it. `distribution(logits, temperature=0)` produces the argmax point mass used for greedy decoding; a positive temperature produces a softmax distribution. The rows must match the actual target decoding distribution, including any probability transformations used by the caller.

Under exhaustive traversal, the emitted continuation exits the candidate tree. If only that tree was scored, the new context has no reusable row. Setting `stop_at_leaf=True` instead emits a leaf draw as the continuation: it preserves that leaf's row at the cost of one accepted token on a covered round. Both policies return a `VerificationResult` with `accepted`, `continuation`, `emitted`, and `covered`.

`ExactRowCache` uses the **entire token context** as its key. Equal final tokens do not imply equal conditional distributions:

```python
from target_rows import ExactRowCache

cache = ExactRowCache()
cache.put((10, 20, 30), [0.2, 0.8])
assert cache.get((10, 20, 30)) is not None
assert cache.get((11, 20, 30)) is None
```

Each cache belongs to one fixed target computation. Clear it, or use a new instance, when model weights, tokenizer, attention or position semantics, numerical execution semantics, temperature, or other probability transformations change. The cache checks the address; the model integration is responsible for the computation being unchanged.

## Measuring acceptance

For a fixed prefix-closed tree $S$, exhaustive verification has expected accepted draft-token count

$$V(S)=\sum_{u\in S}p(u\mid c).$$

`expected_acceptance(tree, rows)` evaluates this identity. `expected_root_draw_acceptance(pool, rows, budget)` averages over the first target draw, including draws outside the candidate pool. Its leading one counts that accepted first token. Neither quantity includes the continuation token or estimates wall-clock speed.

The tests enumerate small trees and target streams to check oracle selection, acceptance, output distributions, the coverage cost of stopping, zero-probability branches, and reuse at exact addresses. To connect a model, provide its token paths, cumulative draft scores, and target rows; model execution and KV-cache management remain with the caller.

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

The preselected-tree comparison follows the selection setting discussed in [PEARL](https://arxiv.org/abs/2408.11850). This repository implements its pruning rule for a supplied candidate pool; it does not include PEARL's GPU scheduling system.
