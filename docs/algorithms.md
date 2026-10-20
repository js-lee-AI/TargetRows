# Algorithms and integration

[Back to the README](../README.md)

All paths below are relative to one committed context. The empty tuple `()` is the root. A probability row is a complete next-token distribution at a scored prefix.

## Candidate pool

`CandidatePool` requires unique nonempty paths and all their non-root ancestors. Parents precede children. `draft_log_scores` contains finite cumulative log probabilities that do not increase along a path.

An optional `draft_rank` permutation preserves a drafter's original ordering, with lower ranks preferred. Parents must rank before their children. Without an explicit rank, the pool ranks by descending draft score, then depth, then input position.

The budget counts non-root nodes. If fewer eligible nodes are available, selection returns a smaller tree.

## Root selection

For a path of tokens, root selection replaces the first draft probability with the target probability and keeps the deeper draft factors:

```math
w_{\mathrm{root}}(u)
= p(x_1 \mid c)\prod_{i=2}^{d}q(x_i \mid c,x_1,\ldots,x_{i-1}).
```

`select_root` ranks by this weight. Ties are resolved by depth and then draft rank, preserving prefix closure even when target probabilities are zero. `select_draft` instead retains the original draft order. Root selection changes the allocation but does not guarantee improvement for every pool and target distribution.

`select_oracle` ranks by exact target path probabilities. It is a reference for analysis: obtaining rows throughout the candidate pool has a cost.

## Root-draw selection

`select_root_draw` receives the first token **already sampled from the exact target root row**. It retains that token and fills the remaining budget with its descendants in draft order. A first token absent from the pool is returned alone; its target row must still be evaluated during verification.

`select_preselected` provides the PEARL-style comparison. It first selects a fixed draft tree, then keeps the branch beneath the same target draw. Both rules retain and count the first draw.

## Verification

`verify_tree` takes the selected tree and a `rows` mapping containing the root and every tree node. All rows use the same vocabulary and target decoding distribution, including any temperature or probability transformations applied by the caller. Additional scored prefixes may be supplied to examine coverage.

For an ordinary fixed tree, verification samples each token from the target row at the current prefix. For a tree allocated after the first draw, pass `first_token` so that draw is consumed once, without resampling. An arbitrarily chosen first token does not provide the same distributional guarantee.

The result contains:

| Field | Meaning |
| --- | --- |
| `accepted` | The accepted token path. |
| `continuation` | One additional emitted token. |
| `emitted` | Accepted tokens followed by the continuation. |
| `covered` | Whether a scored row exists at the full emitted path. |

Exhaustive traversal accepts target draws while they remain in the tree and emits the first exit as the continuation. If only the tree was scored, this exit has no reusable row.

With `stop_at_leaf=True`, a leaf draw is emitted as the continuation rather than accepted. The leaf row then covers the new context, at the cost of one accepted token on that covered round.

`distribution(logits, temperature=0)` returns a point mass at the argmax, matching greedy decoding. Positive temperatures use softmax. The probability transformation must match the target decoding rule used for verification.

## Exact-address cache

`ExactRowCache` keys rows by the full token context. `put_pass(context, rows)` converts scored relative prefixes into full addresses. Equal final tokens do not establish equal conditional distributions.

The cache belongs to one fixed target computation. Clear it or create a new instance when weights, tokenizer, attention or position semantics, numerical execution semantics, temperature, or other probability transformations change. Address lookup alone cannot establish that two model computations are equivalent. Model execution and KV-cache management remain with the caller.

## Expected acceptance

For a fixed prefix-closed tree, exhaustive verification has expected accepted draft-token count:

```math
V(S)=\sum_{u\in S}p(u\mid c).
```

`expected_acceptance(tree, rows)` evaluates this sum. Each path probability is the product of the corresponding target edge probabilities.

`expected_root_draw_acceptance(pool, rows, budget)` averages over the target first draw, including draws outside the pool. Its leading one counts that accepted first token, followed by the target path masses of the selected descendants.

Both quantities exclude the continuation token. They measure expected acceptance, not wall-clock speed. The tests enumerate small trees and target streams to check these identities and the effect of stopping on coverage.
