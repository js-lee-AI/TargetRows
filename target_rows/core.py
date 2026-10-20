"""CPU reference routines; token paths are relative to the current context."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import numpy as np

Path = tuple[int, ...]
Rows = Mapping[Path, Sequence[float]]


def _token(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Token IDs must be nonnegative integers")
    return int(value)


def _path(path: Iterable[int]) -> Path:
    return tuple(_token(token) for token in path)


def _budget(value: int, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Budget must be an integer >= {minimum}")
    return int(value)


def _probabilities(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("A probability row must be a finite nonnegative vector")
    total = p.sum()
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("A probability row must sum to one")
    return p / total


def _tree(paths: Iterable[Path]) -> tuple[Path, ...]:
    paths = tuple(_path(path) for path in paths)
    nodes = set(paths)
    if len(nodes) != len(paths) or () in nodes:
        raise ValueError("Tree nodes must be unique nonempty paths; the root is implicit")
    if any(len(path) > 1 and path[:-1] not in nodes for path in paths):
        raise ValueError("The tree must contain every non-root ancestor")
    return paths


def _rows(rows: Rows) -> dict[Path, np.ndarray]:
    checked = {_path(path): _probabilities(row) for path, row in rows.items()}
    if len({len(row) for row in checked.values()}) > 1:
        raise ValueError("All rows must use the same vocabulary")
    return checked


def distribution(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Softmax for positive temperature; an argmax point mass at temperature zero."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("Logits must be a finite nonempty vector")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    if temperature == 0:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((logits - logits.max()) / temperature)
    return weights / weights.sum()


@dataclass(frozen=True)
class CandidatePool:
    """Prefix-closed paths with cumulative draft log probabilities and draft ranks.

    Parents must precede children. Scores are finite log q(path), not edge scores.
    An optional rank preserves a drafter's original ordering; lower ranks are better.
    """

    paths: tuple[Path, ...]
    draft_log_scores: tuple[float, ...]
    draft_rank: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        paths = _tree(self.paths)
        scores = np.asarray(self.draft_log_scores, dtype=np.float64)
        if scores.shape != (len(paths),) or not np.isfinite(scores).all() or (scores > 0).any():
            raise ValueError("Provide one finite nonpositive cumulative log score per path")
        index = {path: i for i, path in enumerate(paths)}
        for i, path in enumerate(paths):
            if len(path) > 1:
                parent = index[path[:-1]]
                if parent >= i or scores[i] > scores[parent]:
                    raise ValueError("Parents must precede children and have at least their score")
        if self.draft_rank is None:
            order = sorted(range(len(paths)), key=lambda i: (-scores[i], len(paths[i]), i))
            ranks = [0] * len(paths)
            for rank, i in enumerate(order):
                ranks[i] = rank
        else:
            ranks = [_budget(rank) for rank in self.draft_rank]
            if sorted(ranks) != list(range(len(paths))):
                raise ValueError("Draft ranks must be a permutation of 0, ..., N-1")
        if any(len(p) > 1 and ranks[index[p[:-1]]] >= ranks[i] for i, p in enumerate(paths)):
            raise ValueError("Draft ranking must place parents before children")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "draft_log_scores", tuple(float(x) for x in scores))
        object.__setattr__(self, "draft_rank", tuple(ranks))


def _top(pool: CandidatePool, scores: Sequence[float], budget: int) -> tuple[Path, ...]:
    budget = _budget(budget)
    order = sorted(range(len(pool.paths)), key=lambda i: (
        -scores[i], len(pool.paths[i]), pool.draft_rank[i],
    ))
    return tuple(pool.paths[i] for i in order[:budget])


def select_draft(pool: CandidatePool, budget: int) -> tuple[Path, ...]:
    """Retain at most budget non-root nodes in the original draft order."""
    return _top(pool, [-rank for rank in pool.draft_rank], budget)


def select_root(pool: CandidatePool, root_row: Sequence[float], budget: int) -> tuple[Path, ...]:
    """Rank by p(x1 | context) times the remaining draft path factors."""
    p = _probabilities(root_row)
    index = {path: i for i, path in enumerate(pool.paths)}
    scores = []
    for i, path in enumerate(pool.paths):
        if path[0] >= len(p):
            raise ValueError("A first token lies outside the root row's vocabulary")
        first = index[path[:1]]
        log_p = np.log(p[path[0]]) if p[path[0]] > 0 else -np.inf
        scores.append(log_p + (pool.draft_log_scores[i] - pool.draft_log_scores[first]))
    return _top(pool, scores, budget)


def select_root_draw(pool: CandidatePool, first_token: int, budget: int) -> tuple[Path, ...]:
    """Keep an already sampled target token, then allocate descendants in draft order.

    The first token counts toward budget. If it is absent from the pool, return it
    alone. The caller must obtain its target row during verification.
    """
    first = (_token(first_token),)
    budget = _budget(budget, 1)
    descendants = [i for i, p in enumerate(pool.paths) if len(p) > 1 and p[:1] == first]
    descendants.sort(key=lambda i: pool.draft_rank[i])
    return (first,) + tuple(pool.paths[i] for i in descendants[:budget - 1])


def select_preselected(pool: CandidatePool, first_token: int, budget: int) -> tuple[Path, ...]:
    """Prune a preselected draft tree beneath the same target draw (PEARL-style)."""
    first = (_token(first_token),)
    tree = select_draft(pool, _budget(budget, 1))
    return (first,) + tuple(p for p in tree if len(p) > 1 and p[:1] == first)


def _path_masses(tree: Iterable[Path], rows: Rows) -> dict[Path, float]:
    tree, checked = _tree(tree), _rows(rows)
    masses = {(): 1.0}
    for path in sorted(tree, key=len):
        parent, token = path[:-1], path[-1]
        if parent not in checked or token >= len(checked[parent]):
            raise ValueError("Every edge needs its parent's target row and a valid token ID")
        masses[path] = masses[parent] * checked[parent][token]
    del masses[()]
    return masses


def expected_acceptance(tree: Iterable[Path], rows: Rows) -> float:
    """Sum target path masses: expected accepted draft tokens, excluding continuation."""
    return float(sum(_path_masses(tree, rows).values()))


def select_oracle(pool: CandidatePool, rows: Rows, budget: int) -> tuple[Path, ...]:
    """Reference ranking by exact target path probabilities; needs rows throughout the pool."""
    masses = _path_masses(pool.paths, rows)
    return _top(pool, [masses[p] for p in pool.paths], budget)


def expected_root_draw_acceptance(pool: CandidatePool, rows: Rows, budget: int) -> float:
    """Average acceptance over the target first draw, counting that draw as accepted."""
    budget = _budget(budget, 1)
    if () not in rows:
        raise ValueError("Expected root-draw acceptance requires the root distribution")
    masses = _path_masses(pool.paths, rows)
    total = 1.0
    for token in {p[0] for p in pool.paths}:
        total += sum(masses[p] for p in select_root_draw(pool, token, budget) if len(p) > 1)
    return float(total)


@dataclass(frozen=True)
class VerificationResult:
    accepted: Path
    continuation: int
    covered: bool

    @property
    def emitted(self) -> Path:
        """The accepted path followed by one continuation token."""
        return self.accepted + (self.continuation,)


def verify_tree(
    tree: Iterable[Path], rows: Rows, rng: np.random.Generator, *,
    first_token: int | None = None, stop_at_leaf: bool = False,
) -> VerificationResult:
    """Follow target draws through a tree and emit one additional token.

    rows must contain the root and every tree node, as scored by a verification
    pass. Additional scored prefixes may be supplied to evaluate address coverage.
    first_token, when supplied, MUST already be drawn from the exact root row;
    it is consumed once, without resampling. stop_at_leaf emits a leaf draw as
    the continuation instead of accepting it.
    """
    nodes, checked = set(_tree(tree)), _rows(rows)
    if not nodes.union({()}) <= checked.keys():
        raise ValueError("Verification requires target rows at the root and every tree node")
    vocabulary = len(checked[()])
    if any(token >= vocabulary for path in nodes for token in path):
        raise ValueError("A tree token lies outside the target vocabulary")
    if first_token is not None:
        first_token = _token(first_token)
        if first_token >= vocabulary or checked[()][first_token] == 0:
            raise ValueError("The supplied first token must have positive target probability")
    parents = {path[:-1] for path in nodes}
    accepted: Path = ()
    while True:
        if not accepted and first_token is not None:
            token = first_token
        else:
            token = int(rng.choice(vocabulary, p=checked[accepted]))
        child = accepted + (token,)
        if child not in nodes or (stop_at_leaf and child not in parents):
            return VerificationResult(accepted, token, child in checked)
        accepted = child


class ExactRowCache:
    """Rows addressed by the full token context for one fixed target configuration.

    Use a new cache, or clear it, whenever target weights, tokenizer, position or
    attention semantics, temperature, or probability transformations change.
    A cache cannot itself establish that two model computations are equivalent.
    """

    def __init__(self) -> None:
        self._values: dict[Path, np.ndarray] = {}

    def put(self, full_context: Iterable[int], row: Sequence[float]) -> None:
        self._values[_path(full_context)] = _probabilities(row).copy()

    def put_pass(self, context: Iterable[int], rows: Rows) -> None:
        context = _path(context)
        checked = _rows(rows)
        for path, row in checked.items():
            self.put(context + path, row)

    def get(self, full_context: Iterable[int]) -> np.ndarray | None:
        row = self._values.get(_path(full_context))
        return None if row is None else row.copy()

    def clear(self) -> None:
        self._values.clear()
