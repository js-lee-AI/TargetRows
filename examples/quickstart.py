"""A synthetic three-token example; these values are not benchmark results."""

import numpy as np

from target_rows import (
    CandidatePool, ExactRowCache, expected_acceptance, expected_root_draw_acceptance,
    select_draft, select_oracle, select_preselected, select_root, select_root_draw,
    verify_tree,
)


def main() -> None:
    paths = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0, 0), (1, 0, 0))
    draft_mass = (.75, .20, .45, .30, .16, .04, .27, .08)
    pool = CandidatePool(paths, tuple(np.log(draft_mass)))
    rows = {path: [.3, .4, .3] for path in paths}
    rows.update({
        (): [.2, .7, .1], (0,): [.6, .3, .1], (1,): [.8, .15, .05],
        (0, 0): [.4, .5, .1], (1, 0): [.65, .2, .15],
        (2,): [.3, .4, .3],
    })
    budget = 4
    print("Synthetic example: expected accepted tokens (continuation excluded)")
    for name, tree in [
        ("Draft", select_draft(pool, budget)),
        ("Root", select_root(pool, rows[()], budget)),
        ("Oracle", select_oracle(pool, rows, budget)),
    ]:
        print(f"{name:12s} {expected_acceptance(tree, rows):.3f}  {tree}")
    print(f"Root-draw    {expected_root_draw_acceptance(pool, rows, budget):.3f}")

    # Average the pruned-tree baseline over the same target first draw.
    baseline = 1.0
    for token, probability in enumerate(rows[()]):
        if probability:
            tree = select_preselected(pool, token, budget)
            baseline += expected_acceptance(tree, rows) - probability
    print(f"Preselected  {baseline:.3f}")

    rng = np.random.default_rng(7)
    first = int(rng.choice(3, p=rows[()]))
    tree = select_root_draw(pool, first, budget)
    # A model integration obtains rows at each selected prefix before verification.
    pass_rows = {path: rows[path] for path in ((), *tree)}
    result = verify_tree(tree, pass_rows, rng, first_token=first)
    print(f"One root-draw round: emitted={result.emitted}, covered={result.covered}")

    cache = ExactRowCache()
    cache.put_pass((10, 20), pass_rows)
    assert cache.get((10, 20)) is not None
    assert cache.get((11, 20)) is None
    print("Exact-address lookup: same final token with a different prefix is a miss.")


if __name__ == "__main__":
    main()
