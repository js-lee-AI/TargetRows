import itertools
import math
import unittest

import numpy as np

from target_rows import (
    CandidatePool, ExactRowCache, distribution, expected_acceptance,
    expected_root_draw_acceptance, select_draft, select_oracle, select_preselected,
    select_root, select_root_draw, verify_tree,
)


def fixture():
    paths = ((0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1), (0, 0, 0), (1, 0, 0))
    pool = CandidatePool(paths, tuple(np.log([.75, .20, .45, .30, .16, .04, .27, .08])))
    rows = {p: [.3, .4, .3] for p in paths}
    rows.update({(): [.2, .7, .1], (0,): [.6, .3, .1], (1,): [.8, .15, .05],
                 (0, 0): [.4, .5, .1], (1, 0): [.65, .2, .15], (2,): [.3, .4, .3]})
    return pool, rows


def path_mass(path, rows):
    return math.prod(rows[path[:i]][token] for i, token in enumerate(path))


class FixedDraws:
    """Feed each possible target stream through the actual verification routine."""
    def __init__(self, draws):
        self.draws = iter(draws)
        self.calls = 0

    def choice(self, vocabulary, p):
        token = next(self.draws)
        if not 0 <= token < vocabulary or p[token] <= 0:
            raise AssertionError("Invalid enumerated draw")
        self.calls += 1
        return token


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool, self.rows = fixture()

    def test_known_selection_values(self):
        self.assertAlmostEqual(expected_acceptance(select_draft(self.pool, 4), self.rows), .428)
        self.assertAlmostEqual(expected_acceptance(select_root(self.pool, self.rows[()], 4), self.rows), 1.824)
        self.assertAlmostEqual(expected_root_draw_acceptance(self.pool, self.rows, 4), 2.257)

    def test_root_factor_replacement(self):
        expected = ((1,), (1, 0), (1, 0, 0), (0,))
        self.assertEqual(select_root(self.pool, self.rows[()], 4), expected)
        self.assertEqual(select_root(self.pool, [.75, .20, .05], 4), select_draft(self.pool, 4))

    def test_root_draw_is_adaptive(self):
        self.assertEqual(select_root_draw(self.pool, 1, 4), ((1,), (1, 0), (1, 0, 0), (1, 1)))
        self.assertEqual(select_preselected(self.pool, 1, 4), ((1,),))
        self.assertEqual(select_root_draw(self.pool, 2, 4), ((2,),))
        self.assertEqual(select_preselected(self.pool, 2, 4), ((2,),))

    def test_root_draw_expectation_by_conditioning(self):
        for budget in range(1, 10):
            value = 0.0
            for first, p in enumerate(self.rows[()]):
                tree = select_root_draw(self.pool, first, budget)
                conditional = 1 + sum(path_mass(path, self.rows) / p for path in tree if len(path) > 1)
                value += p * conditional
            self.assertAlmostEqual(value, expected_root_draw_acceptance(self.pool, self.rows, budget))

    def test_oracle_against_all_prefix_closed_subsets(self):
        for budget in range(len(self.pool.paths) + 1):
            best = 0.0
            for subset in itertools.combinations(self.pool.paths, budget):
                if all(len(p) == 1 or p[:-1] in subset for p in subset):
                    best = max(best, sum(path_mass(p, self.rows) for p in subset))
            selected = select_oracle(self.pool, self.rows, budget)
            self.assertAlmostEqual(expected_acceptance(selected, self.rows), best)

    def test_prefix_closure_with_ties_and_zero_target_mass(self):
        pool = CandidatePool(((0,), (1,), (0, 0), (1, 0)), (0., 0., 0., 0.))
        for root in ([1., 0.], [0., 1.], [.5, .5]):
            for budget in range(6):
                tree = select_root(pool, root, budget)
                self.assertTrue(all(len(p) == 1 or p[:-1] in tree for p in tree))
        self.assertEqual(select_root(pool, [.5, .5], 4), pool.paths)

    def test_zero_and_oversized_budgets(self):
        self.assertEqual(select_draft(self.pool, 0), ())
        self.assertEqual(len(select_draft(self.pool, 999)), len(self.pool.paths))
        self.assertEqual(expected_root_draw_acceptance(self.pool, self.rows, 1), 1.)
        with self.assertRaises(ValueError):
            select_root_draw(self.pool, 0, 0)

    def test_external_draft_rank_is_preserved(self):
        pool = CandidatePool(((0,), (1,), (0, 0)), (-.2, -.5, -.8), (1, 0, 2))
        self.assertEqual(select_draft(pool, 2), ((1,), (0,)))

    def test_invalid_pool(self):
        for paths, scores, ranks in [
            (((0, 1),), (-1.,), None),
            (((0,), (0,)), (-1., -1.), None),
            (((0,), (0, 1)), (-1., -.5), None),
            (((0,),), (-np.inf,), None),
            (((0,),), (.1,), None),
            (((0,), (0, 1)), (-1., -2.), (1, 0)),
            (((0,),), (-1.,), (2,)),
            (((False,),), (-1.,), None),
        ]:
            with self.subTest(paths=paths, scores=scores, ranks=ranks), self.assertRaises(ValueError):
                CandidatePool(paths, scores, ranks)

    def test_invalid_rows_and_budget(self):
        for row in ([.2, .2], [-.1, 1.1], [np.nan, 0.], [[.5, .5]]):
            with self.assertRaises(ValueError):
                select_root(self.pool, row, 4)
        with self.assertRaises(ValueError):
            select_draft(self.pool, -1)
        with self.assertRaises(ValueError):
            select_draft(self.pool, 2.5)
        with self.assertRaises(ValueError):
            expected_acceptance(((0,), (0, 0)), {(): [1.]})


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tree = ((0,), (1,), (0, 0), (0, 1))
        self.rows = {(): [.4, .6], (0,): [.7, .3], (1,): [.2, .8],
                     (0, 0): [.1, .9], (0, 1): [.6, .4]}

    def enumerate_rounds(self, stop_at_leaf=False, extra=False):
        rows = dict(self.rows)
        if extra:
            rows[(1, 0)] = [.5, .5]
        outputs = {}
        acceptance = coverage = 0.0
        # Three draws cover every possible exit from this depth-two tree.
        for stream in itertools.product(range(2), repeat=3):
            result = verify_tree(self.tree, rows, FixedDraws(stream), stop_at_leaf=stop_at_leaf)
            outputs[result.emitted] = result
        probability = 0.0
        for emitted, result in outputs.items():
            mass = path_mass(emitted, rows)
            probability += mass
            acceptance += mass * len(result.accepted)
            coverage += mass * result.covered
        return probability, acceptance, coverage, outputs

    def test_exhaustive_acceptance_and_no_reusable_exit(self):
        probability, acceptance, coverage, _ = self.enumerate_rounds()
        self.assertAlmostEqual(probability, 1.)
        self.assertAlmostEqual(acceptance, expected_acceptance(self.tree, self.rows))
        self.assertEqual(coverage, 0.)

    def test_leaf_stop_coverage_equals_acceptance_cost(self):
        probability, acceptance, coverage, _ = self.enumerate_rounds(stop_at_leaf=True)
        self.assertAlmostEqual(probability, 1.)
        self.assertAlmostEqual(expected_acceptance(self.tree, self.rows) - acceptance, coverage)
        self.assertAlmostEqual(coverage, 1.)

    def test_extra_scored_frontier_gives_coverage(self):
        probability, acceptance, coverage, _ = self.enumerate_rounds(extra=True)
        self.assertAlmostEqual(probability, 1.)
        self.assertAlmostEqual(acceptance, expected_acceptance(self.tree, self.rows))
        self.assertAlmostEqual(coverage, .6 * .2)

    def test_output_distribution_matches_autoregressive_prefixes(self):
        # Finish each variable-length round with ordinary target draws, then compare
        # every fixed-length output probability to direct autoregressive generation.
        for leaf_stop in (False, True):
            _, _, _, outcomes = self.enumerate_rounds(stop_at_leaf=leaf_stop)
            rows = dict(self.rows)
            rows[(1, 0)], rows[(1, 1)] = [.5, .5], [.3, .7]
            for stream in itertools.product(range(2), repeat=3):
                matching = [p for p in outcomes if stream[:len(p)] == p]
                self.assertEqual(len(matching), 1)
                prefix = matching[0]
                probability = path_mass(prefix, rows)
                for i in range(len(prefix), 3):
                    probability *= rows[stream[:i]][stream[i]]
                self.assertAlmostEqual(probability, path_mass(stream, rows))

    def test_supplied_first_draw_is_not_resampled(self):
        rng = FixedDraws((0, 1))
        result = verify_tree(self.tree, self.rows, rng, first_token=0)
        self.assertEqual(result.emitted, (0, 0, 1))
        self.assertEqual(rng.calls, 2)

    def test_first_draw_outside_pool_is_verified_alone(self):
        pool, rows = fixture()
        tree = select_root_draw(pool, 2, 4)
        result = verify_tree(tree, {(): rows[()], (2,): rows[(2,)]}, FixedDraws((1,)), first_token=2)
        self.assertEqual(result.accepted, (2,))
        self.assertEqual(result.continuation, 1)
        self.assertFalse(result.covered)

    def test_empty_tree_emits_one_target_token(self):
        result = verify_tree((), {(): [.4, .6]}, FixedDraws((1,)))
        self.assertEqual(result.emitted, (1,))
        self.assertFalse(result.covered)

    def test_missing_row_and_impossible_first_draw(self):
        with self.assertRaises(ValueError):
            verify_tree(((0,),), {(): [1.]}, FixedDraws(()))
        with self.assertRaises(ValueError):
            verify_tree((), {(): [1., 0.]}, FixedDraws(()), first_token=1)


class DistributionAndCacheTests(unittest.TestCase):
    def test_temperature_zero_and_stable_softmax(self):
        np.testing.assert_array_equal(distribution([2., 4., 1.], 0), [0., 1., 0.])
        np.testing.assert_allclose(distribution([1000., 1001.]), distribution([0., 1.]))
        np.testing.assert_allclose(distribution([0., 2.], 2), distribution([0., 1.]))
        with self.assertRaises(ValueError):
            distribution([1., 2.], -1)

    def test_full_context_not_final_token(self):
        cache = ExactRowCache()
        cache.put_pass((4, 5), {(): [.3, .7], (1,): [.8, .2]})
        np.testing.assert_allclose(cache.get((4, 5, 1)), [.8, .2])
        self.assertIsNone(cache.get((5, 1)))
        self.assertIsNone(cache.get((9, 5, 1)))

    def test_cache_isolation(self):
        cache = ExactRowCache()
        row = np.array([.3, .7])
        cache.put((1,), row)
        row[:] = [1., 0.]
        retrieved = cache.get((1,))
        retrieved[:] = [0., 1.]
        np.testing.assert_allclose(cache.get((1,)), [.3, .7])
        self.assertIsNone(ExactRowCache().get((1,)))
        cache.clear()
        self.assertIsNone(cache.get((1,)))


if __name__ == "__main__":
    unittest.main()
