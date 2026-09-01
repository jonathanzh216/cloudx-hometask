"""Pure, no-network tests for the warm-started hierarchical bandit.

Three separate claims are made about this design (see bandit.py's docstring);
each gets its own test rather than one vague "it seems to work" check:

1. With no prior (a ValuationModel that never saw data), plain Thompson
   sampling still finds a true optimum that sits in the *interior* of the
   floor grid - i.e. it isn't limited to only ever recommending the lowest
   floor. This scenario (an interior optimum) can only happen if floor
   genuinely affects the reward - see point 3 below.
2. When the environment really does match the no-anchoring assumption (reward
   is `bid * 1{bid >= floor}` for one fixed bid distribution - which is
   mathematically guaranteed to have its optimum at the lowest floor, see
   test_valuation.py), a bandit warm-started from a ValuationModel prior gets
   there with much lower regret than one starting flat.
3. If the prior is *wrong* - built the same no-anchoring way, but the live
   environment actually rewards a higher floor more (simulating real
   floor-dependent bidder behavior) - enough live data overrides the bad prior
   rather than getting stuck recommending it forever.
"""
import math
import random
import unittest

from floorbot.bandit import HierarchicalThompsonBandit
from floorbot.policy import CONTEXT_KEY_LEVELS, FLOOR_GRID, build_context
from floorbot.valuation import ValuationModel

CONTEXT = build_context(
    {"ad_unit": "mrec", "device_os": "ios", "country": "US", "session_duration_seconds": 100}
)


def empty_valuation_model() -> ValuationModel:
    return ValuationModel(low_floor=FLOOR_GRID[0], key_levels=CONTEXT_KEY_LEVELS)


def run_bandit(bandit, reward_fn, rng, n_rounds):
    chosen = []
    for _ in range(n_rounds):
        floor = bandit.choose_floor(CONTEXT)
        reward = max(0.0, reward_fn(floor, rng))
        bandit.update(CONTEXT, floor, reward)
        chosen.append(floor)
    return chosen


class TestConvergesToInteriorOptimum(unittest.TestCase):
    """No prior at all - can plain Thompson sampling still find a floor-dependent
    optimum that isn't just 'the lowest floor'?"""

    def test_finds_interior_peak_with_no_prior(self):
        true_best = 6
        assert true_best in FLOOR_GRID

        def reward_fn(floor, rng):
            # bell-shaped in log(floor), peaking at true_best - an environment
            # only explainable by floor genuinely affecting revenue.
            peak = 12.0
            width = 0.6
            mean = peak * math.exp(-((math.log(floor) - math.log(true_best)) ** 2) / (2 * width**2))
            return rng.gauss(mean, 3.0)

        bandit = HierarchicalThompsonBandit(
            FLOOR_GRID, CONTEXT_KEY_LEVELS, empty_valuation_model(), seed=42
        )
        rng = random.Random(7)
        chosen = run_bandit(bandit, reward_fn, rng, n_rounds=3000)

        last_window = chosen[-300:]
        most_common = max(set(last_window), key=last_window.count)
        self.assertIn(most_common, [5, 6, 8], f"converged to {most_common}, expected near {true_best}")


class TestWarmStartReducesRegret(unittest.TestCase):
    """When the environment matches the no-anchoring assumption, a warm-started
    bandit should waste far fewer early pulls on bad (high) floors than a flat
    one, since its prior already points at the true optimum (the lowest floor).
    """

    def _true_bid_distribution(self, rng):
        return rng.lognormvariate(2.0, 1.0)

    def _reward_fn(self, floor, rng):
        bid = self._true_bid_distribution(rng)
        return bid if bid >= floor else 0.0

    def _warm_valuation_model(self):
        model = empty_valuation_model()
        rng = random.Random(123)
        for _ in range(20000):
            bid = self._true_bid_distribution(rng)
            revenue = bid if bid >= FLOOR_GRID[0] else 0.0
            model.ingest(CONTEXT, floor_used=FLOOR_GRID[0], revenue=revenue)
        return model

    def test_warm_started_bandit_has_lower_early_regret(self):
        # true best floor is FLOOR_GRID[0] by the mathematical fact in valuation.py
        rng_ref = random.Random(999)
        true_best_mean = sum(self._reward_fn(FLOOR_GRID[0], rng_ref) for _ in range(50000)) / 50000

        def regret_for_floors(floors, seed):
            # common random numbers across warm/flat for a fair, lower-variance comparison
            rng = random.Random(seed)
            return sum(true_best_mean - max(0.0, self._reward_fn(f, rng)) for f in floors)

        n_rounds = 150
        n_trials = 5
        warm_model = self._warm_valuation_model()  # built once; identical prior across trials
        warm_regrets, flat_regrets = [], []
        for trial in range(n_trials):
            warm_bandit = HierarchicalThompsonBandit(
                FLOOR_GRID, CONTEXT_KEY_LEVELS, warm_model, seed=trial
            )
            flat_bandit = HierarchicalThompsonBandit(
                FLOOR_GRID, CONTEXT_KEY_LEVELS, empty_valuation_model(), seed=trial
            )
            warm_floors = run_bandit(warm_bandit, self._reward_fn, random.Random(100 + trial), n_rounds)
            flat_floors = run_bandit(flat_bandit, self._reward_fn, random.Random(100 + trial), n_rounds)
            warm_regrets.append(regret_for_floors(warm_floors, seed=2000 + trial))
            flat_regrets.append(regret_for_floors(flat_floors, seed=2000 + trial))

        avg_warm = sum(warm_regrets) / n_trials
        avg_flat = sum(flat_regrets) / n_trials
        self.assertLess(
            avg_warm, avg_flat * 0.85,
            f"avg warm={avg_warm:.1f} avg flat={avg_flat:.1f} "
            f"(warm trials={['%.0f' % r for r in warm_regrets]}, flat trials={['%.0f' % r for r in flat_regrets]})",
        )


class TestCorrectsAWrongPrior(unittest.TestCase):
    """The prior assumes no anchoring (built from a fixed bid distribution), but
    the live environment actually rewards a higher floor more - simulating real
    floor-dependent bidder behavior the prior can't see. Enough live data should
    shift the bandit's choices toward the true optimum instead of leaving it
    stuck on the prior's recommendation.
    """

    def _wrong_prior_model(self):
        model = empty_valuation_model()
        rng = random.Random(11)
        for _ in range(20000):
            bid = rng.lognormvariate(1.0, 0.5)  # low-value baseline -> prior favors low floors
            revenue = bid if bid >= FLOOR_GRID[0] else 0.0
            model.ingest(CONTEXT, floor_used=FLOOR_GRID[0], revenue=revenue)
        return model

    def test_corrects_a_wrong_prior_with_enough_live_data(self):
        true_best = 17  # far from what the low-value prior would recommend
        assert true_best in FLOOR_GRID

        def reward_fn(floor, rng):
            width = 0.5
            mean = 20.0 * math.exp(-((math.log(floor) - math.log(true_best)) ** 2) / (2 * width**2))
            return rng.gauss(mean, 2.0)

        bandit = HierarchicalThompsonBandit(
            FLOOR_GRID, CONTEXT_KEY_LEVELS, self._wrong_prior_model(), seed=3
        )
        rng = random.Random(4)

        early_choices = run_bandit(bandit, reward_fn, rng, n_rounds=50)
        later_choices = run_bandit(bandit, reward_fn, rng, n_rounds=3000)

        early_near_true_best = sum(1 for f in early_choices if f in [13, 17, 22]) / len(early_choices)
        late_window = later_choices[-300:]
        late_near_true_best = sum(1 for f in late_window if f in [13, 17, 22]) / len(late_window)

        self.assertGreater(
            late_near_true_best, early_near_true_best,
            f"early={early_near_true_best:.2f} late={late_near_true_best:.2f}",
        )
        self.assertGreater(late_near_true_best, 0.5)


if __name__ == "__main__":
    unittest.main()
