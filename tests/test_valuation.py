"""Pure, no-network unit tests for the baseline valuation estimator.

The second test (`test_best_floor_is_lowest_under_no_anchoring`) is a
deliberate regression guard: the first draft of this project's plan proposed
`expected_revenue(floor) = floor * fill_rate(floor)`, which is simply wrong
(revenue is the winning bid, not the floor) and would *not* reliably recover
the lowest floor here - it trades off a rising floor against a falling fill
rate and can land somewhere in the middle. The corrected estimator has no such
free parameter to get wrong: under the no-anchoring assumption, expected
revenue is mathematically non-increasing in the floor (raising the floor only
ever discards positive-value mass), so the true optimum is always the lowest
floor tried. If this test ever fails, that mathematical fact - not the test -
should be the first thing to doubt.
"""
import random
import unittest

from floorbot.policy import CONTEXT_KEY_LEVELS, FLOOR_GRID, build_context
from floorbot.valuation import ValuationModel


def make_model(shrink_k: float = 8.0) -> ValuationModel:
    return ValuationModel(low_floor=FLOOR_GRID[0], key_levels=CONTEXT_KEY_LEVELS, shrink_k=shrink_k)


SOME_CONTEXT = build_context(
    {"ad_unit": "mrec", "device_os": "ios", "country": "US", "session_duration_seconds": 100}
)


class TestRevenueCurveAccuracy(unittest.TestCase):
    """Fit against synthetic log-normal bids with a known distribution and check
    the estimated curve tracks the true population expectation."""

    def setUp(self):
        random.seed(0)
        self.mu, self.sigma = 2.0, 1.0  # log-normal bid distribution
        self.model = make_model()
        for _ in range(20000):
            true_bid = random.lognormvariate(self.mu, self.sigma)
            observed_revenue = true_bid if true_bid >= FLOOR_GRID[0] else 0.0
            self.model.ingest(SOME_CONTEXT, floor_used=FLOOR_GRID[0], revenue=observed_revenue)

    def true_expected_revenue(self, floor: float, n: int = 200000) -> float:
        random.seed(1)  # independent-in-spirit fixed seed, large N for a stable reference
        total = 0.0
        for _ in range(n):
            x = random.lognormvariate(self.mu, self.sigma)
            if x >= floor:
                total += x
        return total / n

    def test_curve_tracks_true_distribution(self):
        curve = self.model.revenue_curve(SOME_CONTEXT, FLOOR_GRID)
        for floor in [0.5, 2, 6, 17, 36]:
            estimated = curve[floor]
            true_value = self.true_expected_revenue(floor)
            self.assertAlmostEqual(
                estimated, true_value, delta=0.15 * true_value + 0.5,
                msg=f"floor={floor}: estimated={estimated:.3f} true={true_value:.3f}",
            )

    def test_best_floor_is_lowest_under_no_anchoring(self):
        best_floor, _ = self.model.best_floor(SOME_CONTEXT, FLOOR_GRID)
        self.assertEqual(best_floor, FLOOR_GRID[0])


class TestHierarchicalFallback(unittest.TestCase):
    def setUp(self):
        random.seed(2)
        self.model = make_model(shrink_k=8.0)

    def _ingest_n(self, context, n, mu, sigma):
        for _ in range(n):
            true_bid = random.lognormvariate(mu, sigma)
            revenue = true_bid if true_bid >= FLOOR_GRID[0] else 0.0
            self.model.ingest(context, floor_used=FLOOR_GRID[0], revenue=revenue)

    def test_sparse_context_falls_back_to_coarse_aggregate(self):
        # Lots of data for the ad_unit-level aggregate, via many different fine
        # contexts sharing ad_unit="banner", but the specific fine context under
        # test never gets its own observation.
        rich_mu, rich_sigma = 1.0, 0.5
        for country in ["US", "GB", "DE", "CA"]:
            ctx = build_context(
                {"ad_unit": "banner", "device_os": "android", "country": country,
                 "session_duration_seconds": 100}
            )
            self._ingest_n(ctx, 500, rich_mu, rich_sigma)

        never_seen_context = build_context(
            {"ad_unit": "banner", "device_os": "ios", "country": "AU",
             "session_duration_seconds": 100}
        )
        curve = self.model.revenue_curve(never_seen_context, FLOOR_GRID)
        coarse_curve = self.model.revenue_curve(
            build_context({"ad_unit": "banner", "device_os": "android", "country": "US",
                            "session_duration_seconds": 100}),
            FLOOR_GRID,
        )
        # The unseen fine context should land close to the well-observed coarse
        # aggregate (same ad_unit), not at the zero-data global default.
        for floor in [FLOOR_GRID[0], 2, 6]:
            self.assertGreater(curve[floor], 0.0)
            self.assertAlmostEqual(curve[floor], coarse_curve[floor], delta=0.25 * coarse_curve[floor] + 0.2)

    def test_rich_fine_context_deviates_from_coarse_aggregate(self):
        # ad_unit-level aggregate dominated by low-value banner traffic...
        for country in ["US", "GB", "DE"]:
            ctx = build_context(
                {"ad_unit": "mrec", "device_os": "android", "country": country,
                 "session_duration_seconds": 100}
            )
            self._ingest_n(ctx, 300, mu=1.0, sigma=0.3)

        # ...but this specific fine context has plenty of its own, much higher, data.
        rich_context = build_context(
            {"ad_unit": "mrec", "device_os": "ios", "country": "US", "session_duration_seconds": 100}
        )
        self._ingest_n(rich_context, 3000, mu=3.0, sigma=0.3)

        rich_curve = self.model.revenue_curve(rich_context, FLOOR_GRID)
        coarse_only_context = build_context(
            {"ad_unit": "mrec", "device_os": "android", "country": "US", "session_duration_seconds": 100}
        )
        coarse_curve = self.model.revenue_curve(coarse_only_context, FLOOR_GRID)
        self.assertGreater(rich_curve[FLOOR_GRID[0]], coarse_curve[FLOOR_GRID[0]] * 2)


if __name__ == "__main__":
    unittest.main()
