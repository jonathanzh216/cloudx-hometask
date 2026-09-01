"""Baseline valuation estimate: what would expected revenue be at each
candidate floor, estimated from observations where we already tried the lowest
floor on the grid.

Why this works: whenever we submit `FLOOR_GRID[0]` (near-zero) and the auction
fills, the returned `revenue` *is* the true top bid, unfiltered - so it's a
(near-)uncensored draw from the context's bid distribution. And whenever it
doesn't fill even at that floor, the true bid was below FLOOR_GRID[0], which is
so low it would also fail to clear any higher floor on the grid - so treating
that observation's contribution as exactly 0 for every candidate floor is
*exact*, not an approximation, regardless of what the (tiny, unobserved) true
bid actually was.

That means a single batch of near-zero-floor observations lets you compute
`mean(revenue * 1{revenue >= f})` for *every* candidate floor `f` at once, from
the same sample - far more sample-efficient than trying each floor separately
and waiting to see what happens.

Important caveat, expanded on in WRITEUP.md: this assumes bidders' bids don't
depend on the floor we set (no anchoring/shading). If that assumption is false,
this estimate is systematically wrong at the floors it was never actually
observed at. That's exactly why it's used only as a *prior* for the live
bandit (bandit.py) rather than as a standalone decision rule - and why
Phase 5's anchoring check exists, to test the assumption rather than take it
on faith.

One direct mathematical consequence worth stating plainly: under the
no-anchoring assumption, E[revenue(f)] = integral_{x>=f} x dF(x) is
*mathematically non-increasing in f* for any bid distribution F, because raising
f only ever removes positive-value mass. So if bidders truly don't react to the
floor, the revenue-maximizing floor is always the lowest one available - not
approximately, exactly. The interesting version of this problem only exists if
raising the floor changes what bidders bid. See test_valuation.py, which turns
that fact into a regression test.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

ContextKeyFn = Callable[[dict], tuple]


class ValuationModel:
    """Collects near-zero-floor revenue observations per context (with
    coarse-to-fine hierarchical fallback) and derives an expected-revenue
    curve over an arbitrary floor grid from them.
    """

    def __init__(
        self,
        low_floor: float,
        key_levels: Sequence[ContextKeyFn],
        shrink_k: float = 8.0,
    ):
        self.low_floor = low_floor
        self.key_levels = list(key_levels)
        self.shrink_k = shrink_k
        # observations[level_index][group_key] -> list of revenue values from
        # trials where the floor used was `low_floor`.
        self.observations: list[dict[tuple, list[float]]] = [
            defaultdict(list) for _ in self.key_levels
        ]

    def _group_keys(self, context: dict) -> list[tuple]:
        return [level(context) for level in self.key_levels]

    def ingest(self, context: dict, floor_used: float, revenue: float) -> None:
        """Only observations taken at `low_floor` carry information (see module
        docstring for why); everything else is ignored here."""
        if floor_used != self.low_floor:
            return
        for level_idx, key in enumerate(self._group_keys(context)):
            self.observations[level_idx][key].append(revenue)

    @staticmethod
    def _raw_expected_revenue(sample: list[float], floor: float) -> float | None:
        if not sample:
            return None
        return sum(x for x in sample if x >= floor) / len(sample)

    def revenue_curve(self, context: dict, floor_grid: Sequence[float]) -> dict[float, float]:
        """Expected revenue at each floor in `floor_grid`, cascading coarse-to-fine
        shrinkage the same way bandit.py does: each level's blended estimate
        becomes the prior for the next, finer level, weighted by that level's own
        sample count against `shrink_k` pseudo-observations.
        """
        group_keys = self._group_keys(context)
        curve = {f: 0.0 for f in floor_grid}  # global fallback: no data anywhere -> 0
        for floor in floor_grid:
            blended = 0.0
            for level_idx, key in enumerate(group_keys):
                sample = self.observations[level_idx].get(key, [])
                raw = self._raw_expected_revenue(sample, floor)
                if raw is None:
                    continue
                n = len(sample)
                blended = (self.shrink_k * blended + n * raw) / (self.shrink_k + n)
            curve[floor] = blended
        return curve

    def best_floor(self, context: dict, floor_grid: Sequence[float]) -> tuple[float, float]:
        curve = self.revenue_curve(context, floor_grid)
        best_floor = max(curve, key=curve.get)
        return best_floor, curve[best_floor]
