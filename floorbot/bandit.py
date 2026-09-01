"""Hierarchical Thompson sampling over (context, floor) arms, warm-started from
`valuation.ValuationModel` instead of a flat/uninformative prior.

Why warm-start: with ~20 contexts splitting ~10 requests/sec across a 16-point
floor grid, a flat-prior bandit would spend a long time re-discovering, arm by
arm, what valuation.py can already estimate in one pass from near-zero-floor
data (see valuation.py's docstring). Treating that estimate as a prior - not a
fixed decision - means it accelerates convergence when it's right and gets
overridden by live evidence when it's wrong (e.g. if floors *do* turn out to
shift bidder behavior, contradicting the no-anchoring assumption it relies on;
see test_bandit.py's `test_corrects_a_wrong_prior_with_enough_live_data`).

How much to trust the prior: an earlier version of this file gave the prior a
flat pseudo-count of `shrink_k` (8) regardless of how much data actually built
it - which meant a curve estimated from 20,000 real observations got the same
say as one estimated from 8. Concretely, with `default_var=400` (a realistic
per-observation variance given the heavy-tailed bid distribution observed
live), 8 pseudo-observations implies a mean standard error of
sqrt(400/8) ≈ 7.1 - larger than the entire spread of the curve across the
floor grid (~1.9 to ~12.3 in one measurement) - so Thompson-sampling noise
completely swamped the prior and a live regret comparison test showed no
benefit from warm-starting at all. Fixed by using
`ValuationModel.sample_size(context)`, capped at `prior_max_pseudo_n`, as the
prior's pseudo-count instead: a context backed by thousands of real
observations anchors the bandit firmly; a context with only a handful is
appropriately easy for live data to override. The cap keeps the prior
falsifiable - even a very well-observed prior can't become permanently immune
to contradicting live evidence, which matters most for the case where the
no-anchoring assumption is wrong for that context.
"""
from __future__ import annotations

import math
import random
import threading
from collections import defaultdict
from typing import Callable, Sequence

from .valuation import ValuationModel

ContextKeyFn = Callable[[dict], tuple]


class ArmStats:
    """Online mean/variance for one (context-level, floor) arm via Welford's algorithm."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def var(self):
        return self.m2 / (self.n - 1) if self.n > 1 else None


class HierarchicalThompsonBandit:
    def __init__(
        self,
        floor_grid: Sequence[float],
        key_levels: Sequence[ContextKeyFn],
        valuation_model: ValuationModel,
        shrink_k: float = 8.0,
        default_var: float = 400.0,
        prior_max_pseudo_n: float = 50.0,
        seed: int | None = None,
    ):
        self.floor_grid = list(floor_grid)
        self.key_levels = list(key_levels)
        self.valuation_model = valuation_model
        self.shrink_k = shrink_k
        self.default_var = default_var
        self.prior_max_pseudo_n = prior_max_pseudo_n
        # stats[level_index][group_key][floor] -> ArmStats
        self.stats: list[dict] = [defaultdict(dict) for _ in self.key_levels]
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    def _group_keys(self, context: dict) -> list[tuple]:
        return [level(context) for level in self.key_levels]

    def _shrunk_posterior(self, group_keys: list[tuple], context: dict, floor: float, prior_mean: float):
        prior_n = min(self.valuation_model.sample_size(context), self.prior_max_pseudo_n)
        mean, var, eff_n = prior_mean, self.default_var, max(prior_n, 1.0)
        for level_idx, key in enumerate(group_keys):
            arm = self.stats[level_idx].get(key, {}).get(floor)
            if arm is None or arm.n == 0:
                continue
            level_var = arm.var if arm.var is not None else var
            mean = (self.shrink_k * mean + arm.n * arm.mean) / (self.shrink_k + arm.n)
            eff_n = self.shrink_k + arm.n
            var = level_var
        return mean, var, eff_n

    def choose_floor(self, context: dict) -> float:
        group_keys = self._group_keys(context)
        prior_curve = self.valuation_model.revenue_curve(context, self.floor_grid)
        best_floor, best_sample = self.floor_grid[0], -math.inf
        with self._lock:
            for floor in self.floor_grid:
                mean, var, eff_n = self._shrunk_posterior(group_keys, context, floor, prior_curve[floor])
                std = math.sqrt(max(var, 1e-6) / max(eff_n, 1.0))
                sample = self._rng.gauss(mean, std)
                if sample > best_sample:
                    best_sample, best_floor = sample, floor
        return best_floor

    def update(self, context: dict, floor: float, reward: float) -> None:
        group_keys = self._group_keys(context)
        with self._lock:
            for level_idx, key in enumerate(group_keys):
                level_stats = self.stats[level_idx][key]
                arm = level_stats.get(floor)
                if arm is None:
                    arm = ArmStats()
                    level_stats[floor] = arm
                arm.update(reward)

    def best_floor_estimate(self, context: dict) -> tuple[float, float]:
        """Point estimate (no sampling) of the best floor for a context - for reporting."""
        group_keys = self._group_keys(context)
        prior_curve = self.valuation_model.revenue_curve(context, self.floor_grid)
        best_floor, best_mean = self.floor_grid[0], -math.inf
        for floor in self.floor_grid:
            mean, _, _ = self._shrunk_posterior(group_keys, context, floor, prior_curve[floor])
            if mean > best_mean:
                best_mean, best_floor = mean, floor
        return best_floor, best_mean
