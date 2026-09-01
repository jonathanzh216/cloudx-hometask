"""Phase 5 step 2: the anchoring check.

`floorbot/valuation.py`'s baseline estimator assumes bidders don't change what
they bid in response to the floor we set (no anchoring) - under that
assumption, expected revenue is mathematically guaranteed to be highest at the
lowest floor (see that file's docstring, and the regression test in
test_valuation.py). The Phase 4 dry run's flat-prior bandit instead converged
heavily onto floor=13, well above the lowest grid point - which is either
early Thompson-sampling noise amplified by hierarchical shrinkage, or real
evidence the assumption is wrong for this simulator.

This script tells them apart with a controlled comparison instead of guessing:
it deterministically rotates through a fixed set of *other* floors (skipping
the baseline floor, which is already covered by collect_baseline.py) so every
probed floor gets a comparable, context-mixed sample, then reports realized
average revenue per (ad_unit, floor) against what the baseline distribution -
recorded independently, before any of these floors were tried - predicts for
that same floor. Deliberately grouped by ad_unit only (not the full
hierarchy): the comparison needs to be legible and independent of the bandit's
own shrinkage machinery, since that machinery is one of the things under
suspicion.
"""
import argparse
import itertools
import json
import statistics
import sys
import threading
import time
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from floorbot.client import SimulatorClient
from floorbot.config import BASE_URL, get_candidate_key
from floorbot.policy import FLOOR_GRID
from floorbot.runner import SessionRunner

PROBE_FLOORS = [2, 6, 13, 22, 46]  # spread across the grid; skips FLOOR_GRID[0], already covered by the baseline


class ThreadSafeRoundRobin:
    def __init__(self, values):
        self._cycle = itertools.cycle(values)
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            return next(self._cycle)


def load_ad_unit_revenue(path: str, floor_filter=None) -> dict:
    """ad_unit -> list of realized revenue values, optionally restricted to a
    specific floor (used for the baseline log, always at FLOOR_GRID[0])."""
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            floor_used = record.get("floor_set")
            context = record.get("context")
            result = record.get("result") or {}
            if context is None or floor_used is None:
                continue
            if floor_filter is not None and floor_used != floor_filter:
                continue
            out[context["ad_unit"]].append((floor_used, float(result.get("revenue") or 0.0)))
    return out


def predicted_revenue_at_floor(baseline_observations, floor: float) -> float | None:
    """mean(revenue * 1{revenue >= floor}) over baseline (near-zero-floor)
    observations - the same estimator as ValuationModel, applied here directly
    and un-shrunk, deliberately, for transparency."""
    if not baseline_observations:
        return None
    revenues = [rev for _, rev in baseline_observations]
    return sum(r for r in revenues if r >= floor) / len(revenues)


def run_probe(duration, log_path, candidate_key):
    client = SimulatorClient(BASE_URL, get_candidate_key(candidate_key))
    rr = ThreadSafeRoundRobin(PROBE_FLOORS)
    print(f"[anchoring] logging to {log_path}, rotating floors {PROBE_FLOORS}", file=sys.stderr)
    runner = SessionRunner(client, lambda ctx: rr.next(), log_path)
    runner.run(duration_s=duration)


def analyze(baseline_log: str, probe_log: str):
    baseline_by_ad_unit = load_ad_unit_revenue(baseline_log, floor_filter=FLOOR_GRID[0])
    probe_by_ad_unit = load_ad_unit_revenue(probe_log)

    print("\n=== Anchoring check: predicted (from baseline) vs realized (from probe) ===")
    for ad_unit, baseline_obs in sorted(baseline_by_ad_unit.items()):
        print(f"\nad_unit={ad_unit} (baseline n={len(baseline_obs)})")
        print(f"  {'floor':>6} {'predicted':>10} {'realized':>10} {'n':>5} {'diff':>8}")
        probe_obs = probe_by_ad_unit.get(ad_unit, [])
        for floor in PROBE_FLOORS:
            realized_vals = [rev for f, rev in probe_obs if f == floor]
            predicted = predicted_revenue_at_floor(baseline_obs, floor)
            if predicted is None or not realized_vals:
                continue
            realized = statistics.mean(realized_vals)
            print(
                f"  {floor:6.1f} {predicted:10.3f} {realized:10.3f} {len(realized_vals):5d} "
                f"{realized - predicted:+8.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--baseline-log", required=True, help="output of collect_baseline.py")
    parser.add_argument("--log", default=None)
    parser.add_argument("--candidate-key", default=None)
    parser.add_argument("--analyze-only", action="store_true", help="skip the live probe, just re-analyze --log")
    args = parser.parse_args()

    log_path = args.log or f"data/anchoring_probe_{int(time.time())}.jsonl"
    if not args.analyze_only:
        run_probe(args.duration, log_path, args.candidate_key)
    analyze(args.baseline_log, log_path)


if __name__ == "__main__":
    main()
