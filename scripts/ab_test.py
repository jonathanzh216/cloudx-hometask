"""Live, concurrent head-to-head: the dynamic bandit vs. a fixed floor, run at
the same time against fresh incoming bid requests, so both see the same
conditions in the same window - a cleaner comparison than stitching together
historical logs collected at different times for different purposes (which is
what the previous, underpowered comparison had to do).

The simulator allows multiple concurrent SSE sessions on one candidate key
(confirmed earlier), each with its own session_id and independent 20 req/s
cap - so this runs two full SessionRunners in parallel threads, each with its
own SimulatorClient/connection.
"""
import argparse
import statistics
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from floorbot.bandit import HierarchicalThompsonBandit
from floorbot.client import SimulatorClient
from floorbot.config import BASE_URL, get_candidate_key
from floorbot.policy import CONTEXT_KEY_LEVELS, FLOOR_GRID
from floorbot.runner import SessionRunner
from floorbot.valuation import ValuationModel
from floorbot.warmstart import load_bandit_from_logs, load_valuation_model_from_logs


def summarize(path):
    import json
    n = 0
    total = 0.0
    filled = 0
    late = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            res = r.get("result") or {}
            n += 1
            total += float(res.get("revenue") or 0.0)
            filled += int(bool(res.get("filled", False)))
            late += int(bool(res.get("late", False)))
    mean = total / n if n else 0.0
    # per-request revenue values needed for SEM; re-read for that
    revs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            revs.append(float((r.get("result") or {}).get("revenue") or 0.0))
    sem = statistics.pstdev(revs) / len(revs) ** 0.5 if len(revs) > 1 else float("nan")
    return {"n": n, "mean": mean, "sem": sem, "fill_rate": filled / n if n else 0.0,
            "late_rate": late / n if n else 0.0, "total": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument(
        "--warm-start", nargs="*",
        default=["data/baseline_run.jsonl", "data/anchoring_probe.jsonl", "data/final_run.jsonl"],
        help="historical logs to warm-start the dynamic bandit's arm from (carries forward everything learned)",
    )
    parser.add_argument("--fixed-floor", type=float, default=6.0)
    parser.add_argument("--candidate-key", default=None)
    args = parser.parse_args()

    candidate_key = get_candidate_key(args.candidate_key)

    # Dynamic arm: mature bandit, carrying forward everything learned so far.
    valuation_model = ValuationModel(low_floor=FLOOR_GRID[0], key_levels=CONTEXT_KEY_LEVELS)
    used = load_valuation_model_from_logs(valuation_model, args.warm_start)
    bandit = HierarchicalThompsonBandit(FLOOR_GRID, CONTEXT_KEY_LEVELS, valuation_model)
    seeded = load_bandit_from_logs(bandit, args.warm_start)
    print(f"[ab_test] dynamic arm warm-started: {used} low-floor obs, {seeded} total arm updates", file=sys.stderr)

    def on_result(context, floor, result):
        bandit.update(context, floor, float(result.get("revenue") or 0.0))

    dynamic_client = SimulatorClient(BASE_URL, candidate_key)
    fixed_client = SimulatorClient(BASE_URL, candidate_key)

    dynamic_log = f"data/ab_dynamic_{int(time.time())}.jsonl"
    fixed_log = f"data/ab_fixed{args.fixed_floor}_{int(time.time())}.jsonl"

    dynamic_runner = SessionRunner(dynamic_client, bandit.choose_floor, dynamic_log, on_result=on_result)
    fixed_runner = SessionRunner(fixed_client, lambda ctx: args.fixed_floor, fixed_log)

    print(f"[ab_test] running {args.duration}s concurrently: dynamic -> {dynamic_log}, "
          f"fixed({args.fixed_floor}) -> {fixed_log}", file=sys.stderr)

    t1 = threading.Thread(target=dynamic_runner.run, kwargs={"duration_s": args.duration})
    t2 = threading.Thread(target=fixed_runner.run, kwargs={"duration_s": args.duration})
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    dyn_stats = summarize(dynamic_log)
    fixed_stats = summarize(fixed_log)

    print("\n=== A/B RESULT ===")
    print(f"dynamic : n={dyn_stats['n']:5d} mean={dyn_stats['mean']:.3f} sem={dyn_stats['sem']:.3f} "
          f"fill={dyn_stats['fill_rate']:.1%} late={dyn_stats['late_rate']:.1%} total={dyn_stats['total']:.2f} "
          f"session_id={dynamic_runner.session_id}")
    print(f"fixed={args.fixed_floor}: n={fixed_stats['n']:5d} mean={fixed_stats['mean']:.3f} sem={fixed_stats['sem']:.3f} "
          f"fill={fixed_stats['fill_rate']:.1%} late={fixed_stats['late_rate']:.1%} total={fixed_stats['total']:.2f} "
          f"session_id={fixed_runner.session_id}")

    diff = dyn_stats["mean"] - fixed_stats["mean"]
    se_diff = (dyn_stats["sem"] ** 2 + fixed_stats["sem"] ** 2) ** 0.5
    z = diff / se_diff if se_diff else float("nan")
    print(f"\ndiff (dynamic - fixed) = {diff:+.3f}   pooled SE = {se_diff:.3f}   z = {z:.2f}")


if __name__ == "__main__":
    main()
