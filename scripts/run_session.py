"""Entry point: run a live floor-pricing session against the CloudX simulator.

Example:
    python3 scripts/run_session.py --duration 60
    python3 scripts/run_session.py --duration 600 --warm-start "data/*.jsonl"
"""
import argparse
import glob
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from floorbot.bandit import HierarchicalThompsonBandit
from floorbot.client import SimulatorClient
from floorbot.config import BASE_URL, get_candidate_key
from floorbot.policy import CONTEXT_KEY_LEVELS, FLOOR_GRID
from floorbot.runner import SessionRunner
from floorbot.valuation import ValuationModel
from floorbot.warmstart import load_valuation_model_from_logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=None, help="seconds; omit to run until Ctrl+C")
    parser.add_argument(
        "--warm-start", nargs="*", default=[],
        help="JSONL log file(s) or glob pattern(s) to seed the valuation model's prior from",
    )
    parser.add_argument("--log", default=None, help="output JSONL path (default: data/run_<timestamp>.jsonl)")
    parser.add_argument("--candidate-key", default=None)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--rate-limit-qps", type=float, default=18.0)
    args = parser.parse_args()

    candidate_key = get_candidate_key(args.candidate_key)
    client = SimulatorClient(BASE_URL, candidate_key)

    valuation_model = ValuationModel(low_floor=FLOOR_GRID[0], key_levels=CONTEXT_KEY_LEVELS)
    warm_start_paths = sorted({p for pattern in args.warm_start for p in glob.glob(pattern)})
    if warm_start_paths:
        used = load_valuation_model_from_logs(valuation_model, warm_start_paths)
        print(f"[warm-start] ingested {used} low-floor observations from {len(warm_start_paths)} file(s)", file=sys.stderr)

    bandit = HierarchicalThompsonBandit(FLOOR_GRID, CONTEXT_KEY_LEVELS, valuation_model)

    log_path = args.log or f"data/run_{int(time.time())}.jsonl"
    print(f"[run] logging to {log_path}", file=sys.stderr)
    runner = SessionRunner(
        client, bandit, log_path,
        num_workers=args.num_workers, rate_limit_qps=args.rate_limit_qps,
    )
    runner.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
