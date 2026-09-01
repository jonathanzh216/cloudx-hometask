"""Phase 5 step 1: deliberately probe the lowest grid floor across all
contexts, to build a clean baseline sample for the valuation model's prior
(see floorbot/valuation.py). Unlike a bandit run, every request here uses the
same fixed floor, so there is no confound between "what floor got tried" and
"which context happened to see it" - which is exactly the kind of confound
that made the very first ad-hoc sweep (data/exploration_log.jsonl, referenced
in WRITEUP.md) too noisy to trust.
"""
import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from floorbot.client import SimulatorClient
from floorbot.config import BASE_URL, get_candidate_key
from floorbot.policy import FLOOR_GRID
from floorbot.runner import SessionRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--log", default=None)
    parser.add_argument("--candidate-key", default=None)
    args = parser.parse_args()

    client = SimulatorClient(BASE_URL, get_candidate_key(args.candidate_key))
    log_path = args.log or f"data/baseline_{int(time.time())}.jsonl"
    print(f"[baseline] logging to {log_path}, floor fixed at {FLOOR_GRID[0]}", file=sys.stderr)

    runner = SessionRunner(client, lambda ctx: FLOOR_GRID[0], log_path)
    runner.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
