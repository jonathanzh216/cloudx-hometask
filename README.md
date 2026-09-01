# CloudX floor-pricing take-home

Chooses a floor price for each incoming bid request from the CloudX simulator,
in real time, to maximize average revenue per bid request. See `WRITEUP.md`
for the approach, findings, and final results.

## Setup

Requires Python 3.9+ and the `requests` package (the only third-party
dependency):

```bash
pip install requests
cp .env.example .env   # then edit .env and set CANDIDATE_KEY
```

## Running the tests

Pure, no-network unit tests (~90s):

```bash
python3 -m unittest discover -s tests -v
```

## Running the strategy live

```bash
# 1. Build a baseline sample (constant near-zero floor across all contexts).
python3 scripts/collect_baseline.py --duration 180 --log data/baseline_run.jsonl

# 2. Check whether floors actually change bidder behavior (they do - see
#    WRITEUP.md). Probes a fixed rotation of higher floors and compares
#    realized revenue against what the baseline predicts under a
#    no-anchoring assumption.
python3 scripts/anchoring_check.py --duration 120 \
    --baseline-log data/baseline_run.jsonl --log data/anchoring_probe.jsonl

# 3. Run the actual policy, warm-started from both logs above.
python3 scripts/run_session.py --duration 600 \
    --warm-start data/baseline_run.jsonl data/anchoring_probe.jsonl \
    --log data/final_run.jsonl

# 4. Compute the task's defined metrics from any run log(s).
python3 scripts/summarize.py data/final_run.jsonl

# 5. Optional: live head-to-head test of the dynamic bandit vs. a fixed floor,
#    run concurrently so both see identical conditions (see WRITEUP.md).
python3 scripts/ab_test.py --duration 300
```

`run_session.py --duration` may be omitted to run until Ctrl+C. All scripts
accept `--candidate-key` to override the `.env`/environment value.

A lower-level smoke test for the HTTP/SSE client alone, independent of any
strategy code:

```bash
python3 scripts/smoke_test_client.py --count 5
```

## Reported results

See `WRITEUP.md` for the final run's session ID and metrics, and the
reasoning behind the approach.
