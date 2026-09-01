# CloudX floor-pricing take-home

Chooses a floor price for each incoming bid request from the CloudX simulator,
in real time, to maximize average revenue per bid request.

## Setup

Requires Python 3.9+ and the `requests` package (the only third-party
dependency):

```bash
pip install requests
cp .env.example .env   # then edit .env and set CANDIDATE_KEY
```

## Running

Being filled in phase by phase; see `PLAN.md`-equivalent context in the repo
history. A smoke test for the HTTP/SSE client alone:

```bash
python3 scripts/smoke_test_client.py --count 5
```

Full write-up, final run instructions, and results: see `WRITEUP.md` (added at
the end, once there's a final run to report).
