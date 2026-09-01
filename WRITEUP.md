# CloudX floor-pricing take-home

## Result

Final run — session `sess_68f524fa-451b-4adc-84cc-897b9ed6fca0`, 10 min, 6000 requests:

| avg revenue/request | total revenue | fill rate | late rate |
|---|---|---|---|
| 18.22 | 109,338 | 76.5% | 2.1% |

A constant floor of 0.5 (the lowest allowed) gets 16.67 avg revenue/request over the same traffic mix. A live A/B test (dynamic bandit vs. a fixed floor of 6, run concurrently for 5 minutes so both saw identical conditions) confirmed the adaptive policy really does beat a simple fixed floor: 18.33 vs 16.76, a statistically significant difference (z=3.05). Reproduce with `python3 scripts/summarize.py data/final_run.jsonl` or `python3 scripts/ab_test.py --duration 300`.

## Approach

Each bid request gets a floor chosen by a Thompson-sampling bandit, one arm per (context, floor) pair, seeded with an informed prior instead of starting blind. The prior comes from a batch of near-zero-floor observations: since fill-rate is provably monotone in floor, that one batch tells you the expected revenue at every candidate floor at once, not just the floor you tried. Live data then corrects the prior as the bandit runs.

That correction matters because of the central finding: floors aren't neutral here. I ran a controlled experiment (hold floor near zero to build a baseline, then probe a fixed rotation of higher floors) and found realized revenue at higher floors was consistently 3-16x above what a "bidders don't react to the floor" model would predict. So the naive answer — always use the lowest floor, which is mathematically optimal if bidding is floor-independent — is wrong here, and the whole point of the exercise is picking floors that account for that.

Features: `ad_unit` (mrec bids ~5x banner), a bucketed `session_duration_seconds` (fresh sessions bid higher — reads like new, unproven users get priced optimistically), plus `country`/`device_os` at the finest context level, all folded through a coarse-to-fine hierarchy so sparse contexts borrow from better-observed ones.

## How I evaluated it

Unit tests on synthetic data before ever touching the live API — these caught two real bugs: an early version of the revenue formula used `floor × fill_rate`, which is wrong (revenue is the winning bid, not the floor) and a test proved it picks the wrong answer on synthetic data; and the bandit's prior was initially so underweighted that Thompson-sampling noise swamped it entirely, which a regret test caught. Beyond that: the anchoring experiment above, a live dry run to confirm the concurrency design actually meets the 500ms deadline, and the concurrent A/B test as the final check that the added complexity pays for itself (an earlier attempt to answer that from historical logs alone was underpowered and gave a misleading "maybe not" — worth mentioning since it's the kind of mistake that's easy to publish by accident).

## How I used AI

I used claude code, it probed the live simulator directly to map out real behavior (latency limits, validation edge cases, whether floors affect bidding) rather than guessing, and two of the bugs above were caught by tests failing, not by asking whether code looked right. I reviewed the plan before any implementation started and rejected a first attempt that skipped that step. Every phase was implemented, tested, and committed separately rather than delivered as one opaque dump.

## Time

Roughly: live API exploration and edge-case mapping → framing the statistics (why this is a censored-feedback pricing problem) → a reviewed implementation plan → phase-by-phase build with tests → the anchoring experiment, which is what actually determined the strategy → the final run → the A/B test → this write-up.

## With a month and more data

- Real function approximation (e.g. LinUCB) instead of bucketed country/duration and a fixed floor grid.
- Model the five bidders individually — the anchoring effect may only come from some of them, and pooling hides that.
- Handle non-stationarity: if bidder behavior adapts to the platform's own floors over time, a sliding-window bandit or change-point detection would catch drift a static prior can't.
- Off-policy evaluation before shipping any change, not just before/after comparison.
- Repeat the A/B test across more time windows before trusting it as permanent.
- Think about bidder trust over weeks, not minutes — an aggressive or volatile floor policy could suppress future participation in ways a single session's revenue can't show.
