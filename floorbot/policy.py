"""Context features and the floor grid, plus the coarse-to-fine context hierarchy
used for hierarchical shrinkage in both valuation.py and bandit.py.

Feature choices are grounded in a live exploration run against the real
simulator (see WRITEUP.md for the numbers):
- `ad_unit` is the single strongest driver of value (mrec bid roughly 5x what
  banner did, at a near-zero floor, in an unweighted comparison).
- `session_duration_seconds` is negatively correlated with bid value (short/new
  sessions bid higher than long-running ones - reads like a user-acquisition
  pattern, where an unproven user is priced optimistically), so it's bucketed
  into two bins at a 300s cutoff.
- `country` and `device_os` have a visible but smaller and noisier effect, so
  they only enter at the finest context level, where they can still help once
  enough traffic accumulates, but a sparse cell falls back toward the coarser
  ad_unit/duration aggregate instead of overfitting to a handful of samples.
"""
from __future__ import annotations

NEW_SESSION_THRESHOLD_SECONDS = 300

# Log-spaced with extra resolution in the 1-20 range, where the fill/revenue
# tradeoff bent in the initial exploratory sweep. FLOOR_GRID[0] doubles as the
# "near-zero" floor used to collect (near-)uncensored bid samples (valuation.py).
FLOOR_GRID = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 46, 60]


def duration_bucket(session_duration_seconds: float) -> str:
    return "new" if session_duration_seconds < NEW_SESSION_THRESHOLD_SECONDS else "established"


def build_context(bidreq: dict) -> dict:
    return {
        "ad_unit": bidreq["ad_unit"],
        "device_os": bidreq["device_os"],
        "country": bidreq["country"],
        "dur_bucket": duration_bucket(bidreq["session_duration_seconds"]),
    }


# Coarsest first: each level's fit is used as the shrinkage prior for the next.
CONTEXT_KEY_LEVELS = [
    lambda ctx: (),
    lambda ctx: (ctx["ad_unit"],),
    lambda ctx: (ctx["ad_unit"], ctx["dur_bucket"]),
    lambda ctx: (ctx["ad_unit"], ctx["device_os"], ctx["country"], ctx["dur_bucket"]),
]
