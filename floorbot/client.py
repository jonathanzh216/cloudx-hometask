"""HTTP/SSE client for the CloudX floor-pricing simulator.

Split into a stream reader and a floor submitter on purpose: a naive
"read one bid request, submit its floor, read the next" loop cannot keep up.
The stream pushes ~10 events/sec and the server's 500ms latency budget is
measured from when *it* emitted the bid request, not from when our POST call
starts - so any time spent blocked on a previous request's HTTP round trip
is time stolen from every request queued up behind it. Measured live against
the real simulator: a sequential loop reading several events before responding
to the first pushed observed latency past 500-800ms and got responses back
marked `late`. The fix is to decouple reading from submitting: one thread does
nothing but read the SSE stream and enqueue, a pool of workers drain the queue
and POST concurrently (see runner.py).

Other behavior confirmed live against https://ml-interview.fly.dev and baked
in here:
  - Body validation happens before the timing check: a negative or wrong-typed
    `floor` gets a 400 regardless of lateness. A *missing* `floor` field is
    NOT rejected (silently defaults, unclear to what) - so `submit_floor`
    requires a floor argument, there is no way to call it without one.
  - Exactly one submission is accepted per request_id; a second attempt for
    the same id returns 404 ("not found, expired, already submitted, or
    belongs to another candidate"). This client never retries a submission
    once sent - retrying can only ever turn a real answer into a wasted,
    guaranteed-404 request against the rate budget.
  - A late submission still returns 200 with late:true/filled:false/revenue:0,
    even many seconds after the deadline (observed at 8s delay). There is no
    separate "expired" error to handle for lateness alone.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import requests


class RateLimiter:
    """Token bucket, used as a safety net under the simulator's 20 req/s/session cap.
    Natural stream throughput (~10/s, one submission per bid request) stays well
    under this, so in normal operation this should never block.
    """

    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.005)


@dataclass
class BidRequest:
    data: dict
    received_at: float  # time.monotonic() when we saw it, used for our own queue-wait metric


class SimulatorClient:
    def __init__(self, base_url: str, candidate_key: str, connect_timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.candidate_key = candidate_key
        self.connect_timeout = connect_timeout
        self.http = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

    def stream_bid_requests(self, stop_event: threading.Event, on_session=None):
        """Generator yielding `BidRequest`s from the SSE stream. Reconnects with
        backoff on transient failures; stops cleanly when `stop_event` is set."""
        backoff = 1.0
        while not stop_event.is_set():
            try:
                resp = self.http.get(
                    f"{self.base_url}/bidreqs",
                    stream=True,
                    timeout=(self.connect_timeout, None),
                    params={"candidate_key": self.candidate_key},
                    headers={"Accept": "text/event-stream"},
                )
                resp.raise_for_status()
                backoff = 1.0
                event = None
                for raw in resp.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        break
                    if not raw:
                        continue
                    if raw.startswith("event:"):
                        event = raw.split(":", 1)[1].strip()
                    elif raw.startswith("data:"):
                        payload = json.loads(raw.split(":", 1)[1].strip())
                        if event == "session" and on_session:
                            on_session(payload)
                        elif event == "bidreq":
                            yield BidRequest(data=payload, received_at=time.monotonic())
                resp.close()
            except (requests.RequestException, json.JSONDecodeError):
                if stop_event.is_set():
                    return
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    def submit_floor(self, request_id: str, floor: float, timeout: float = 3.0) -> dict:
        """Submits a floor for a request_id. `floor` must be an explicit
        non-negative number - see module docstring for why this is not optional.
        Never call this twice for the same request_id.
        """
        if floor < 0:
            raise ValueError(f"floor must be non-negative, got {floor}")
        try:
            resp = self.http.post(
                f"{self.base_url}/floor",
                json={
                    "candidate_key": self.candidate_key,
                    "request_id": request_id,
                    "floor": float(floor),
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                return {"error": "rate_limited", "filled": False, "revenue": 0, "late": False}
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "filled": False, "revenue": 0, "late": False}
