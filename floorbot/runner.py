"""Orchestrates a live session: one SSE-reader thread feeding a queue, a pool
of worker threads choosing a floor and submitting it, and a JSONL log of every
bid request - including ones we fail to answer in time - for later scoring.

The reader/worker split exists because a naive "read one bid request, submit
its floor, read the next" loop cannot meet the 500ms SLA: confirmed live
against the real simulator, where a sequential loop pushed observed latency
past 500-800ms on the very first request. The stream doesn't wait for you.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .bandit import HierarchicalThompsonBandit
from .client import BidRequest, RateLimiter, SimulatorClient
from .policy import build_context


class RollingStats:
    def __init__(self):
        self.n = 0
        self.filled = 0
        self.late = 0
        self.total_revenue = 0.0
        self._lock = threading.Lock()

    def record(self, filled: bool, late: bool, revenue: float) -> None:
        with self._lock:
            self.n += 1
            self.filled += int(filled)
            self.late += int(late)
            self.total_revenue += revenue

    def snapshot(self):
        with self._lock:
            avg = self.total_revenue / self.n if self.n else 0.0
            fill_rate = self.filled / self.n if self.n else 0.0
            late_rate = self.late / self.n if self.n else 0.0
            return self.n, avg, fill_rate, late_rate, self.total_revenue


class SessionRunner:
    def __init__(
        self,
        client: SimulatorClient,
        bandit: HierarchicalThompsonBandit,
        log_path: str,
        num_workers: int = 24,
        rate_limit_qps: float = 18.0,
        print_every_s: float = 5.0,
    ):
        self.client = client
        self.bandit = bandit
        self.log_path = log_path
        self.num_workers = num_workers
        self.rate_limiter = RateLimiter(rate_limit_qps)
        self.print_every_s = print_every_s
        self.stats = RollingStats()
        self.session_id = None
        self._log_lock = threading.Lock()
        self._log_file = open(log_path, "a", buffering=1)

    def _on_session(self, payload: dict) -> None:
        self.session_id = payload.get("session_id")
        print(f"[session] {payload}", file=sys.stderr)

    def _log(self, record: dict) -> None:
        with self._log_lock:
            self._log_file.write(json.dumps(record) + "\n")

    def _handle_one(self, item: BidRequest) -> None:
        req = item.data
        context = build_context(req)
        floor = self.bandit.choose_floor(context)

        self.rate_limiter.acquire()
        t0 = time.monotonic()
        result = self.client.submit_floor(req["id"], floor)
        rtt_ms = (time.monotonic() - t0) * 1000

        revenue = float(result.get("revenue") or 0.0)
        filled = bool(result.get("filled", False))
        late = bool(result.get("late", False))

        self.bandit.update(context, floor, revenue)
        self.stats.record(filled=filled, late=late, revenue=revenue)
        self._log(
            {
                **req,
                "context": context,
                "floor_set": floor,
                "result": result,
                "client_rtt_ms": rtt_ms,
                "queue_wait_ms": (time.monotonic() - item.received_at) * 1000 - rtt_ms,
            }
        )

    def run(self, duration_s: float | None = None) -> None:
        stop_event = threading.Event()
        work_q: "queue.Queue[BidRequest]" = queue.Queue()

        def reader():
            for bidreq in self.client.stream_bid_requests(stop_event, on_session=self._on_session):
                work_q.put(bidreq)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        start = time.monotonic()
        last_print = start
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                pending = set()
                while duration_s is None or time.monotonic() - start < duration_s:
                    try:
                        item = work_q.get(timeout=0.5)
                        pending.add(pool.submit(self._handle_one, item))
                    except queue.Empty:
                        pass
                    pending = {f for f in pending if not f.done()}

                    now = time.monotonic()
                    if now - last_print >= self.print_every_s:
                        last_print = now
                        n, avg, fill_rate, late_rate, total = self.stats.snapshot()
                        print(
                            f"[{now - start:6.1f}s] n={n:5d} avg_rev={avg:7.3f} "
                            f"fill={fill_rate:5.1%} late={late_rate:5.1%} total_rev={total:9.2f}",
                            file=sys.stderr,
                        )
                # drain remaining in-flight work before shutting down
                for f in pending:
                    f.result(timeout=5)
        except KeyboardInterrupt:
            print("\n[runner] interrupted, shutting down...", file=sys.stderr)
        finally:
            stop_event.set()
            # anything still sitting in the queue never got a floor submitted:
            # counts as "unanswered" per the task's scoring rule, and is logged
            # as such rather than silently dropped.
            drained = 0
            while True:
                try:
                    item = work_q.get_nowait()
                except queue.Empty:
                    break
                req = item.data
                context = build_context(req)
                self.stats.record(filled=False, late=False, revenue=0.0)
                self._log({**req, "context": context, "floor_set": None, "result": {"unanswered": True}})
                drained += 1
            if drained:
                print(f"[runner] {drained} requests left unanswered at shutdown", file=sys.stderr)
            self._log_file.close()

        n, avg, fill_rate, late_rate, total = self.stats.snapshot()
        print(
            f"\n[final] session={self.session_id} n={n} avg_revenue_per_request={avg:.4f} "
            f"fill_rate={fill_rate:.3%} late_rate={late_rate:.3%} total_revenue={total:.2f}",
            file=sys.stderr,
        )
