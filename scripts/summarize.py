"""Computes the final metrics exactly as the task defines them, directly from
a run's JSONL log: average revenue per bid request over *every* request the
stream sent (filled, unfilled, late, or unanswered all count as 0 unless
filled), fill rate, late rate, total revenue, and the session_id(s) involved.
"""
import argparse
import json


def summarize(path: str) -> dict:
    n = 0
    total_revenue = 0.0
    filled = 0
    late = 0
    session_ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n += 1
            result = record.get("result") or {}
            total_revenue += float(result.get("revenue") or 0.0)
            filled += int(bool(result.get("filled", False)))
            late += int(bool(result.get("late", False)))
            sid = record.get("session_id")
            if sid:
                session_ids.add(sid)
    return {
        "n": n,
        "avg_revenue_per_request": total_revenue / n if n else 0.0,
        "fill_rate": filled / n if n else 0.0,
        "late_rate": late / n if n else 0.0,
        "total_revenue": total_revenue,
        "session_ids": sorted(session_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="one or more run JSONL log files")
    args = parser.parse_args()
    for path in args.logs:
        m = summarize(path)
        print(f"\n=== {path} ===")
        print(f"  session_id(s):    {', '.join(m['session_ids']) or '(none)'}")
        print(f"  n (bid requests): {m['n']}")
        print(f"  avg revenue/req:  {m['avg_revenue_per_request']:.4f}")
        print(f"  fill rate:        {m['fill_rate']:.3%}")
        print(f"  late rate:        {m['late_rate']:.3%}")
        print(f"  total revenue:    {m['total_revenue']:.2f}")


if __name__ == "__main__":
    main()
