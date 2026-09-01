"""Phase 1 verification: exercise SimulatorClient directly against the live
simulator (SSE read + floor submit), independent of any strategy code. Reads a
handful of bid requests, submits a fixed low floor for each, and prints the
raw responses. Not part of the submitted strategy - just a wiring check.
"""
import argparse
import sys
import threading

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from floorbot.client import SimulatorClient
from floorbot.config import BASE_URL, get_candidate_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--floor", type=float, default=1.0)
    parser.add_argument("--candidate-key", default=None)
    args = parser.parse_args()

    client = SimulatorClient(BASE_URL, get_candidate_key(args.candidate_key))
    stop_event = threading.Event()

    def on_session(payload):
        print("SESSION", payload)

    seen = 0
    for bidreq in client.stream_bid_requests(stop_event, on_session=on_session):
        result = client.submit_floor(bidreq.data["id"], args.floor)
        print("BIDREQ", bidreq.data)
        print("RESULT", result)
        seen += 1
        if seen >= args.count:
            stop_event.set()
            break


if __name__ == "__main__":
    main()
