"""Rebuilds a ValuationModel from previously logged JSONL runs, so a fresh
session doesn't start every context's prior from zero low-floor observations.
"""
from __future__ import annotations

import json

from .valuation import ValuationModel


def load_valuation_model_from_logs(model: ValuationModel, paths) -> int:
    """Ingests every usable record across `paths` into `model`. Returns the
    count of records actually taken at the model's low_floor - everything
    else (higher floors, unanswered requests) is skipped by `ingest` itself,
    this just reports how much actually counted.
    """
    used = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                context = record.get("context")
                floor_used = record.get("floor_set")
                result = record.get("result") or {}
                if context is None or floor_used is None or floor_used != model.low_floor:
                    continue
                revenue = float(result.get("revenue") or 0.0)
                model.ingest(context, floor_used=floor_used, revenue=revenue)
                used += 1
    return used
