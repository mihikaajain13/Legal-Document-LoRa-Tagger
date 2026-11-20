import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("PAYMENT", "CONFIDENTIALITY", "TERMINATION", "AUDIT")

def records(split):
    rows = (ROOT / "examples/clauses.jsonl").read_text(encoding="utf-8-sig").splitlines()
    return [row for row in map(json.loads, rows) if row["split"] == split]
