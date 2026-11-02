"""Generate entity-disjoint train / held-out test sets of synthetic financial documents.

    python scripts/generate_data.py --train-per-type 150 --test-per-type 55
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docusight.synth import generate  # noqa: E402


def _entities(records: list[dict]) -> set[str]:
    keys = ("account_holder", "employee_name", "name", "father_name", "vendor_name")
    return {r["fields"][k] for r in records for k in keys if r["fields"].get(k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-per-type", type=int, default=150)
    ap.add_argument("--test-per-type", type=int, default=55)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out = Path(args.out)
    train = generate(out / "train", args.train_per_type, "train", seed=args.seed)
    test = generate(out / "test", args.test_per_type, "test", seed=args.seed + 1000)

    # Leakage audit: no shared entity strings and no duplicate page text across splits.
    shared_entities = _entities(train) & _entities(test)
    h = lambda r: hashlib.sha1(r["text"].encode()).hexdigest()  # noqa: E731
    shared_text = {h(r) for r in train} & {h(r) for r in test}
    report = {
        "train_docs": len(train),
        "test_docs": len(test),
        "shared_entities": sorted(shared_entities),
        "duplicate_pages_across_splits": len(shared_text),
        "test_anomalies": {k: sum(r["anomaly"] == k for r in test) for k in ("inconsistent", "missing")},
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if shared_entities or shared_text:
        raise SystemExit("Leakage detected between train and test splits")


if __name__ == "__main__":
    main()
