"""Run the pipeline on one image and render extracted source regions.

Green = trusted field, red = field with a validation issue, blue = table rows.

    python scripts/visualize.py data/test/images/test_invoice_0001.jpg --out reports/overlay.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401
import cv2  # noqa: E402

from docusight.observability import redact  # noqa: E402
from docusight.pipeline import DocumentPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="reports/overlay.png")
    ap.add_argument("--json", help="also write the result JSON to this path")
    ap.add_argument("--show-pii", action="store_true", help="print unmasked values")
    args = ap.parse_args()

    result, art = DocumentPipeline().run(cv2.imread(args.image), return_artifacts=True)
    canvas = art.pre.image.copy()
    flagged = {i.field for i in result.issues}
    for table in result.tables:
        for row in table.rows:
            if row.bbox:
                x0, y0, x1, y1 = row.bbox
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 140, 0), 1)
    for name, f in result.fields.items():
        if f.bbox:
            color = (0, 0, 220) if name in flagged else (0, 170, 0)
            x0, y0, x1, y1 = f.bbox
            cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), color, 2)
            cv2.putText(canvas, f"{name} {f.confidence:.2f}", (x0, max(10, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    payload = json.dumps(result.model_dump(mode="json") if args.show_pii else redact(result), indent=2)
    if args.json:
        Path(args.json).write_text(payload)
    print(payload)
    print(f"overlay written to {args.out}")


if __name__ == "__main__":
    main()
