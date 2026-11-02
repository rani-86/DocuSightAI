"""Train the document classifiers.

1. scikit-learn TF-IDF + logistic-regression baseline.
2. Fine-tuned Hugging Face BERT (PyTorch training loop).

Training text is the ground-truth page text corrupted with an OCR-noise model
(character confusions, drops, merged/split words, dropped lines) so the model
is robust to real OCR output; the held-out test set is scored on *actual*
PaddleOCR output by scripts/evaluate.py.

    python scripts/train_classifier.py --epochs 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docusight  # noqa: E402,F401  (loads torch before paddle on Windows)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

from docusight import DOC_TYPES  # noqa: E402
from docusight.classifier import HF_DIR, SK_FILE, SklearnDocClassifier, prepare_text  # noqa: E402
from docusight.synth import load_labels  # noqa: E402

CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "I": "l", "5": "S", "S": "5", "8": "B", "B": "8", "e": "c",
              "c": "e", "m": "rn", "rn": "m", "a": "o", ".": ",", ",": "."}


def ocr_noise(text: str, rng: random.Random, strength: float) -> str:
    """Simulate OCR artefacts at a given strength in [0, 1]."""
    lines = [ln for ln in text.splitlines() if rng.random() > 0.15 * strength]
    rng.shuffle(lines) if rng.random() < 0.1 * strength else None  # occasional reading-order errors
    out = []
    for ln in lines:
        chars = []
        for ch in ln:
            r = rng.random()
            if r < 0.04 * strength and ch in CONFUSIONS:
                chars.append(CONFUSIONS[ch])
            elif r < 0.06 * strength:
                continue  # dropped glyph
            elif ch == " " and r < 0.10 * strength:
                continue  # merged words
            else:
                chars.append(ch)
        out.append("".join(chars))
    return "\n".join(out)


def augment(records: list[dict], copies: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for rec in records:
        for k in range(copies):
            texts.append(rec["text"] if k == 0 else ocr_noise(rec["text"], rng, rng.uniform(0.2, 1.0)))
            labels.append(rec["doc_type"])
    return texts, labels


def train_hf(train_x, train_y, val_x, val_y, base_model: str, out_dir: Path, epochs: int, lr: float, batch: int, seed: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    label2id = {d: i for i, d in enumerate(DOC_TYPES)}
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(DOC_TYPES), id2label={i: d for d, i in label2id.items()}, label2id=label2id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def encode(xs):
        return tok([prepare_text(x) for x in xs], truncation=True, max_length=256, padding="max_length", return_tensors="pt")

    tr_enc, va_enc = encode(train_x), encode(val_x)
    tr_y = torch.tensor([label2id[y] for y in train_y])
    va_y = np.array([label2id[y] for y in val_y])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    best_f1, history = -1.0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            out = model(input_ids=tr_enc["input_ids"][idx].to(device), attention_mask=tr_enc["attention_mask"][idx].to(device),
                        labels=tr_y[idx].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(out.loss.item())
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(input_ids=va_enc["input_ids"][i:i + 64].to(device),
                                      attention_mask=va_enc["attention_mask"][i:i + 64].to(device)).logits.cpu()
                                for i in range(0, len(val_x), 64)])
        f1 = f1_score(va_y, logits.argmax(-1).numpy(), average="macro")
        history.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "val_macro_f1": round(f1, 4)})
        print(history[-1])
        if f1 > best_f1:  # keep the best checkpoint (early-stopping style)
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return best_f1, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--models", default="models")
    ap.add_argument("--base-model", default="google/bert_uncased_L-2_H-128_A-2")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--copies", type=int, default=4, help="noisy copies per document")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_labels(Path(args.data))
    # Split by *document* before augmentation so noisy copies of one page never
    # land on both sides of the validation boundary.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(records, [r["doc_type"] for r in records]))
    tr_x, tr_y = augment([records[i] for i in tr_idx], args.copies, args.seed)
    va_x, va_y = augment([records[i] for i in va_idx], 2, args.seed + 1)

    models = Path(args.models)
    models.mkdir(exist_ok=True)
    report: dict = {"train_docs": len(tr_idx), "val_docs": len(va_idx), "train_texts": len(tr_x)}

    t0 = time.time()
    sk = SklearnDocClassifier.build().fit(tr_x, tr_y)
    sk_pred = [DOC_TYPES[i] for i in sk.predict_proba(va_x).argmax(1)]
    report["tfidf_logreg"] = {"val_macro_f1": round(f1_score(va_y, sk_pred, average="macro"), 4),
                              "train_seconds": round(time.time() - t0, 1)}
    sk.save(models / SK_FILE)
    print("TF-IDF baseline\n", classification_report(va_y, sk_pred, digits=4))

    t0 = time.time()
    best, hist = train_hf(tr_x, tr_y, va_x, va_y, args.base_model, models / HF_DIR, args.epochs, args.lr, args.batch, args.seed)
    report["hf_transformer"] = {"base_model": args.base_model, "best_val_macro_f1": round(best, 4), "history": hist,
                                "train_seconds": round(time.time() - t0, 1)}
    (models / "classifier_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
