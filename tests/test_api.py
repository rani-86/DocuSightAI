import cv2
import numpy as np
from fastapi.testclient import TestClient

from conftest import FakeEngine
from docusight import api
from docusight.classifier import KeywordDocClassifier
from docusight.pipeline import DocumentPipeline


def _png() -> bytes:
    img = np.full((1100, 900, 3), 255, np.uint8)
    cv2.putText(img, "TAX INVOICE", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def test_extract_endpoint_end_to_end(monkeypatch, invoice_tokens):
    monkeypatch.setattr(api, "DocumentPipeline",
                        lambda: DocumentPipeline(engine=FakeEngine(invoice_tokens), classifier=KeywordDocClassifier()))
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/v1/documents/extract", files={"file": ("inv.png", _png(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["doc_type"] == "invoice"
        assert body["fields"]["total_amount"]["value"] == "14868.00"
        assert body["fields"]["total_amount"]["bbox"] is not None
        assert body["tables"][0]["rows"][0]["cells"]["item"] == "Office Chair"
        assert client.get(f"/v1/documents/{body['doc_id']}").status_code == 200
        assert "docusight_documents_total" in client.get("/metrics").text

        bad = client.post("/v1/documents/extract", files={"file": ("x.txt", b"hello", "text/plain")})
        assert bad.status_code == 415
