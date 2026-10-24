"""
RAG + API plumbing WITHOUT external services.

Real sentence-transformers weights and Ollama are replaced by a deterministic bag-of-words
embedding and a stubbed LLM response. This verifies the wiring (RBI PDF -> chunks -> ChromaDB ->
query built from SHAP features -> prompt -> letter -> quality check -> API response), NOT retrieval
quality or letter quality; those need the real models.
"""
import hashlib
import json
import re

import duckdb
import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

DIM = 256


class BowEmbedding(chromadb.EmbeddingFunction):
    """Hashed bag-of-words: texts sharing words get close vectors."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z]{3,}", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    @staticmethod
    def name():
        return "bow-test"


class FakeResp:
    def __init__(self, text):
        self._t = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._t, "eval_count": 10, "prompt_eval_count": 20}


@pytest.fixture
def env(tmp_path, monkeypatch, rbi_pdf_dir):
    from src.python.rag import chroma_store, credit_decision_explainer as ex, regulation_kb

    # index the (generated) RBI-style PDF, not the legacy simulated markdown
    monkeypatch.setattr(regulation_kb, "PDF_DIR", rbi_pdf_dir)

    monkeypatch.setattr(chroma_store, "CHROMA_PERSIST_DIR", tmp_path / "chroma")
    monkeypatch.setattr(chroma_store, "get_embedding_function", lambda: BowEmbedding())
    chroma_store.build_collection()

    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    feats = [
        {"feature": "revolving_utilization", "shap_value": 0.9, "direction": "increases_risk"},
        {"feature": "total_past_due", "shap_value": 0.7, "direction": "increases_risk"},
        {"feature": "age", "shap_value": -0.2, "direction": "decreases_risk"},
    ]
    con.execute("CREATE TABLE prediction_explanations (member_id_hash VARCHAR, predicted_proba DOUBLE, top_shap_features_json VARCHAR)")
    con.execute("INSERT INTO prediction_explanations VALUES ('m1', 0.45, ?)", [json.dumps(feats)])
    con.close()
    monkeypatch.setattr(ex, "DUCKDB_PATH", db)

    letter = "Your application was declined. Credit utilization and past due payments drove this; age helped."
    monkeypatch.setattr(ex.requests, "post", lambda *a, **k: FakeResp(letter))
    return ex


def test_letter_generation_end_to_end_with_stubs(env):
    result = env.generate_credit_decision_letter("m1")
    assert result["risk_tier"] == "HIGH" and result["decision"] == "REJECTED"
    assert result["regulation_references"], "no regulation chunks retrieved"
    assert all("rbi" in r.lower() or "Reserve Bank" in r for r in result["regulation_references"])
    assert any("p." in r or "pp." in r for r in result["regulation_references"]), "citations lack page numbers"
    assert result["narrative"].startswith("Your application")
    assert result["top_shap_features"][0]["feature"] == "revolving_utilization"


def test_unknown_applicant_raises_value_error(env):
    with pytest.raises(ValueError):
        env.generate_credit_decision_letter("does-not-exist")


def test_api_endpoint_returns_letter_and_404(env, monkeypatch):
    from fastapi.testclient import TestClient

    import src.python.api.main as api

    monkeypatch.setattr(api, "create_audit_table", lambda: None)
    monkeypatch.setattr(api, "log_audit_entry", lambda **k: None)
    monkeypatch.setattr(api, "log_llmops_metrics", lambda *a, **k: None)
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/credit-decision/m1")
    assert r.status_code == 200 and r.json()["decision"] == "REJECTED"
    assert client.post("/credit-decision/nope").status_code == 404
