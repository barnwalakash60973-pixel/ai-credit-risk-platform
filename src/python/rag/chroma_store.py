"""
ChromaDB storage
Embed regulation chunks (official RBI PDFs) with all-MiniLM-L6-v2 and store them in a
ChromaDB persistent collection, keeping source file / page / chapter metadata for citations.

Rebuild the index after adding or replacing PDFs:
    python -m src.python.rag.chroma_store
"""
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.python.rag.regulation_kb import RegulationChunk, load_regulation_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "chroma_db"
COLLECTION_NAME = "credit_regulation_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)


def _metadata(c: RegulationChunk) -> dict:
    # Chroma metadata values must be str/int/float/bool (never None)
    return {
        "section_title": c.section_title or "",
        "subsection_title": c.subsection_title or "",
        "section_number": c.section_number or "",
        "source_file": c.source_file or "",
        "doc_title": c.doc_title or "",
        "page_start": int(c.page_start or 0),
        "page_end": int(c.page_end or 0),
    }


def build_collection(chunks: list[RegulationChunk] | None = None) -> chromadb.Collection:
    client = get_chroma_client()
    embed_fn = get_embedding_function()

    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted old collection '{COLLECTION_NAME}', rebuilding ...")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    chunks = chunks or load_regulation_chunks()
    if not chunks:
        raise RuntimeError("No regulation chunks to index - add the RBI PDF to data/knowledge_base/pdfs/")

    batch = 128  # keep memory flat for large PDFs
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            documents=[c.content for c in part],
            metadatas=[_metadata(c) for c in part],
        )
    logger.info(f"Collection '{COLLECTION_NAME}' created with {len(chunks)} chunks")
    return collection


def get_collection() -> chromadb.Collection:
    client = get_chroma_client()
    embed_fn = get_embedding_function()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def query_regulation(query_text: str, n_results: int = 3) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)

    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
        )
    return output


def format_citation(metadata: dict) -> str:
    """Human-readable citation, e.g. 'Chapter III, Para 12 - <doc title> (p.7)'."""
    where = ", ".join(x for x in (metadata.get("section_number"), metadata.get("subsection_title")) if x)
    doc = metadata.get("doc_title") or metadata.get("source_file") or ""
    p0, p1 = metadata.get("page_start") or 0, metadata.get("page_end") or 0
    pages = ""
    if p0:
        pages = f" (p.{p0})" if p0 == p1 else f" (pp.{p0}\u2013{p1})"
    return f"{where} - {doc}{pages}".strip(" -")


def collection_summary() -> dict:
    """Chunk count and per-source breakdown, for the dashboard."""
    col = get_collection()
    data = col.get(include=["metadatas"])
    per_source: dict[str, int] = {}
    for m in data["metadatas"]:
        key = m.get("doc_title") or m.get("source_file") or "unknown"
        per_source[key] = per_source.get(key, 0) + 1
    return {"chunks": col.count(), "sources": per_source}


if __name__ == "__main__":
    build_collection()

    test_query = "high DTI ratio and high credit utilization"
    results = query_regulation(test_query, n_results=3)
    print(f"\nQuery: '{test_query}'")
    for r in results:
        print(f"\n[{r['chunk_id']}] {format_citation(r['metadata'])}")
        print(f"Distance: {r['distance']:.4f}")
        print(r["content"][:200], "...")
