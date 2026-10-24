"""
FAME (RBI Financial Awareness Messages) knowledge base.

The FAME booklet is NOT structured as numbered regulatory paragraphs, so the existing
paragraph-aware chunker in regulation_kb.py does not fit. Here we chunk by
"Message N <title>" instead: each message is a short, self-contained lesson.

The PDF lives in data/knowledge_base/fame/ (NOT pdfs/, which regulation_kb.py indexes) - run:
    python -m src.python.rag.fame_kb
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import pdfplumber
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "fame" / "RBI_FAME_financial_awareness_messages.pdf"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION = "fame_financial_literacy"
EMBED_MODEL = "all-MiniLM-L6-v2"
DOC_TITLE = "RBI Financial Awareness Messages (FAME), 3rd Ed., July 2020"

MSG_RE = re.compile(r"Message\s+(\d{1,2})\s+([A-Z][^\n]{3,80})")
NOISE = re.compile(r"^(FAME|FINANCIAL AWARENESS MESSAGES|\d{1,2})\s*$", re.M)


@dataclass
class FameChunk:
    chunk_id: str
    message_no: int
    title: str
    content: str
    page_start: int
    page_end: int


def load_fame_chunks(pdf_path: Path = PDF_PATH) -> list[FameChunk]:
    # 1) page texts
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            pages.append((i, NOISE.sub("", p.extract_text() or "")))

    # 2) group consecutive pages under the latest "Message N" header
    msgs: dict[int, dict] = {}
    current = None
    for page_no, text in pages:
        m = MSG_RE.search(text)
        if m:
            current = int(m.group(1))
            if current not in msgs:  # ignore the table-of-contents entry
                msgs[current] = {"title": m.group(2).strip(), "text": [], "pages": []}
        if current is not None:
            msgs[current]["text"].append(text)
            msgs[current]["pages"].append(page_no)

    # 3) one chunk per message (messages are ~1-3 pages, fits the 512-token budget well enough;
    #    split further here if a message exceeds your embedding model's limit)
    chunks = []
    for no, d in sorted(msgs.items()):
        body = re.sub(r"\n{2,}", "\n", "\n".join(d["text"])).strip()
        if len(body) < 150:
            continue
        chunks.append(FameChunk(f"fame_msg_{no:02d}", no, d["title"], body, min(d["pages"]), max(d["pages"])))
    return chunks


def _collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        return client.create_collection(COLLECTION, embedding_function=ef)
    return client.get_collection(COLLECTION, embedding_function=ef)


def build_fame_collection():
    chunks = load_fame_chunks()
    col = _collection(reset=True)
    col.add(
        ids=[c.chunk_id for c in chunks],
        documents=[f"Message {c.message_no}: {c.title}\n{c.content}" for c in chunks],
        metadatas=[{"message_no": c.message_no, "title": c.title,
                    "page_start": c.page_start, "page_end": c.page_end,
                    "doc_title": DOC_TITLE} for c in chunks],
    )
    logger.info("Indexed %d FAME messages", len(chunks))


def query_fame(q: str, n: int = 3) -> list[dict]:
    r = _collection().query(query_texts=[q], n_results=n)
    return [{"content": r["documents"][0][i], "metadata": r["metadatas"][0][i],
             "distance": r["distances"][0][i]} for i in range(len(r["ids"][0]))]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_fame_collection()
    for hit in query_fame("how can I improve my credit score"):
        print(round(hit["distance"], 3), hit["metadata"]["title"])
