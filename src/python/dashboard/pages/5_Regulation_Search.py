"""
Page 5 - Regulation Search
Semantic search over the official RBI PDFs indexed in ChromaDB (the RAG knowledge base).
Also lets you add/replace the source PDFs and rebuild the index.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

import streamlit as st

from src.python.dashboard import theme
from src.python.rag.regulation_kb import PDF_DIR

theme.setup("Regulation Search", "📚")
theme.page_header("📚", "Regulation Search", "Ask a question — get the exact RBI paragraph, chapter and page")

EXAMPLES = [
    "How should a bank assess the repayment capacity of a borrower?",
    "What are the rules on overdue payments and asset classification?",
    "Disclosure of loan terms and interest rate to the borrower",
    "Credit risk management policy approved by the Board",
]


def _summary():
    from src.python.rag.chroma_store import collection_summary

    return collection_summary()


try:
    summary = _summary()
except Exception:
    summary = None

pdfs = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
theme.kpi_row(
    [
        {"label": "PDFs in folder", "value": str(len(pdfs)), "hint": "data/knowledge_base/pdfs", "accent": theme.INDIGO},
        {"label": "Indexed chunks", "value": f"{summary['chunks']:,}" if summary else "—", "hint": "ChromaDB collection", "accent": theme.TEAL},
        {"label": "Embedding model", "value": "MiniLM", "hint": "all-MiniLM-L6-v2", "accent": theme.VIOLET},
    ]
)

tab_search, tab_kb = st.tabs(["🔎 Search", "🗂️ Knowledge base"])

with tab_search:
    if summary is None:
        theme.empty_state(
            "Index not built yet",
            "Add the RBI PDF in the 'Knowledge base' tab (or copy it to data/knowledge_base/pdfs/) and build the index.",
            "🧱",
        )
    else:
        st.write("")
        if "reg_query" not in st.session_state:
            st.session_state["reg_query"] = ""
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, key=f"ex_{ex}", width="stretch"):
                st.session_state["reg_query"] = ex
        query = st.text_input("Your question", key="reg_query", placeholder="e.g. what does the RBI say about overdue loans?")
        n = st.slider("Number of passages", 1, 8, 4)

        if query.strip():
            from src.python.rag.chroma_store import format_citation, query_regulation

            with st.spinner("Searching the RBI directions…"):
                hits = query_regulation(query, n_results=n)
            if not hits:
                theme.empty_state("No passages found", "Try rephrasing the question.", "🤷")
            for i, h in enumerate(hits, start=1):
                m = h["metadata"]
                text = h["content"] if len(h["content"]) < 1200 else h["content"][:1200] + " …"
                theme.regulation_card(i, format_citation(m), text, h["distance"], tags=[m.get("source_file", "")] if m.get("source_file") else [])

with tab_kb:
    st.markdown("**Source documents** — official RBI PDFs that feed the RAG pipeline.")
    if summary:
        for name, count in summary["sources"].items():
            st.markdown(f"- {name} — `{count}` chunks")
    elif pdfs:
        for p in pdfs:
            st.markdown(f"- {p.name}")
    else:
        st.info("No PDF yet. Upload the original RBI PDF below.")

    with st.container(border=True):
        uploaded = st.file_uploader("Add an official RBI PDF", type=["pdf"], accept_multiple_files=True)
        replace = st.checkbox("Replace existing PDFs (remove the ones already in the folder)")
        if st.button("Save & rebuild index", type="primary", disabled=not uploaded):
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            if replace:
                for old in PDF_DIR.glob("*.pdf"):
                    old.unlink()
            for f in uploaded:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f.name)
                (PDF_DIR / safe).write_bytes(f.getbuffer())
            with st.spinner("Parsing PDFs and embedding chunks (first run downloads the embedding model)…"):
                try:
                    from src.python.rag.chroma_store import build_collection

                    col = build_collection()
                    st.success(f"Index rebuilt: {col.count()} chunks.")
                except Exception as e:
                    st.error(f"Index build failed: {e}")
