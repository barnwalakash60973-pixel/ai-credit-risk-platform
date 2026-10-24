# RBI source PDFs

Drop the **original RBI PDF(s)** here (e.g. a Master Direction on credit risk management / loans
downloaded from https://www.rbi.org.in -> Rules & Regulations -> Master Directions).

Then rebuild the vector index:

    python -m src.python.rag.chroma_store

Notes
- Text-based PDFs only. For a scanned PDF run OCR first: `ocrmypdf in.pdf out.pdf`
- Every PDF in this folder is indexed; chunks keep file name, chapter, paragraph and page for citations.
- If this folder has no PDF, the code falls back to the SIMULATED markdown (and logs a warning).
