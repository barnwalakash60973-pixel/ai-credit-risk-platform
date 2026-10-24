"""
Regulation Knowledge Base - PDF ingestion & chunking

Reads the ORIGINAL RBI (Reserve Bank of India) PDFs from ``data/knowledge_base/pdfs/``,
extracts the text page by page, detects the document structure (chapters / numbered
paragraphs), and packs the paragraphs into ~512-token chunks with rich metadata
(source file, page range, chapter, paragraph numbers) ready for ChromaDB.

Why paragraph-aware chunking?  RBI directions are written as numbered paragraphs
("12. A bank shall ..."). Keeping a paragraph intact (and never crossing a chapter
boundary) gives the LLM a self-contained clause it can cite as "Chapter III, Para 12, p.7".

The legacy simulated markdown knowledge base is kept ONLY as a fallback so unit tests and
fresh clones still run; a loud warning is logged whenever it is used.
"""
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
PDF_DIR = KB_DIR / "pdfs"  # <- drop the official RBI PDF(s) here
LEGACY_MD_PATH = KB_DIR / "credit_regulation.md"
KB_PATH = LEGACY_MD_PATH  # backwards-compatible alias

MAX_CHUNK_TOKENS = 512  # approx, rough estimate of 1 token ~ 0.75 words
MIN_TEXT_CHARS = 200  # below this a PDF is treated as scanned / image-only


@dataclass
class RegulationChunk:
    chunk_id: str
    section_title: str
    subsection_title: str
    content: str
    section_number: str
    # --- provenance (filled for PDF sources) ---
    source_file: str = ""
    doc_title: str = ""
    page_start: int = 0
    page_end: int = 0


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


# --------------------------------------------------------------------------------------
# PDF text extraction
# --------------------------------------------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "doc"


def _pretty_title(pdf_path: Path, meta_title: str | None) -> str:
    if meta_title and len(meta_title.strip()) > 5:
        return meta_title.strip()
    return re.sub(r"[_\-]+", " ", pdf_path.stem).strip().title()


def _clean_page_text(raw: str) -> str:
    raw = raw.replace("\u00a0", " ").replace("\u200b", "")
    raw = re.sub(r"-\n(?=[a-z])", "", raw)  # re-join hyphenated line breaks
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw


def extract_pdf_pages(pdf_path: Path) -> tuple[list[tuple[int, list[str]]], str]:
    """Return ([(page_number, [clean lines])...], document_title).

    Lines that repeat on many pages (running headers/footers) and bare page numbers are dropped.
    """
    from pypdf import PdfReader  # imported lazily so the markdown fallback needs no PDF stack

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError(f"{pdf_path.name} is password protected")

    meta_title = None
    try:
        meta_title = reader.metadata.title if reader.metadata else None
    except Exception:
        pass

    pages: list[tuple[int, list[str]]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_page_text(page.extract_text() or "")
        except Exception as exc:  # a single broken page should not kill the whole ingest
            logger.warning("%s p.%d: text extraction failed (%s)", pdf_path.name, idx, exc)
            text = ""
        lines = [ln.strip() for ln in text.split("\n")]
        pages.append((idx, [ln for ln in lines if ln]))

    # --- drop running headers / footers: short lines at the very top / bottom of a page that
    # repeat (ignoring digits) on >= 40% of the pages. Body lines are never candidates.
    n_pages = max(len(pages), 1)
    counter: Counter[str] = Counter()
    for _, lines in pages:
        edge = set(lines[:2] + lines[-2:])
        for ln in edge:
            if len(ln) <= 120:
                counter[re.sub(r"\d+", "#", ln)] += 1
    repeated = {k for k, v in counter.items() if n_pages >= 5 and v / n_pages >= 0.4}

    page_num_only = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
    cleaned: list[tuple[int, list[str]]] = []
    for num, lines in pages:
        edge_idx = set(range(min(2, len(lines)))) | set(range(max(len(lines) - 2, 0), len(lines)))
        keep = [
            ln
            for i, ln in enumerate(lines)
            if not page_num_only.match(ln)
            and not (i in edge_idx and re.sub(r"\d+", "#", ln) in repeated)
        ]
        cleaned.append((num, keep))

    return cleaned, _pretty_title(pdf_path, meta_title)


# --------------------------------------------------------------------------------------
# Structure detection
# --------------------------------------------------------------------------------------
# "Chapter II - Credit Risk Evaluation", "CHAPTER III: ...", "Annex 1", "Annexure A", "Part B"
_CHAPTER_RE = re.compile(
    r"^(?P<label>(?:chapter|annex(?:ure)?|part|schedule|appendix)\s+[IVXLCDM0-9A-Z][\w\-]*)"
    r"\s*[-\u2013\u2014:.]*\s*(?P<title>.*)$",
    re.IGNORECASE,
)
# "12. A bank shall ...", "3.1 Banks must ...", "(iv) ..." is NOT a new paragraph (sub-clause)
_PARA_RE = re.compile(r"^(?P<num>\d{1,3}(?:\.\d{1,2})?)[.)]?\s+(?P<rest>[A-Z\u2018\u201c\"'(].*)$")


@dataclass
class _Block:
    chapter_label: str
    chapter_title: str
    para_no: str
    page_start: int
    page_end: int
    text: str


def _split_into_blocks(pages: list[tuple[int, list[str]]]) -> list[_Block]:
    blocks: list[_Block] = []
    chapter_label, chapter_title = "", ""
    cur: _Block | None = None
    pending_chapter_title = False

    def close():
        nonlocal cur
        if cur and cur.text.strip():
            blocks.append(cur)
        cur = None

    for page_no, lines in pages:
        for ln in lines:
            m_ch = _CHAPTER_RE.match(ln)
            # a chapter heading is short and does not look like a sentence
            if m_ch and len(ln) <= 140 and not ln.endswith("."):
                close()
                chapter_label = m_ch.group("label").strip()
                chapter_title = m_ch.group("title").strip()
                pending_chapter_title = not chapter_title
                continue
            if pending_chapter_title:
                # title printed on the line below "Chapter II"
                if len(ln) <= 120 and not _PARA_RE.match(ln):
                    chapter_title = ln
                    pending_chapter_title = False
                    continue
                pending_chapter_title = False

            m_para = _PARA_RE.match(ln)
            if m_para:
                close()
                cur = _Block(chapter_label, chapter_title, m_para.group("num"), page_no, page_no, ln)
            else:
                if cur is None:
                    cur = _Block(chapter_label, chapter_title, "", page_no, page_no, ln)
                else:
                    cur.text += " " + ln
                    cur.page_end = page_no
    close()
    return blocks


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split an oversize paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(\u2018\u201c\"'])", text)
    parts, buf = [], ""
    for s in sentences:
        if buf and _estimate_tokens(buf + " " + s) > max_tokens:
            parts.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        parts.append(buf.strip())
    # hard split for pathological text without punctuation
    out = []
    for p in parts:
        if _estimate_tokens(p) <= max_tokens:
            out.append(p)
        else:
            words = p.split()
            step = int(max_tokens * 0.75)
            out.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return out


def _pack_blocks(blocks: list[_Block], source_file: str, doc_title: str) -> list[RegulationChunk]:
    """Greedily merge consecutive paragraphs of the same chapter up to MAX_CHUNK_TOKENS."""
    slug = _slug(Path(source_file).stem)
    chunks: list[RegulationChunk] = []
    counter = 0

    def emit(group: list[_Block], text_override: str | None = None):
        nonlocal counter
        text = text_override or "\n\n".join(b.text for b in group)
        if not text.strip():
            return
        counter += 1
        first, last = group[0], group[-1]
        nums = [b.para_no for b in group if b.para_no]
        if nums:
            para = f"Para {nums[0]}" if nums[0] == nums[-1] else f"Para {nums[0]}\u2013{nums[-1]}"
        else:
            para = f"p.{first.page_start}"
        chapter_label = first.chapter_label or "General"
        chunks.append(
            RegulationChunk(
                chunk_id=f"{slug}_c{counter:04d}",
                section_title=first.chapter_title or chapter_label,
                subsection_title=para,
                content=text.strip(),
                section_number=chapter_label,
                source_file=source_file,
                doc_title=doc_title,
                page_start=first.page_start,
                page_end=last.page_end,
            )
        )

    group: list[_Block] = []
    tokens = 0
    for b in blocks:
        b_tokens = _estimate_tokens(b.text)
        if b_tokens > MAX_CHUNK_TOKENS:
            if group:
                emit(group)
                group, tokens = [], 0
            for part in _split_long_text(b.text, MAX_CHUNK_TOKENS):
                emit([b], text_override=part)
            continue
        same_chapter = not group or group[0].chapter_label == b.chapter_label
        if group and (not same_chapter or tokens + b_tokens > MAX_CHUNK_TOKENS):
            emit(group)
            group, tokens = [], 0
        group.append(b)
        tokens += b_tokens
    if group:
        emit(group)
    return chunks


def parse_regulation_pdf(pdf_path: Path) -> list[RegulationChunk]:
    """Parse one RBI PDF into metadata-rich chunks."""
    pages, doc_title = extract_pdf_pages(pdf_path)
    total_chars = sum(len(ln) for _, lines in pages for ln in lines)
    if total_chars < MIN_TEXT_CHARS:
        logger.warning(
            "%s has almost no extractable text (%d chars) - it looks like a scanned PDF. "
            "Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) and re-ingest.",
            pdf_path.name,
            total_chars,
        )
        return []
    blocks = _split_into_blocks(pages)
    chunks = _pack_blocks(blocks, source_file=pdf_path.name, doc_title=doc_title)
    logger.info("Parsed %d chunks from %s (%d pages)", len(chunks), pdf_path.name, len(pages))
    return chunks


def parse_regulation_pdfs(pdf_dir: Path | None = None) -> list[RegulationChunk]:
    """Parse every ``*.pdf`` in ``pdf_dir`` (default: data/knowledge_base/pdfs)."""
    pdf_dir = Path(pdf_dir) if pdf_dir else PDF_DIR
    pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    all_chunks: list[RegulationChunk] = []
    for pdf in pdfs:
        try:
            all_chunks.extend(parse_regulation_pdf(pdf))
        except Exception as exc:
            logger.error("Skipping %s: %s", pdf.name, exc)
    return all_chunks


# --------------------------------------------------------------------------------------
# Legacy markdown parser (kept as fallback only)
# --------------------------------------------------------------------------------------
def parse_regulation_markdown(md_path: Path = LEGACY_MD_PATH) -> list[RegulationChunk]:
    """Parse the SIMULATED markdown regulation into chunks (legacy fallback)."""
    text = Path(md_path).read_text(encoding="utf-8")

    lines = text.split("\n")
    chunks: list[RegulationChunk] = []
    current_section = ""
    current_section_number = ""
    current_subsection = ""
    current_content_lines: list[str] = []
    chunk_counter = 0

    def flush_chunk():
        nonlocal chunk_counter
        content = "\n".join(current_content_lines).strip()
        if content:
            chunk_counter += 1
            chunks.append(
                RegulationChunk(
                    chunk_id=f"reg_chunk_{chunk_counter:03d}",
                    section_title=current_section,
                    subsection_title=current_subsection,
                    content=content,
                    section_number=current_section_number,
                    source_file=Path(md_path).name,
                    doc_title="Simulated Consumer Credit Regulation (demo)",
                )
            )

    for line in lines:
        h2_match = re.match(r"^## (Section \d+): (.+)", line)
        h3_match = re.match(r"^### ([\d.]+) (.+)", line)

        if h2_match:
            flush_chunk()
            current_content_lines = []
            current_section_number = h2_match.group(1)
            current_section = h2_match.group(2)
            current_subsection = ""
        elif h3_match:
            flush_chunk()
            current_content_lines = []
            current_subsection = f"{h3_match.group(1)} {h3_match.group(2)}"
            current_content_lines.append(line)
        else:
            current_content_lines.append(line)

    flush_chunk()

    final_chunks: list[RegulationChunk] = []
    for c in chunks:
        if _estimate_tokens(c.content) <= MAX_CHUNK_TOKENS:
            final_chunks.append(c)
            continue
        buffer, part = "", 1
        for p in c.content.split("\n\n"):
            if _estimate_tokens(buffer + p) > MAX_CHUNK_TOKENS and buffer:
                final_chunks.append(_replace_part(c, buffer, part))
                part += 1
                buffer = p
            else:
                buffer += "\n\n" + p
        if buffer.strip():
            final_chunks.append(_replace_part(c, buffer, part))

    logger.info("Parsed %d regulation chunks from %s", len(final_chunks), md_path)
    return final_chunks


def _replace_part(c: RegulationChunk, text: str, part: int) -> RegulationChunk:
    return RegulationChunk(
        chunk_id=f"{c.chunk_id}_p{part}",
        section_title=c.section_title,
        subsection_title=c.subsection_title,
        content=text.strip(),
        section_number=c.section_number,
        source_file=c.source_file,
        doc_title=c.doc_title,
    )


# --------------------------------------------------------------------------------------
# Public loader used by chroma_store
# --------------------------------------------------------------------------------------
def load_regulation_chunks(pdf_dir: Path | None = None) -> list[RegulationChunk]:
    """Official RBI PDFs first; the simulated markdown only if no usable PDF exists."""
    chunks = parse_regulation_pdfs(pdf_dir)
    if chunks:
        return chunks
    logger.warning(
        "No usable PDF found in %s - FALLING BACK to the SIMULATED markdown regulation. "
        "Copy the official RBI PDF into that folder and rebuild the index "
        "(`python -m src.python.rag.chroma_store`).",
        pdf_dir or PDF_DIR,
    )
    return parse_regulation_markdown()


if __name__ == "__main__":
    found = load_regulation_chunks()
    for c in found[:3]:
        print(
            f"--- {c.chunk_id} | {c.section_number}: {c.section_title} > {c.subsection_title} "
            f"| {c.source_file} p.{c.page_start}-{c.page_end} ---"
        )
        print(c.content[:300], "...\n")
