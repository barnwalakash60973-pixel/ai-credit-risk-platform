from src.python.rag.regulation_kb import (
    MAX_CHUNK_TOKENS,
    _estimate_tokens,
    load_regulation_chunks,
    parse_regulation_markdown,
    parse_regulation_pdfs,
)


# ---------------------------------------------------------------- PDF (primary source)
def test_pdf_produces_chunks_with_provenance(rbi_pdf_dir):
    chunks = parse_regulation_pdfs(rbi_pdf_dir)
    assert len(chunks) >= 3
    assert all(c.content for c in chunks)
    assert all(c.source_file == "rbi_credit_risk_directions.pdf" for c in chunks)
    assert all(1 <= c.page_start <= c.page_end for c in chunks)


def test_pdf_detects_chapters_and_paragraph_numbers(rbi_pdf_dir):
    chunks = parse_regulation_pdfs(rbi_pdf_dir)
    chapters = {c.section_number for c in chunks}
    assert "Chapter I" in chapters and "Chapter II" in chapters
    assert any(c.subsection_title.startswith("Para ") for c in chunks)


def test_pdf_chunks_respect_token_limit_and_chapter_boundaries(rbi_pdf_dir):
    chunks = parse_regulation_pdfs(rbi_pdf_dir)
    assert max(_estimate_tokens(c.content) for c in chunks) <= MAX_CHUNK_TOKENS
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique ids


def test_running_header_and_page_numbers_are_stripped(rbi_pdf_dir):
    chunks = parse_regulation_pdfs(rbi_pdf_dir)
    assert not any("Reserve Bank of India - Credit Risk" in c.content for c in chunks)


def test_loader_prefers_pdf_over_markdown(rbi_pdf_dir):
    chunks = load_regulation_chunks(rbi_pdf_dir)
    assert chunks and all(c.source_file.endswith(".pdf") for c in chunks)


def test_loader_falls_back_to_markdown_when_no_pdf(tmp_path):
    chunks = load_regulation_chunks(tmp_path / "empty")
    assert chunks and all(c.source_file.endswith(".md") for c in chunks)


# ---------------------------------------------------------------- legacy markdown fallback
def test_parse_regulation_markdown_produces_chunks():
    chunks = parse_regulation_markdown()
    assert len(chunks) > 5
    assert all(c.content for c in chunks)


def test_chunks_have_section_metadata():
    section_numbers = {c.section_number for c in parse_regulation_markdown()}
    assert "Section 1" in section_numbers
    assert "Section 7" in section_numbers  # PDPA section


def test_estimate_tokens_reasonable():
    assert _estimate_tokens("this is a five word example") > 0
