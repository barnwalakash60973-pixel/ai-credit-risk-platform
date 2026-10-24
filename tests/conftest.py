"""Shared pytest fixtures across the entire test suite."""
import pandas as pd
import pytest


@pytest.fixture
def sample_credit_df() -> pd.DataFrame:
    """Dummy dataframe matching the structure of Give Me Some Credit, used for unit testing."""
    return pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1, 0, 0, 1],
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.9, 0.3, 0.05, 1.5],
            "age": [35, 45, 29, 60, 50],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 0, 0, 3],
            "DebtRatio": [0.2, 0.8, 0.1, 0.05, 0.6],
            "MonthlyIncome": [5000.0, 3000.0, 7000.0, None, 4500.0],
            "NumberOfOpenCreditLinesAndLoans": [5, 8, 3, 10, 6],
            "NumberOfTimes90DaysLate": [0, 1, 0, 0, 2],
            "NumberRealEstateLoansOrLines": [1, 0, 2, 3, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 0, 1],
            "NumberOfDependents": [2, 0, 1, None, 3],
        }
    )


# ---- RBI-style PDF helper (used by RAG tests) ----
from pathlib import Path  # noqa: E402


def make_rbi_style_pdf(path: Path, n_chapters: int = 3) -> Path:
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    topics = [
        "repayment capacity of the borrower, income and existing obligations",
        "overdue payments, delinquency and asset classification of the account",
        "disclosure of interest rate, fees and key facts of the loan to the borrower",
        "credit limit utilisation and total exposure of the borrower across lenders",
    ]
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Reserve Bank of India (Credit Risk Management) Directions, 2025")
    para_no = 0
    for page in range(1, n_chapters * 2 + 1):
        c.setFont("Helvetica", 8)
        c.drawString(50, 810, "Reserve Bank of India - Credit Risk Management Directions")  # running header
        y = 780
        c.setFont("Helvetica", 10)
        lines = []
        if page % 2 == 1:
            lines.append(f"Chapter {'I' * ((page + 1) // 2)} - Section {(page + 1) // 2} Requirements")
        for k in range(4):
            para_no += 1
            topic = topics[(para_no + k) % len(topics)]
            lines.append(
                f"{para_no}. A bank shall have a Board approved policy covering {topic}. "
                f"Item {para_no} requires documented review and periodic reporting to the Board of Directors. "
                * 2
            )
        for ln in lines:
            for w in simpleSplit(ln, "Helvetica", 10, 490):
                c.drawString(50, y, w)
                y -= 13
        c.drawString(290, 30, str(page))
        c.showPage()
    c.save()
    return path


@pytest.fixture
def rbi_pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    make_rbi_style_pdf(d / "rbi_credit_risk_directions.pdf")
    return d
