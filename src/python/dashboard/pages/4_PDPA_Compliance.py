"""
Page 4 - PDPA Compliance
Lineage log, erasure request queue, consent status, submit erasure request
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

import streamlit as st

from src.python.dashboard import theme
from src.python.dashboard.db_utils import query_postgres

theme.setup("PDPA Compliance", "🔒")
theme.page_header("🔒", "PDPA Compliance", "Data lineage, right-to-erasure queue and SLA tracking")

tab1, tab2, tab3 = st.tabs(["📋 Data lineage", "🗑️ Erasure requests", "➕ Submit request"])

with tab1:
    try:
        lineage_df = query_postgres(
            "SELECT pipeline_stage, source_file, target_path, record_count, ingest_timestamp, status "
            "FROM audit.data_lineage ORDER BY ingest_timestamp DESC LIMIT 50"
        )
        st.dataframe(lineage_df, width="stretch", hide_index=True)
    except Exception as e:
        theme.empty_state("No lineage data yet", str(e), "📋")

with tab2:
    try:
        erasure_df = query_postgres(
            "SELECT member_id_hash, requested_at, sla_deadline, processed_at, status "
            "FROM audit.erasure_audit_log ORDER BY requested_at DESC LIMIT 100"
        )
        if erasure_df.empty:
            theme.empty_state("No erasure requests", "Nothing has been submitted yet.", "✨")
        else:
            pending = int((erasure_df["status"] == "PENDING").sum())
            done = int((erasure_df["status"] == "COMPLETED").sum())
            theme.kpi_row(
                [
                    {"label": "Total requests", "value": str(len(erasure_df)), "accent": theme.INDIGO},
                    {"label": "Pending", "value": str(pending), "hint": "72h SLA running", "accent": theme.RISK_COLORS["HIGH"]},
                    {"label": "Completed", "value": str(done), "accent": theme.RISK_COLORS["LOW"]},
                ]
            )
            st.write("")

            def highlight_sla(r):
                return ["background-color: #FEF2F2"] * len(r) if r["status"] == "PENDING" else [""] * len(r)

            st.dataframe(erasure_df.style.apply(highlight_sla, axis=1), width="stretch", hide_index=True)
    except Exception as e:
        theme.empty_state("Erasure log unavailable", str(e), "🗑️")

with tab3:
    st.markdown("Simulate a customer exercising the **right to erasure** under PDPA. Processing SLA: **72 hours**.")
    with st.container(border=True):
        member_id_input = st.text_input("Customer member_id_hash")
        if st.button("Submit erasure request", type="primary"):
            if not member_id_input:
                st.error("member_id_hash cannot be empty")
            else:
                try:
                    from src.python.compliance.pdpa_erasure import create_erasure_audit_table, submit_erasure_request

                    create_erasure_audit_table()
                    result = submit_erasure_request(member_id_input)
                    st.success(f"Request for `{member_id_input}` submitted. SLA deadline: {result['sla_deadline']}")
                except Exception as e:
                    st.error(f"Failed to submit request: {e}")

    theme.section("Process queue", "Soft-deletes the Gold-layer records of every pending request")
    if st.button("Process pending erasures now"):
        try:
            from src.python.compliance.pdpa_erasure import process_pending_erasures

            n = process_pending_erasures()
            st.success(f"{n} erasure request(s) processed (Gold layer soft-deleted).")
        except Exception as e:
            st.error(f"Failed to process: {e}")
