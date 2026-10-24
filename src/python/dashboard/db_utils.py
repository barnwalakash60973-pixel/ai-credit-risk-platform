"""Shared DB connection helpers for all dashboard pages."""
import duckdb
import pandas as pd
import streamlit as st
import os
import sys
from pathlib import Path
from src.python.db import get_engine

from src.python.config import settings
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "dbt/local_dev.duckdb")

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

@st.cache_resource
def get_duckdb_connection():
    return duckdb.connect(DUCKDB_PATH)

@st.cache_resource
def get_postgres_engine():
    return get_engine()


@st.cache_data(ttl=60)
def query_duckdb(sql: str) -> pd.DataFrame:
    con = get_duckdb_connection()
    return con.execute(sql).fetchdf()


@st.cache_data(ttl=60)
def query_postgres(sql: str) -> pd.DataFrame:
    engine = get_postgres_engine()
    return pd.read_sql(sql, engine)