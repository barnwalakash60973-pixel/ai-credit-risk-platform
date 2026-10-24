"""
Block 2: EDA + PII Identification
Dataset: Give Me Some Credit (data/raw/cs-training.csv)
"""
import pandas as pd

from src.python.config import RAW_DIR

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# 1. Load dataset -----------------------------------------------------------
csv_path = RAW_DIR / "cs-training.csv"
df = pd.read_csv(csv_path, index_col=0)

print("=" * 80)
print("SHAPE:", df.shape)
print("=" * 80)
print(df.info())
print("=" * 80)
print(df.head())

# 2. Check missing values ---------------------------------------------------
print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}))

# 3. Target distribution analysis -------------------------------------------
print("\n--- TARGET DISTRIBUTION: SeriousDlqin2yrs ---")
target_dist = df["SeriousDlqin2yrs"].value_counts(normalize=True) * 100
print(target_dist)
print(f"Imbalance ratio -> Positive class (default): {target_dist.get(1, 0):.2f}%")

# 4. Descriptive statistics --------------------------------------------------
print("\n--- DESCRIBE ---")
print(df.describe().T)

# 5. Identify PII columns -------------------------------------------------
# NOTE: The original Give Me Some Credit dataset DOES NOT have real name/contact columns.
# For this project simulation (since the real dataset is generic & anonymous by default),
# we INTENTIONALLY add synthetic PII columns (member_id, name, contact)
# so that the PII pseudonymization pipeline in Block 3 has something real to process.
# This is documented in ADR-002.
pii_columns_simulated = ["member_id", "name", "contact"]
print("\n--- SIMULATED PII COLUMNS (will be added in Bronze ingest) ---")
print(pii_columns_simulated)

# 6. Correlation of numerical features vs target ---------------------------
print("\n--- CORRELATION WITH TARGET ---")
corr = df.corr(numeric_only=True)["SeriousDlqin2yrs"].sort_values(ascending=False)
print(corr)

# 7. Quick outlier check (example: RevolvingUtilizationOfUnsecuredLines) ----
print("\n--- OUTLIER PREVIEW (RevolvingUtilizationOfUnsecuredLines) ---")
col = "RevolvingUtilizationOfUnsecuredLines"
print(df[col].describe())
print(f"Values > 1.0 (should be ratio, reasonable max ~1-2): {(df[col] > 2).sum()} rows")

print("\nEDA completed. Proceeding to write results to docs/data_dictionary.md")