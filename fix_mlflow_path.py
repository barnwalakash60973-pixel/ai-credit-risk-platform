"""Fix 'Confusion matrix not logged' on the Model Performance page.

Cause: mlflow.db stores ABSOLUTE artifact paths from the machine where the model was trained.
After the project is copied/unzipped elsewhere those paths do not exist, so MLflow cannot find
confusion_matrix.json and the page shows "Confusion matrix not logged".

Usage (from the project root, the folder that contains mlflow.db and mlruns/):
    python fix_mlflow_paths.py
Safe to run repeatedly. Makes a backup copy mlflow.db.bak the first time.
"""
import re
import shutil
import sqlite3
import sys
from pathlib import Path

root = Path.cwd()
db, mlruns = root / "mlflow.db", root / "mlruns"
if not db.exists() or not mlruns.exists():
    sys.exit(f"Run this from the project root. Missing: {db if not db.exists() else mlruns}")

new = mlruns.resolve().as_posix()
con = sqlite3.connect(str(db))
row = con.execute("select artifact_location from experiments where artifact_location like '%mlruns%' limit 1").fetchone()
if not row:
    sys.exit("No MLflow experiment with an mlruns path found in mlflow.db.")
m = re.match(r"^(?:file:/{2,3})?(.*?mlruns)(?:[/\\]|$)", row[0])
old = m.group(1) if m else None
print(f"stored path : {old}\ncurrent path: {new}")

if old and old != new:
    if not (root / "mlflow.db.bak").exists():
        shutil.copy(db, root / "mlflow.db.bak")
    n = 0
    for table, col in [("runs", "artifact_uri"), ("experiments", "artifact_location"),
                       ("logged_models", "artifact_location"), ("model_versions", "storage_location")]:
        try:
            n += con.execute(f"update {table} set {col}=replace({col}, ?, ?) where {col} like ?",
                             (old, new, f"%{old}%")).rowcount
        except sqlite3.OperationalError:
            pass
    con.commit()
    print(f"rewrote {n} rows")
else:
    print("paths already match - nothing to rewrite")

rows = con.execute(
    "select r.run_uuid, r.artifact_uri from runs r join experiments e on e.experiment_id = r.experiment_id "
    "where e.name = 'credit_risk_scoring' order by r.start_time desc"
).fetchall()
con.close()
print(f"training runs in mlflow.db: {len(rows)}")
ok = 0
for run_id, uri in rows:
    path = Path(re.sub(r"^file:/{2,3}", "", uri)) / "confusion_matrix.json"
    ok += path.exists()
    print(f"  run {run_id[:8]}  confusion_matrix.json exists: {path.exists()}")
if rows and not ok:
    print("-> No training run has the file on disk. Re-run:  python -m src.python.models.train_xgboost")
print("Done. Restart Streamlit and reload page 2.")