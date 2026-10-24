"""
MLflow Model Registry helpers.

Uses registry ALIASES (`champion`) instead of the deprecated stage API
(`transition_model_version_stage`). The feature list used at training time is stored as a run
artifact (`feature_columns.json`) and travels with the model, so scoring/SHAP never guess it.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from src.python.config import BASE_DIR, MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "credit_risk_xgboost"
CHAMPION_ALIAS = "champion"
FEATURES_ARTIFACT = "feature_columns.json"
SELECTION_METRIC = "test_auc"

# Portable copy of the champion (native XGBoost JSON + feature list + metrics). MLflow stores ABSOLUTE
# artifact paths, so mlflow.db / mlruns break when the project folder is moved or unzipped elsewhere;
# this export does not. load_champion() falls back to it automatically.
EXPORT_DIR = BASE_DIR / "models" / "champion"


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_champion_version():
    """Return the ModelVersion currently aliased as champion, or None."""
    try:
        return _client().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None


def _metric(run_id: str, key: str = SELECTION_METRIC) -> float | None:
    return _client().get_run(run_id).data.metrics.get(key)


def promote_to_champion(version: str) -> None:
    _client().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, str(version))
    logger.info(f"Model '{MODEL_NAME}' version {version} is now '{CHAMPION_ALIAS}'")


def promote_if_better(version: str, run_id: str) -> bool:
    """Promote `version` only if it beats the current champion on held-out test AUC."""
    new_score = _metric(run_id)
    champ = get_champion_version()
    if champ is not None:
        champ_score = _metric(champ.run_id)
        if champ_score is not None and new_score is not None and new_score <= champ_score:
            logger.info(
                f"Not promoting v{version} ({SELECTION_METRIC}={new_score:.4f}); "
                f"champion v{champ.version} is better or equal ({champ_score:.4f})"
            )
            return False
    promote_to_champion(version)
    return True


def promote_latest_to_production() -> str:
    """Explicitly promote the newest registered version (manual override; name kept for the CLI/README)."""
    versions = _client().search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered model versions found for '{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    promote_to_champion(latest.version)
    return latest.version


def _load_champion_mlflow() -> tuple[object, list[str]]:
    """Load (model, feature_columns) of the champion from the MLflow registry."""
    champ = get_champion_version()
    if champ is None:
        raise RuntimeError(
            f"No '{CHAMPION_ALIAS}' alias on '{MODEL_NAME}'. Run train_xgboost (auto-promotes) "
            "or `python -m src.python.models.model_registry`."
        )
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    features = mlflow.artifacts.load_dict(f"runs:/{champ.run_id}/{FEATURES_ARTIFACT}")
    logger.info(f"Loaded champion v{champ.version} with {len(features)} features")
    return model, features


def export_champion(out_dir: Path = EXPORT_DIR) -> Path:
    """Write the champion as plain files: model.json, feature_columns.json, metadata.json."""
    champ = get_champion_version()
    if champ is None:
        raise RuntimeError("No champion to export - train a model first.")
    model, features = _load_champion_mlflow()
    run = _client().get_run(champ.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    (out_dir / FEATURES_ARTIFACT).write_text(json.dumps(features, indent=2))
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "version": champ.version,
                "run_id": champ.run_id,
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "metrics": run.data.metrics,
                "params": run.data.params,
            },
            indent=2,
        )
    )
    logger.info(f"Champion v{champ.version} exported to {out_dir}")
    return out_dir


def _load_champion_exported(export_dir: Path = EXPORT_DIR) -> tuple[object, list[str]]:
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(str(export_dir / "model.json"))
    features = json.loads((export_dir / FEATURES_ARTIFACT).read_text())
    logger.info(f"Loaded exported champion from {export_dir} with {len(features)} features")
    return model, features


def load_champion() -> tuple[object, list[str]]:
    """Load (model, feature_columns): MLflow registry first, then the portable export in models/champion/."""
    try:
        return _load_champion_mlflow()
    except Exception as e:
        if (EXPORT_DIR / "model.json").exists():
            logger.warning(f"MLflow champion unavailable ({type(e).__name__}); using exported model in {EXPORT_DIR}")
            return _load_champion_exported()
        raise


def load_production_model():
    """Backwards-compatible wrapper returning just the model."""
    return load_champion()[0]


if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        export_champion()
    else:
        promote_latest_to_production()
