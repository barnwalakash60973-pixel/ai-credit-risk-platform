"""
XGBoost hyperparameter tuning with Optuna (single implementation; train_xgboost imports it).
Objective: maximize AUC-ROC via stratified k-fold cross-validation on the TRAINING split only.

No scale_pos_weight: re-weighting the minority class does not help AUC/ranking much but distorts
the predicted probabilities, and the decision tiers are thresholds on probability.
"""
import logging

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_auc(params: dict, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx], verbose=False)
        preds = model.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], preds))
    return float(np.mean(aucs)), float(np.std(aucs))


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    mean_auc, std_auc = _cv_auc(params, X, y, n_splits=n_splits)
    trial.set_user_attr("cv_auc_std", std_auc)
    return mean_auc


def run_optuna_search(
    X: np.ndarray, y: np.ndarray, n_trials: int = 30, seed: int = 42, n_splits: int = 5
) -> tuple[dict, float, float]:
    """Returns (best_params, best_cv_auc_mean, best_cv_auc_std)."""
    logger.info(f"Starting Optuna search: {n_trials} trials x {n_splits}-fold CV ...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: objective(t, X, y, n_splits), n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best CV AUC: {study.best_value:.4f} | params: {study.best_params}")
    return study.best_params, float(study.best_value), float(study.best_trial.user_attrs["cv_auc_std"])
