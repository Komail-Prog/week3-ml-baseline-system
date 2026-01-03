import pandas as pd
import json
import sys
import platform
import logging
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

from .config import TrainCfg

log = logging.getLogger(__name__)

def classification_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def regression_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def create_pipeline(df: pd.DataFrame, k: int = 5):
    num = df.select_dtypes(include=["number"]).columns
    cat = df.select_dtypes(exclude=["number"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ]
    )

    return Pipeline([
        ("pre", pre),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
    ])

def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{run_tag}__session{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.features_path)
    
    X = df.drop(columns=[cfg.target] + list(cfg.id_cols), errors="ignore")
    y = df[cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = create_pipeline(X_train)
    pipe.fit(X_train, y_train) 

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    if cfg.task == "classification":
        y_true = np.asarray(y_test).astype(int)
        y_score = pipe.predict(X_test) 
        dummy_pred = dummy.predict(X_test)
        
        metrics = classification_metrics(y_true, y_score)
        dummy_metrics = classification_metrics(y_true, dummy_pred)
    else:
        y_true = np.asarray(y_test).astype(float)
        y_pred = pipe.predict(X_test)
        dummy_pred = dummy.predict(X_test)
        
        metrics = regression_metrics(y_true, y_pred)
        dummy_metrics = regression_metrics(y_true, dummy_pred)

    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    (run_dir / "metrics" / "baseline_comparison.json").write_text(
        json.dumps({"model": metrics, "dummy": dummy_metrics}, indent=2) + "\n", 
        encoding="utf-8"
    )

    joblib.dump(pipe, run_dir / "model" / "model.pkl")
    
    (run_dir / "schema" / "input_schema.json").write_text(
        json.dumps({"target": cfg.target, "features": list(X.columns)}, indent=2)
    )

    env_meta = {"python": sys.version, "platform": platform.platform()}
    (run_dir / "env" / "env_meta.json").write_text(json.dumps(env_meta, indent=2))

    registry_dir = root / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "latest.txt").write_text(run_id, encoding="utf-8")
    
    return run_dir