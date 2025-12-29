import pandas as pd
import json
import sys
import platform
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib

from .config import TrainCfg

log = logging.getLogger(__name__)

def fit_kmeans(df: pd.DataFrame, *, k: int = 5):
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

    model = Pipeline([
        ("pre", pre),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
    ])
    model.fit(df)
    return model

def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{run_tag}__session{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

 
    df = pd.read_parquet(cfg.features_path)
    
 
    feature_df = df.drop(columns=[cfg.target] + list(cfg.id_cols), errors="ignore")
    model = fit_kmeans(feature_df)

  
    joblib.dump(model, run_dir / "model" / "model.pkl")
    
   
    feature_cols = list(feature_df.columns)
    schema = {
        "target": cfg.target,
        "required_feature_columns": feature_cols,
    }
    (run_dir / "schema" / "input_schema.json").write_text(json.dumps(schema, indent=2))

  
    env_meta = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    (run_dir / "env" / "env_meta.json").write_text(json.dumps(env_meta, indent=2))

    registry_dir = root / "models" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    
   
    (registry_dir / "latest.txt").write_text(run_id, encoding="utf-8")
    
    log.info("Registry updated: %s", run_id)

    return run_dir