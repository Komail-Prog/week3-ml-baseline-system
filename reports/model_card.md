def generate_model_card(cfg: TrainCfg, run_dir: Path, metrics: dict):
    template = f"""# Model Card — {datetime.now().date()}
    
## Problem
- **Predict**: {cfg.target} for each {cfg.id_cols[0] if cfg.id_cols else "unit"}
- **Decision enabled**: Automated segment identification/classification.
- **Constraints**: CPU-only; offline-first; batch inference.

## Data (Contract)
- **Feature table**: {cfg.features_path}
- **Unit of analysis**: One row per unique ID.
- **Target column**: {cfg.target}
- **IDs (passthrough)**: {cfg.id_cols}
- **Holdout strategy**: Random 20% split.

## Metrics
- **Performance**: {json.dumps(metrics, indent=2)}
- **Baseline**: Dummy model (requires implementation).

## Shipping
- **Artifacts**: model.pkl, input_schema.json, holdout_metrics.json
- **Run Directory**: {run_dir.name}
- **Known limitations**: Clustering-based classification; may need retraining as data shifts.
"""
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "model_card.md").write_text(template)

# In run_train:
schema = {
    "version": "1.0",
    "target": cfg.target,
    "features": {
        "n_orders": "int64",
        "avg_amount": "float64",
        "total_amount": "float64",
        "country": "category"
    },
    "id_columns": list(cfg.id_cols),
    "forbidden": [cfg.target] # Never allow target at inference
}