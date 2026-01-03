def generate_model_card(cfg: TrainCfg, run_dir: Path, metrics: dict):
    # Choose the most important metric to highlight at the top
    primary_metric_name = "accuracy" if cfg.task == "classification" else "mse"
    primary_value = metrics.get(primary_metric_name, "N/A")

    template = f"""# Model Card — {datetime.now().date()}
    
## Problem
- **Predict**: {cfg.target} for each {cfg.id_cols[0] if cfg.id_cols else "unit"}
- **Decision enabled**: Automated segment identification/classification.
- **Constraints**: CPU-only; offline-first; batch inference.

## Data (Contract)
- **Feature table**: {cfg.features_path}
- **Holdout strategy**: Random Holdout Split
- **Test Size**: 20%
- **Random Seed**: 42
- **Unit of analysis**: One row per unique ID.

## Metrics
- **Primary Metric ({primary_metric_name})**: {primary_value}
- **Full Metrics**: 
{json.dumps(metrics, indent=2)}

## Shipping
- **Artifacts**: model.pkl, input_schema.json, holdout_metrics.json
- **Run Directory**: {run_dir.name}
- **Known limitations**: Clustering-based classification; performance may be lower than a supervised model.
"""
    # Ensure the reports folder exists inside the specific run directory
    report_path = run_dir / "reports"
    report_path.mkdir(exist_ok=True)
    (report_path / "model_card.md").write_text(template)
    
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