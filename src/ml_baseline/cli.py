import typer
from pathlib import Path
from .config import Paths, TrainCfg
from .train import run_train  
from .sample_data import make_sample_feature_table

app = typer.Typer()

@app.command()
def train(target: str = "is_high_value"):
    """Run training for the sample target."""
    paths = Paths.from_repo_root()
    
  
    cfg = TrainCfg(
        features_path=paths.processed / "features.parquet",
        target=target
    )
    
 
    print(f"Training model for target: {target}...")
    run_dir = run_train(cfg, root=paths.root)
    print(f"Saved run: {run_dir}")

@app.command()
def make_sample_data():
    """Generate sample data."""
    make_sample_feature_table()
    print("Sample data generated!")

if __name__ == "__main__":
    app()