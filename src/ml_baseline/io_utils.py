from pathlib import Path
import pandas as pd


NA_VALUES = ["", "NA", "N/A", "null", "None"]

def parquet_supported() -> bool:
    """Check if the environment can handle parquet files."""
    try:
        import pyarrow
        return True
    except ImportError:
        return False

def read_tabular(path: Path) -> pd.DataFrame:
    """Flexible reader based on file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)

def write_tabular(df: pd.DataFrame, path: Path) -> None:
    """Writes data and ensures the directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_orders_csv(path: Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        dtype={"order_id": "string", "user_id": "string"},
        na_values=NA_VALUES,
    )

def read_users_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"user_id": "string"},
        na_values=NA_VALUES,
    )

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def assert_no_duplicate_key(df, key_cols):
    dup = df.duplicated(subset=key_cols, keep=False)
    assert not dup.any(), (
        f"Duplicate rows for key={key_cols} (n={dup.sum()})"
    )
