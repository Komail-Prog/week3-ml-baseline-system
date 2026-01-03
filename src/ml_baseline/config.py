from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    processed: Path
    reports: Path
    

    @classmethod
    def from_repo_root(cls) -> "Paths":
        root = Path(__file__).parent.parent.parent.resolve()
        return make_paths(root)

def make_paths(root: Path) -> Paths:
    data = root / "data"
    reports = root / "reports"
    return Paths(
        root=root,
        processed=data / "processed",
        reports=reports,
        
    )

@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str
    id_cols: tuple[str, ...] = ("user_id",)
    session_id: int = 42
    time_col: str | None = None
    task: str = "classification"