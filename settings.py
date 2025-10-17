from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """Load and expose configuration for the next pipeline."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    config_path: Path = field(init=False)
    raw: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.config_path = self.root / "config.json"
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.raw = json.load(fh)

    # ---- Convenience properties -------------------------------------------------
    @property
    def data(self) -> Dict[str, Any]:
        return dict(self.raw.get("data", {}))

    @property
    def optimisation(self) -> Dict[str, Any]:
        return dict(self.raw.get("optimisation", {}))

    @property
    def strategy(self) -> Dict[str, Any]:
        return dict(self.raw.get("strategy", {}))

    @property
    def reporting(self) -> Dict[str, Any]:
        return dict(self.raw.get("reporting", {}))

    @property
    def daily(self) -> Dict[str, Any]:
        return dict(self.raw.get("daily", {}))

    # ---- Helper getters ---------------------------------------------------------
    def resolve_path(self, relative: str | Path) -> Path:
        return (self.root / relative).resolve()

    def get_index_list_path(self) -> Path:
        data_cfg = self.data
        path = data_cfg.get("index_list_path")
        if path is None:
            raise ValueError("index_list_path missing in data configuration")
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = self.root.parent / path
        return path_obj

    def get_cache_dir(self) -> Path:
        data_cfg = self.data
        cache_dir = data_cfg.get("cache_dir")
        if cache_dir is None:
            raise ValueError("cache_dir missing in data configuration")
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = self.root.parent / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def optimisation_output_dir(self) -> Path:
        reporting_cfg = self.reporting
        output_dir = reporting_cfg.get("output_dir", "next_pipeline_outputs")
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = self.root.parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def daily_output_dir(self) -> Path:
        base = self.optimisation_output_dir()
        daily_dir = base / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        return daily_dir

    # ---- Derived values ---------------------------------------------------------
    def optimisation_param_space(self) -> Dict[str, tuple[int, int]]:
        cfg = self.optimisation
        return {
            "rebalance_days": tuple(cfg.get("rebalance_days", [5, 60])),
            "min_periods": tuple(cfg.get("min_periods", [60, 360])),
            "future_days": tuple(cfg.get("future_days", [1, 3])),
            "ic_lookback_period": tuple(cfg.get("ic_lookback_period", [10, 90])),
        }

    def allow_cache(self) -> bool:
        return bool(self.data.get("allow_cache", True))

    def end_date(self) -> Optional[str]:
        return self.data.get("end_date")
