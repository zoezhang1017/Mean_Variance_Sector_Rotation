from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import rqdatac  # type: ignore
except ImportError as exc:  # pragma: no cover - environment specific
    rqdatac = None  # fallback for linting; runtime will raise

from .settings import PipelineConfig

LOGGER = logging.getLogger("next_pipeline.data")


@dataclass
class DataManager:
    """Handle index universe loading and price data retrieval with caching."""

    config: PipelineConfig
    force_refresh: bool = False
    _rq_initialised: bool = field(init=False, default=False)

    def index_list(self) -> List[str]:
        path = self.config.get_index_list_path()
        if not path.exists():
            raise FileNotFoundError(f"Index list not found: {path}")
        indices = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        LOGGER.info("Loaded %d indices from %s", len(indices), path)
        return indices

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data_cfg = self.config.data
        start_date = data_cfg.get("start_date", "2015-01-01")
        end_date = data_cfg.get("end_date")
        frequency = data_cfg.get("frequency", "1d")

        results: Dict[str, pd.DataFrame] = {}
        for code in self.index_list():
            try:
                results[code] = self._load_or_request(
                    code, start_date=start_date, end_date=end_date, frequency=frequency
                )
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", code, exc)
        return results

    # -------------------------------------------------------------------------
    def _ensure_rqdatac(self) -> None:
        if rqdatac is None:  # pragma: no cover - import guard
            raise RuntimeError(
                "rqdatac is required but not installed. Please `pip install rqdatac`."
            )
        if not self._rq_initialised:
            LOGGER.debug("Initialising rqdatac connection")
            rqdatac.init()
            self._rq_initialised = True

    def _cache_path(self, code: str) -> Path:
        cache_dir = self.config.get_cache_dir()
        safe_code = code.replace(".", "_")
        return cache_dir / f"{safe_code}.parquet"

    def _load_or_request(
        self,
        code: str,
        *,
        start_date: str,
        end_date: Optional[str],
        frequency: str,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(code)
        allow_cache = self.config.allow_cache() and not self.force_refresh

        if allow_cache and cache_path.exists():
            LOGGER.info("Loading cached data for %s from %s", code, cache_path)
            df = pd.read_parquet(cache_path)
            return self._ensure_columns(df)

        LOGGER.info(
            "Fetching %s data for %s (start=%s, end=%s)",
            frequency,
            code,
            start_date,
            end_date or "latest",
        )
        self._ensure_rqdatac()
        price_df = rqdatac.get_price(  # type: ignore[attr-defined]
            code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            expect_df=True,
        )
        if price_df is None:
            raise ValueError(f"rqdatac returned no data for {code}")
        if price_df.empty:
            raise ValueError(f"rqdatac returned empty data for {code}")
        if isinstance(price_df.index, pd.MultiIndex):
            price_df = price_df.droplevel(0)

        price_df = price_df.sort_index()
        price_df = self._ensure_columns(price_df)

        if allow_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            price_df.to_parquet(cache_path)
            LOGGER.debug("Cached %s rows for %s", len(price_df), code)

        return price_df

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = {"open", "high", "low", "close", "prev_close"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Price data missing columns: {sorted(missing)}")
        df["return"] = (df["close"] - df["prev_close"]) / df["prev_close"]
        df.index = pd.to_datetime(df.index)
        return df
