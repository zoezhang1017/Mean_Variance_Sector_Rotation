from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from mean_variance_system.config import OPTIMISATION_CONFIG, FACTOR_CONFIG
from .factor_engine import EnhancedMeanVarianceStrategy
from .reporting import create_heatmap
from .settings import PipelineConfig

LOGGER = logging.getLogger("next_pipeline.daily")


@dataclass
class DailySignalRunner:
    config: PipelineConfig

    def _params_path(self) -> Path:
        output_dir = self.config.optimisation_output_dir()
        path = output_dir / "optimised_parameters.xlsx"
        if not path.exists():
            raise FileNotFoundError(
                f"Optimised parameter file not found: {path}. "
                "Please run run_full_pipeline.py first."
            )
        return path

    def load_parameters(self) -> pd.DataFrame:
        df = pd.read_excel(self._params_path())
        if "code" not in df.columns:
            raise ValueError("Parameter file must contain a 'code' column.")
        df = df.set_index("code")
        return df

    def _prepare_strategy_config(self, params_row: pd.Series) -> Dict[str, int | str]:
        defaults = {
            "rebalance_freq": OPTIMISATION_CONFIG["rebalance_freq"],
            "min_periods": OPTIMISATION_CONFIG["min_periods"],
            "future_days": OPTIMISATION_CONFIG["future_days"],
            "ic_lookback_period": OPTIMISATION_CONFIG["ic_lookback_period"],
        }
        params = defaults.copy()
        for key in defaults:
            if key in params_row and not pd.isna(params_row[key]):
                value = params_row[key]
                if key == "rebalance_freq":
                    params[key] = str(value)
                else:
                    params[key] = int(value)
        return params

    def run(
        self, price_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        param_table = self.load_parameters()
        zscore_window = max(int(self.config.daily.get("zscore_window", 5)), 2)
        history_window = int(self.config.daily.get("history_window", 60))
        min_history_points = int(self.config.daily.get("min_history_points", 10))

        signal_history = {}
        summary_records = []

        for code, price_df in price_map.items():
            params_row = param_table.loc[code] if code in param_table.index else None
            params = (
                self._prepare_strategy_config(params_row)
                if params_row is not None
                else self._prepare_strategy_config(pd.Series(dtype=float))
            )

            factor_config = copy.deepcopy(FACTOR_CONFIG)
            if "ic_threshold" in self.config.strategy:
                factor_config["ic_threshold"] = self.config.strategy["ic_threshold"]
            if "operator_scope" in self.config.strategy:
                factor_config["operator_scope"] = self.config.strategy["operator_scope"]
            if "operators" in self.config.strategy:
                factor_config["operators"] = self.config.strategy["operators"]

            optimisation_config = {**OPTIMISATION_CONFIG, **params}
            if "weight_bounds" in self.config.strategy:
                bounds = self.config.strategy["weight_bounds"]
                if isinstance(bounds, list):
                    bounds = tuple(bounds)
                optimisation_config["weight_bounds"] = bounds
            if "fallback_strategy" in self.config.strategy:
                optimisation_config["fallback_strategy"] = self.config.strategy[
                    "fallback_strategy"
                ]
            if "max_periods" in self.config.strategy:
                optimisation_config["max_periods"] = self.config.strategy["max_periods"]

            strategy = EnhancedMeanVarianceStrategy(
                index_df=price_df,
                optimisation_config=optimisation_config,
                factor_config=factor_config,
                logger=LOGGER,
            )
            future_position_series, _ = strategy.generate_future_position()
            future_position_series = future_position_series.sort_index()

            rolling_mean = future_position_series.rolling(
                zscore_window, min_periods=min_history_points
            ).mean()
            rolling_std = future_position_series.rolling(
                zscore_window, min_periods=min_history_points
            ).std(ddof=0)
            zscore = (future_position_series - rolling_mean) / rolling_std.replace(
                0, np.nan
            )

            signal_history[code] = pd.DataFrame(
                {
                    "signal": future_position_series,
                    "ma": rolling_mean,
                    "zscore": zscore,
                }
            )

            if not future_position_series.empty:
                last_date = future_position_series.index[-1]
                summary_records.append(
                    {
                        "code": code,
                        "date": last_date,
                        "signal": future_position_series.iloc[-1],
                        "signal_ma": rolling_mean.iloc[-1],
                        "signal_zscore": zscore.iloc[-1],
                    }
                )

        summary_df = pd.DataFrame(summary_records).sort_values(
            by="signal_zscore", ascending=False
        )

        history_frames = []
        for code, df in signal_history.items():
            df = df.tail(history_window)
            df["code"] = code
            history_frames.append(df)
        if history_frames:
            history_concat = (
                pd.concat(history_frames).set_index("code", append=True).swaplevel()
            )
        else:
            history_concat = pd.DataFrame(columns=["signal", "ma", "zscore"])

        heatmap_matrix = (
            history_concat["zscore"].unstack(0).tail(history_window)
            if not history_concat.empty
            else pd.DataFrame()
        )

        return {
            "summary": summary_df,
            "history": history_concat,
            "heatmap": heatmap_matrix,
        }

    # ------------------------------------------------------------------
    def export(
        self,
        results: Dict[str, pd.DataFrame],
        *,
        heatmap_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        today = datetime.now().strftime("%Y%m%d")
        daily_dir = self.config.daily_output_dir() / today
        daily_dir.mkdir(parents=True, exist_ok=True)

        summary_df = results.get("summary", pd.DataFrame())
        history_df = results.get("history", pd.DataFrame())

        excel_name = self.config.reporting.get(
            "excel_filename", "每日信号矩阵.xlsx"
        )
        excel_path = daily_dir / excel_name

        heatmap_matrix = heatmap_data if heatmap_data is not None else results.get("heatmap")
        heatmap_path = None
        if heatmap_matrix is not None and not heatmap_matrix.empty:
            heatmap_name = self.config.reporting.get(
                "heatmap_filename", "每日信号热力图.png"
            )
            heatmap_path = daily_dir / heatmap_name
            create_heatmap(
                heatmap_matrix,
                heatmap_path,
                title=f"Signal Z-Score Heatmap ({today})",
            )

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            if not history_df.empty:
                pivot_signal = history_df["signal"].unstack(0)
                pivot_zscore = history_df["zscore"].unstack(0)
                pivot_signal.to_excel(writer, sheet_name="signal_history")
                pivot_zscore.to_excel(writer, sheet_name="zscore_history")
            if heatmap_path:
                worksheet = writer.book.add_worksheet("heatmap")
                worksheet.insert_image("A1", str(heatmap_path))

        LOGGER.info("Saved daily signal workbook to %s", excel_path)

        return {
            "excel": excel_path,
            "heatmap": heatmap_path,
        }
