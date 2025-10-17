from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

from mean_variance_system.strategy import MeanVarianceStrategy
from mean_variance_system.factors.factor_enhancer import (
    FactorEnhancer as BaseFactorEnhancer,
)


LOGGER = logging.getLogger("next_pipeline.factor")


class EfficientFactorEnhancer(BaseFactorEnhancer):
    """Drop-in replacement that avoids pandas fragmentation warnings."""

    def apply(self, factors: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        if factors.empty:
            self.logger.warning("No factors provided for enhancement")
            return factors

        enhanced = factors.copy()
        new_columns: dict[str, pd.Series] = {}

        for factor_name in factors.columns:
            series = factors[factor_name]
            specs = self._specs_for_factor(factor_name)
            if not specs:
                continue

            for spec in specs:
                operator = self.operators[spec.name]
                transformed = operator(series, spec.params)
                new_col = f"{factor_name}_{spec.suffix}"
                new_columns[new_col] = transformed
                self.logger.debug(
                    "Applied operator %s to %s -> %s",
                    spec.name,
                    factor_name,
                    new_col,
                )

        if new_columns:
            new_df = pd.DataFrame(new_columns, index=enhanced.index)
            enhanced = pd.concat([enhanced, new_df], axis=1)

        enhanced = enhanced.replace([np.inf, -np.inf], np.nan)
        return enhanced


class EnhancedMeanVarianceStrategy(MeanVarianceStrategy):
    """MeanVarianceStrategy using the fragmentation-safe factor enhancer."""

    def _enhance_signals(self, signals: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        enhancer = EfficientFactorEnhancer(
            operator_specs=self.factor_config.get("operators", []),
            operator_scope=self.factor_config.get("operator_scope"),
            logger=self.logger,
        )
        enhanced = enhancer.apply(signals)
        self.logger.info(
            "Enhanced factors from %d to %d columns (fragmentation safe)",
            signals.shape[1],
            enhanced.shape[1],
        )
        return enhanced.fillna(0)

    def _generate_rebalance_dates(  # type: ignore[override]
        self, dates: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Ensure rebalance dates are always returned as a DatetimeIndex."""
        freq = self.optimisation_config["rebalance_freq"]

        if isinstance(freq, str):
            rebalance_series = (
                dates.to_series().resample(freq).first().dropna()
            )
            if isinstance(rebalance_series, pd.Series):
                rebalance_idx = pd.DatetimeIndex(rebalance_series.values)
            else:
                rebalance_idx = pd.DatetimeIndex(rebalance_series)
        else:
            step = max(int(freq), 1)
            all_dates = pd.date_range(start=dates[0], end=dates[-1], freq="D")
            rebalance_idx = all_dates[::step]
            rebalance_idx = rebalance_idx[rebalance_idx.isin(dates)]
            if len(rebalance_idx) and rebalance_idx[0] < dates[0]:
                rebalance_idx = rebalance_idx[1:]

        if rebalance_idx.empty:
            rebalance_idx = pd.DatetimeIndex([dates[-1]])
        elif rebalance_idx[-1] != dates[-1]:
            rebalance_idx = rebalance_idx.append(pd.DatetimeIndex([dates[-1]]))

        return rebalance_idx
