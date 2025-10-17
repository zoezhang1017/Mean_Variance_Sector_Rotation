from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Iterable, Mapping, Union

import numpy as np
import pandas as pd
import warnings
import cvxpy
import scipy.linalg
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from .config import FACTOR_CONFIG, OPTIMISATION_CONFIG
from .factors import FactorEnhancer, TechnicalIndicators


class MeanVarianceStrategy:
    """
    Mean Variance portfolio construction using enhanced technical factors.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        factor_config: Mapping | None = None,
        optimisation_config: Mapping | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("mean_variance")
        self.index_df = index_df.copy()
        self.index_df["daily_return"] = (
            (self.index_df["close"] / self.index_df["prev_close"]) - 1.0
        ).fillna(0.0)

        self.factor_config = dict(FACTOR_CONFIG)
        if factor_config:
            self.factor_config.update(factor_config)

        self.optimisation_config = dict(OPTIMISATION_CONFIG)
        if optimisation_config:
            self.optimisation_config.update(optimisation_config)

        self.logger.info("Initialising MeanVarianceStrategy with %d rows", len(self.index_df))
        self.raw_signals = self._compute_signals()
        self.signals = self._enhance_signals(self.raw_signals)
        (
            self.future_position_df,
            self.position_df,
        ) = self._generate_position_df(self.signals)
        self.signals_return = self.position_df.mul(
            self.index_df["daily_return"], axis=0
        )
        self.weights_df: pd.DataFrame | None = None
        self.optimization_errors = 0
        self.fallback_used = 0
        self.future_position_series: pd.Series | None = None

    def _compute_signals(self) -> pd.DataFrame:
        ti = TechnicalIndicators(self.index_df.copy())
        active = self.factor_config.get("active_factors")
        signal_functions: Iterable[str]

        if active:
            signal_functions = active
        else:
            signal_functions = [
                name
                for name in dir(ti)
                if not name.startswith("_")
                and callable(getattr(ti, name))
                and name not in {"_precalculate", "_cross_above", "_cross_below", "_ma", "_ema", "_sma"}
            ]

        signals: Dict[str, np.ndarray] = {}
        for name in signal_functions:
            func = getattr(ti, name, None)
            if func is None or not callable(func):
                self.logger.debug("Skipping non-callable indicator %s", name)
                continue
            try:
                signals[name] = func()
            except Exception as exc:
                self.logger.exception("Indicator %s failed: %s", name, exc)

        signals_df = pd.DataFrame(signals, index=self.index_df.index).fillna(0)
        self.logger.info("Computed %d base factor signals", signals_df.shape[1])
        return signals_df

    def _enhance_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        enhancer = FactorEnhancer(
            operator_specs=self.factor_config.get("operators", []),
            operator_scope=self.factor_config.get("operator_scope"),
            logger=self.logger,
        )
        enhanced = enhancer.apply(signals)
        self.logger.info("Enhanced factors from %d to %d columns", signals.shape[1], enhanced.shape[1])
        return enhanced.fillna(0)

    @staticmethod
    def _generate_position(signals_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Generate forward and aligned position series given signals."""
        position = pd.Series(0, index=signals_series.index, dtype=float)
        current_pos = 0.0

        for idx, sig in enumerate(signals_series):
            if sig == 1:
                current_pos = 1.0
            elif sig == -1:
                current_pos = 0.0
            position.iloc[idx] = current_pos

        return position.copy(), position.shift(1).fillna(0.0)

    def _generate_position_df(
        self, signals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        future_position_dict = {}
        position_dict = {}
        for col in signals.columns:
            future_position, position = self._generate_position(signals[col])
            future_position_dict[col] = future_position
            position_dict[col] = position
        return pd.DataFrame(future_position_dict), pd.DataFrame(position_dict)

    def get_signals_return(self) -> pd.DataFrame:
        """Return daily returns for each signal."""
        return self.signals_return

    def _generate_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        freq = self.optimisation_config["rebalance_freq"]
        if isinstance(freq, str):
            rebalance_dates = (
                dates.to_series()
                .resample(freq)
                .first()
                .dropna()
                .values
            )
        else:
            start_date = dates[0]
            end_date = dates[-1]
            all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
            rebalance_dates = all_dates[:: int(freq)]
            if rebalance_dates[0] < dates[0]:
                rebalance_dates = rebalance_dates[1:]
            rebalance_dates = rebalance_dates[rebalance_dates.isin(dates)]
            if rebalance_dates[-1] != dates[-1]:
                rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([dates[-1]]))
            return rebalance_dates

        if rebalance_dates[-1] != dates[-1]:
            rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([dates[-1]]))
        return rebalance_dates

    def _optimize_portfolio(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        weight_bounds: tuple[float, float],
        fallback_strategy: str,
    ) -> Dict[str, float]:
        n_assets = len(mu)
        solvers = [cvxpy.OSQP, cvxpy.ECOS, cvxpy.SCS]
        solver_options = {"max_iter": 50_000, "eps_abs": 1e-8, "eps_rel": 1e-8, "verbose": False}

        for solver in solvers[: self.optimisation_config.get("max_optimizer_attempts", 3)]:
            try:
                ef = EfficientFrontier(mu, cov, solver=solver, weight_bounds=weight_bounds)
                weights = ef.max_sharpe(solver_options=solver_options)
                return ef.clean_weights()
            except Exception as exc:
                self.logger.debug("Solver %s failed: %s", solver, exc)
                continue

        try:
            cov_reg = cov + 0.01 * np.eye(cov.shape[0])
            ef = EfficientFrontier(mu, cov_reg, solver=cvxpy.ECOS, weight_bounds=weight_bounds)
            weights = ef.max_sharpe()
            return ef.clean_weights()
        except Exception as exc:
            self.logger.debug("Regularised optimisation failed: %s", exc)

        if fallback_strategy == "min_volatility":
            try:
                ef = EfficientFrontier(mu, cov, solver=cvxpy.ECOS, weight_bounds=weight_bounds)
                weights = ef.min_volatility()
                self.fallback_used += 1
                return ef.clean_weights()
            except Exception as exc:
                self.logger.debug("Min volatility fallback failed: %s", exc)

        self.logger.warning("Using equal weight fallback")
        self.fallback_used += 1
        return {ticker: 1 / n_assets for ticker in mu.index}

    @staticmethod
    def clean_historical_data(
        historical_data: pd.DataFrame, tol: float = 1e-8
    ) -> pd.DataFrame:
        non_zero_cols = historical_data.columns[
            (historical_data.abs().max(axis=0) > tol)
        ]
        historical_data_clean = historical_data[non_zero_cols].copy()
        matrix = historical_data_clean.values
        rank = np.linalg.matrix_rank(matrix, tol=tol)

        if rank == len(historical_data_clean.columns):
            return historical_data_clean

        _, _, pivot = scipy.linalg.qr(matrix, mode="economic", pivoting=True)
        independent_cols = np.sort(pivot[:rank])
        full_rank_cols = historical_data_clean.columns[independent_cols]
        return historical_data_clean[full_rank_cols]

    def generate_future_position(self) -> tuple[pd.Series, pd.DataFrame]:
        returns = self.signals_return.copy()
        future_position_df = self.future_position_df.copy()
        dates = future_position_df.index
        signals = self.signals.copy()
        index_df = self.index_df.copy()

        weight_bounds = tuple(self.optimisation_config["weight_bounds"])
        fallback_strategy = self.optimisation_config["fallback_strategy"]
        min_periods = self.optimisation_config["min_periods"]
        max_periods = self.optimisation_config["max_periods"]
        ic_threshold = self.factor_config.get("ic_threshold", 0.01)
        lookback = self.optimisation_config["ic_lookback_period"]
        future_days = self.optimisation_config["future_days"]

        weights_df = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
        future_position_series = pd.Series(np.nan, index=dates, name="future_position_series")

        rebalance_dates = self._generate_rebalance_dates(dates)
        if len(rebalance_dates) == 0:
            self.logger.warning("No rebalance dates generated")
            self.weights_df = weights_df
            self.future_position_series = future_position_series.fillna(0)
            return self.future_position_series, weights_df

        for idx in range(2, len(rebalance_dates)):
            start_date = rebalance_dates[idx - 1]
            end_date = rebalance_dates[idx]
            historical_mask = dates <= start_date
            historical_data = returns[historical_mask]

            recent_signals = signals[historical_mask].tail(lookback)
            recent_returns = (
                index_df["daily_return"][historical_mask].shift(-future_days)
            ).tail(lookback)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ic_values = recent_signals.apply(lambda x: x.corr(recent_returns))

            if len(historical_data) < min_periods:
                continue

            filtered_data = historical_data.loc[:, ic_values > ic_threshold]
            if filtered_data.empty:
                continue

            if max_periods is None:
                historical_clean = self.clean_historical_data(filtered_data)
            else:
                historical_clean = self.clean_historical_data(filtered_data).tail(max_periods)

            try:
                mu = expected_returns.ema_historical_return(
                    historical_clean, returns_data=True
                )
                cov = risk_models.risk_matrix(
                    historical_clean, returns_data=True, method="ledoit_wolf"
                )
                weights = self._optimize_portfolio(
                    mu, cov, weight_bounds, fallback_strategy
                )
            except Exception as exc:
                self.logger.exception(
                    "Optimisation failed for %s-%s: %s", start_date, end_date, exc
                )
                self.optimization_errors += 1
                continue

            weights_vector = pd.Series(weights)
            period_mask = (dates > start_date) & (dates <= end_date)
            weights_df.loc[period_mask, weights_vector.index] = weights_vector.values

        weights_df = weights_df.dropna(how="all").fillna(0)
        future_position_series = (future_position_df * weights_df).dropna(
            how="all"
        ).sum(axis=1).fillna(0)

        self.future_position_series = future_position_series
        self.weights_df = weights_df

        self.logger.info(
            "Optimisation complete. Errors: %s, fallbacks: %s",
            self.optimization_errors,
            self.fallback_used,
        )
        return future_position_series, weights_df
