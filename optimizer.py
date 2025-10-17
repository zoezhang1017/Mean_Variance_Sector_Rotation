from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from mean_variance_system.config import (
    BACKTEST_CONFIG,
    FACTOR_CONFIG,
    OPTIMISATION_CONFIG,
)
from mean_variance_system.backtest import PositionBacktest

from .factor_engine import EnhancedMeanVarianceStrategy
from .settings import PipelineConfig

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Integer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "scikit-optimize is required for SharpeParameterOptimizer. "
        "Install via `pip install scikit-optimize`."
    ) from exc

LOGGER = logging.getLogger("next_pipeline.optimizer")

METRIC_KEYS = [
    "total_return",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "sharpe",
    "calmar",
    "win_rate",
    "profit_ratio",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TrialRecord:
    params: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_loss: float
    test_sharpe: float


@dataclass
class OptimizationResult:
    code: str
    best_params: Dict[str, Any]
    best_trial: TrialRecord
    trials: List[TrialRecord]
    price_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "best_params": self.best_params,
            "best_trial": {
                "val_loss": self.best_trial.val_loss,
                "test_sharpe": self.best_trial.test_sharpe,
            },
            "price_rows": self.price_rows,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def split_dataset(
    price_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    sorted_df = price_df.sort_index()
    n = len(sorted_df)
    if n == 0:
        return (
            sorted_df,
            sorted_df.iloc[0:0],
            sorted_df.iloc[0:0],
        )

    train_end = max(int(n * train_ratio), 1)
    val_end = train_end + max(int(n * val_ratio), 1)
    val_end = min(val_end, n)

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]
    return train_df, val_df, test_df


def compute_metrics(returns: pd.Series, risk_free_rate: float) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {key: float("nan") for key in METRIC_KEYS}

    value_curve = (1 + returns).cumprod()
    ending_value = value_curve.iloc[-1]
    total_return = ending_value - 1
    ann_factor = 252 / max(len(returns), 1)
    annual_return = (ending_value ** ann_factor) - 1
    std = returns.std(ddof=0)
    annual_volatility = std * np.sqrt(252) if np.isfinite(std) else float("nan")

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    excess_std = excess.std(ddof=0)
    if excess_std and excess_std > 1e-12:
        sharpe = excess.mean() / excess_std * np.sqrt(252)
    else:
        sharpe = float("nan")

    running_max = value_curve.cummax()
    drawdown = value_curve / running_max - 1
    max_drawdown = drawdown.min()
    calmar = (
        annual_return / abs(max_drawdown)
        if max_drawdown < 0 and abs(max_drawdown) > 1e-12
        else float("nan")
    )

    win_rate = (returns > 0).mean()
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    avg_profit = positive.mean() if not positive.empty else float("nan")
    avg_loss = negative.mean() if not negative.empty else float("nan")
    if np.isfinite(avg_profit) and np.isfinite(avg_loss) and avg_loss < 0:
        profit_ratio = avg_profit / abs(avg_loss)
    else:
        profit_ratio = float("nan")

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_ratio": float(profit_ratio),
    }


def evaluate_candidate(
    price_df: pd.DataFrame,
    params: Dict[str, Any],
    *,
    factor_config: Dict[str, Any],
    optimisation_config: Dict[str, Any],
    risk_free_rate: float,
    cost_rate: float,
    train_ratio: float,
    val_ratio: float,
) -> TrialRecord:
    train_df, val_df, test_df = split_dataset(
        price_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    splits = {
        "train": (train_df, train_df),
        "val": (pd.concat([train_df, val_df]), val_df),
        "test": (pd.concat([train_df, val_df, test_df]), test_df),
    }

    metrics = {}
    val_loss = float("inf")
    test_sharpe = float("nan")

    for name, (context_df, evaluation_df) in splits.items():
        if evaluation_df.empty:
            metrics[name] = {key: float("nan") for key in METRIC_KEYS}
            continue

        strategy = EnhancedMeanVarianceStrategy(
            index_df=context_df,
            optimisation_config={**optimisation_config, **params},
            factor_config=factor_config,
            logger=LOGGER,
        )
        future_position_series, _ = strategy.generate_future_position()
        position_series = (
            future_position_series.shift(1)
            .reindex(context_df.index)
            .fillna(0)
        )

        backtester = PositionBacktest(
            position_series=position_series,
            price_df=context_df,
            cost_rate=cost_rate,
            risk_free_rate=risk_free_rate,
            logger=LOGGER,
        )
        backtester.run_backtest()
        returns = backtester.daily_returns.loc[evaluation_df.index]
        metrics[name] = compute_metrics(returns, risk_free_rate)

        if name == "val":
            val_sharpe = metrics[name]["sharpe"]
            val_loss = (
                -val_sharpe if np.isfinite(val_sharpe) else float("inf")
            )
        if name == "test":
            test_sharpe = metrics[name]["sharpe"]

    return TrialRecord(
        params=params,
        train_metrics=metrics["train"],
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        val_loss=val_loss,
        test_sharpe=test_sharpe,
    )


def _build_params(param_values: Iterable[int]) -> Dict[str, Any]:
    rebalance_days, min_periods, future_days, ic_lookback = param_values
    return {
        "rebalance_freq": f"{int(rebalance_days)}D",
        "min_periods": int(min_periods),
        "future_days": int(future_days),
        "ic_lookback_period": int(ic_lookback),
    }


def _objective_factory(
    price_df: pd.DataFrame,
    factor_config: Dict[str, Any],
    optimisation_config: Dict[str, Any],
    risk_free_rate: float,
    cost_rate: float,
    trials: List[TrialRecord],
    train_ratio: float,
    val_ratio: float,
):
    def objective(param_values: List[int]) -> float:
        params = _build_params(param_values)
        # Prepare optimisation config with ratios stored
        optimisation_copy = dict(optimisation_config)
        optimisation_copy["__train_ratio_backup__"] = optimisation_copy["train_ratio"]
        optimisation_copy["__val_ratio_backup__"] = optimisation_copy["val_ratio"]
        trial = evaluate_candidate(
            price_df,
            params,
            factor_config=factor_config,
            optimisation_config=optimisation_copy,
            risk_free_rate=risk_free_rate,
            cost_rate=cost_rate,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        trials.append(trial)
        return trial.val_loss

    return objective


def _optimise_single_index(
    code: str,
    price_df: pd.DataFrame,
    optimisation_cfg: Dict[str, Any],
    factor_config: Dict[str, Any],
    strategy_overrides: Dict[str, Any],
) -> OptimizationResult:
    LOGGER.info("Optimising parameters for %s", code)
    trials: List[TrialRecord] = []

    strategy_cfg = dict(strategy_overrides)
    if "weight_bounds" in strategy_cfg:
        bounds = strategy_cfg["weight_bounds"]
        if isinstance(bounds, list):
            strategy_cfg["weight_bounds"] = tuple(bounds)

    base_optimisation_config = {
        **OPTIMISATION_CONFIG,
        **strategy_cfg,
        **{
            "train_ratio": optimisation_cfg.get("train_ratio", 0.8),
            "val_ratio": optimisation_cfg.get("val_ratio", 0.1),
        },
    }

    risk_free_rate = optimisation_cfg.get(
        "risk_free_rate", BACKTEST_CONFIG["risk_free_rate"]
    )
    cost_rate = optimisation_cfg.get("cost_rate", BACKTEST_CONFIG["cost_rate"])
    train_ratio = optimisation_cfg.get("train_ratio", 0.8)
    val_ratio = optimisation_cfg.get("val_ratio", 0.1)

    param_space = optimisation_cfg.get("param_space")
    if param_space is None:
        param_space = {
            "rebalance_days": optimisation_cfg.get("rebalance_days", [5, 60]),
            "min_periods": optimisation_cfg.get("min_periods", [60, 360]),
            "future_days": optimisation_cfg.get("future_days", [1, 3]),
            "ic_lookback_period": optimisation_cfg.get("ic_lookback_period", [10, 90]),
        }

    dimensions = [
        Integer(low, high, name=name)
        for name, (low, high) in param_space.items()
    ]

    objective = _objective_factory(
        price_df=price_df,
        factor_config=factor_config,
        optimisation_config=base_optimisation_config,
        risk_free_rate=risk_free_rate,
        cost_rate=cost_rate,
        trials=trials,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    n_calls = optimisation_cfg.get("n_calls", 30)
    method = optimisation_cfg.get("method", "gp")
    random_state = optimisation_cfg.get("random_state", 42)

    if method == "forest":
        forest_minimize(
            objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
        )
    else:
        gp_minimize(
            objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
        )

    if not trials:
        raise RuntimeError(f"No trials generated for {code}")

    best_trial = max(trials, key=lambda t: t.test_sharpe)
    LOGGER.info(
        "Best params for %s (test sharpe %.3f): %s",
        code,
        best_trial.test_sharpe,
        best_trial.params,
    )

    return OptimizationResult(
        code=code,
        best_params=best_trial.params,
        best_trial=best_trial,
        trials=trials,
        price_rows=len(price_df),
    )


# ---------------------------------------------------------------------------
# Public optimiser
# ---------------------------------------------------------------------------
@dataclass
class SharpeParameterOptimizer:
    config: PipelineConfig
    factor_config: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(FACTOR_CONFIG)
    )

    def __post_init__(self) -> None:
        strategy_cfg = self.config.strategy
        if "ic_threshold" in strategy_cfg:
            self.factor_config["ic_threshold"] = strategy_cfg["ic_threshold"]
        if "operator_scope" in strategy_cfg:
            self.factor_config["operator_scope"] = strategy_cfg["operator_scope"]
        if "operators" in strategy_cfg:
            self.factor_config["operators"] = strategy_cfg["operators"]

    def optimise(
        self, price_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, OptimizationResult]:
        optimisation_cfg = self.config.optimisation
        strategy_overrides = self.config.strategy
        n_jobs = int(optimisation_cfg.get("n_jobs", 1))

        results: Dict[str, OptimizationResult] = {}

        if n_jobs <= 1:
            for code, price_df in price_map.items():
                results[code] = _optimise_single_index(
                    code,
                    price_df,
                    optimisation_cfg,
                    self.factor_config,
                    strategy_overrides,
                )
            return results

        LOGGER.info("Running optimisation in parallel with %d workers", n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _optimise_single_index,
                    code,
                    price_df,
                    optimisation_cfg,
                    self.factor_config,
                    strategy_overrides,
                ): code
                for code, price_df in price_map.items()
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result()
                except Exception as exc:
                    LOGGER.exception("Optimisation failed for %s: %s", code, exc)
        return results

    # ------------------------------------------------------------------
    def save_results(
        self,
        results: Dict[str, OptimizationResult],
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        output_base = (
            Path(output_dir) if output_dir else self.config.optimisation_output_dir()
        )
        output_base.mkdir(parents=True, exist_ok=True)

        params_records = []
        for code, result in results.items():
            params_records.append(
                {
                    "code": code,
                    **result.best_params,
                    "test_sharpe": result.best_trial.test_sharpe,
                    "val_loss": result.best_trial.val_loss,
                }
            )

        params_df = pd.DataFrame(params_records)
        params_path = output_base / "optimised_parameters.xlsx"
        params_df.to_excel(params_path, index=False)
        LOGGER.info("Saved parameter table to %s", params_path)

        history_path = output_base / self.config.reporting.get(
            "optimisation_log", "optimisation_history.json"
        )
        history = {
            code: [
                {
                    "params": trial.params,
                    "val_loss": trial.val_loss,
                    "test_sharpe": trial.test_sharpe,
                    "train_metrics": trial.train_metrics,
                    "val_metrics": trial.val_metrics,
                    "test_metrics": trial.test_metrics,
                }
                for trial in result.trials
            ]
            for code, result in results.items()
        }
        with history_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        LOGGER.info("Saved optimisation history to %s", history_path)

        return params_path
