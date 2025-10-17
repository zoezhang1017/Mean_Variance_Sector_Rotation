from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skopt import forest_minimize, gp_minimize
from skopt.space import Integer

from .backtest import PositionBacktest
from .config import (
    BACKTEST_CONFIG,
    FACTOR_CONFIG,
    DATA_SPLIT_CONFIG,
    CROSS_VALIDATION_CONFIG,
    OPTIMISED_PARAMS_PATH,
    OPTIMISATION_CONFIG,
)
from .logging_config import setup_logging
from .strategy import MeanVarianceStrategy


DEFAULT_PARAM_SPACE: Sequence[Integer] = (
    Integer(5, 60, name="rebalance_days"),
    Integer(60, 360, name="min_periods"),
    Integer(1, 5, name="future_days"),
    Integer(10, 90, name="ic_lookback_period"),
)


def _build_optimisation_config(
    params: Sequence[int],
    template: Optional[Dict] = None,
) -> Dict:
    rebalance_days, min_periods, future_days, ic_lookback = params
    config = dict(template or OPTIMISATION_CONFIG)
    config.update(
        {
            "rebalance_freq": f"{int(rebalance_days)}D",
            "min_periods": int(min_periods),
            "future_days": int(future_days),
            "ic_lookback_period": int(ic_lookback),
        }
    )
    return config


def _compute_sharpe(returns: pd.Series, risk_free_rate: float) -> float:
    if returns.empty:
        return float("-inf")
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    mean = excess.mean()
    std = excess.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return float("-inf")
    sharpe = mean / std * np.sqrt(252)
    return float(sharpe)


def generate_time_series_folds(
    price_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    folds: int,
    min_train_ratio: float,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    if folds <= 1 or val_ratio <= 0:
        return []

    sorted_df = price_df.sort_index()
    n = len(sorted_df)
    if n == 0:
        return []

    val_size = max(int(n * val_ratio), 1)
    min_train_size = max(int(n * min_train_ratio), 1)
    base_train_size = max(int(n * train_ratio), min_train_size)

    folds_data: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold in range(folds):
        val_start = base_train_size + fold * val_size
        val_end = val_start + val_size
        if val_end > n:
            break
        train_end = val_start
        if train_end < min_train_size:
            continue
        train_df = sorted_df.iloc[:train_end]
        val_df = sorted_df.iloc[val_start:val_end]
        if train_df.empty or val_df.empty:
            continue
        folds_data.append((train_df, val_df))

    return folds_data


def objective(
    params: Sequence[int],
    price_df: pd.DataFrame,
    optimisation_template: Optional[Dict] = None,
    factor_template: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    train_ratio: float = DATA_SPLIT_CONFIG["train_ratio"],
    val_ratio: float = DATA_SPLIT_CONFIG["val_ratio"],
    folds: int = CROSS_VALIDATION_CONFIG["folds"],
    min_train_ratio: float = CROSS_VALIDATION_CONFIG["min_train_ratio"],
) -> float:
    logger = logger or logging.getLogger("mean_variance")
    price_df_sorted = price_df.sort_index()
    folds_data = generate_time_series_folds(
        price_df_sorted,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        folds=folds,
        min_train_ratio=min_train_ratio,
    )

    if not folds_data:
        train_df, val_df, _ = split_dataset(
            price_df_sorted,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        if val_df.empty or train_df.empty:
            logger.warning("Insufficient data for split; returning penalty.")
            return 1e6
        folds_data = [(train_df, val_df)]

    optimisation_config = _build_optimisation_config(params, optimisation_template)

    sharpe_scores: List[float] = []

    for train_df, val_df in folds_data:
        evaluation_df = pd.concat([train_df, val_df])
        strategy = MeanVarianceStrategy(
            index_df=evaluation_df,
            optimisation_config=optimisation_config,
            factor_config=factor_template or FACTOR_CONFIG,
            logger=logger,
        )
        future_position_series, _ = strategy.generate_future_position()

        backtester = PositionBacktest(
            position_series=future_position_series.shift(1).fillna(0),
            price_df=evaluation_df,
            cost_rate=BACKTEST_CONFIG["cost_rate"],
            risk_free_rate=BACKTEST_CONFIG["risk_free_rate"],
            logger=logger,
        )
        backtester.run_backtest()
        validation_returns = backtester.daily_returns.loc[val_df.index]
        sharpe = _compute_sharpe(
            validation_returns,
            BACKTEST_CONFIG["risk_free_rate"],
        )
        if np.isfinite(sharpe):
            sharpe_scores.append(sharpe)
        else:
            sharpe_scores.append(float("-inf"))

    valid_scores = [s for s in sharpe_scores if np.isfinite(s)]
    if not valid_scores:
        logger.warning("All folds produced invalid Sharpe ratios; returning penalty.")
        return 1e6

    mean_sharpe = float(np.mean(valid_scores))
    return -mean_sharpe


def optimise_parameters(
    price_df: pd.DataFrame,
    param_space: Sequence[Integer] = DEFAULT_PARAM_SPACE,
    n_calls: int = 25,
    random_state: int = 42,
    method: str = "gp",
    optimisation_template: Optional[Dict] = None,
    factor_template: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    train_ratio: float = DATA_SPLIT_CONFIG["train_ratio"],
    val_ratio: float = DATA_SPLIT_CONFIG["val_ratio"],
    folds: int = CROSS_VALIDATION_CONFIG["folds"],
    min_train_ratio: float = CROSS_VALIDATION_CONFIG["min_train_ratio"],
):
    logger = logger or setup_logging()

    objective_fn = lambda params: objective(
        params=params,
        price_df=price_df,
        optimisation_template=optimisation_template,
        factor_template=factor_template,
        logger=logger,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        folds=folds,
        min_train_ratio=min_train_ratio,
    )

    if method == "forest":
        result = forest_minimize(
            objective_fn,
            dimensions=list(param_space),
            n_calls=n_calls,
            random_state=random_state,
        )
    else:
        result = gp_minimize(
            objective_fn,
            dimensions=list(param_space),
            n_calls=n_calls,
            random_state=random_state,
        )

    return result


def extract_parameters(result) -> Dict[str, int | str]:
    rebalance_days, min_periods, future_days, ic_lookback = result.x
    return {
        "rebalance_freq": f"{int(rebalance_days)}D",
        "min_periods": int(min_periods),
        "future_days": int(future_days),
        "ic_lookback_period": int(ic_lookback),
    }


def save_parameters(
    params: Dict[str, int | str],
    path: Path = OPTIMISED_PARAMS_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return path


def optimise_and_save(
    price_df: pd.DataFrame,
    param_space: Sequence[Integer] = DEFAULT_PARAM_SPACE,
    n_calls: int = 25,
    random_state: int = 42,
    method: str = "gp",
    optimisation_template: Optional[Dict] = None,
    factor_template: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    train_ratio: float = DATA_SPLIT_CONFIG["train_ratio"],
    val_ratio: float = DATA_SPLIT_CONFIG["val_ratio"],
    folds: int = CROSS_VALIDATION_CONFIG["folds"],
    min_train_ratio: float = CROSS_VALIDATION_CONFIG["min_train_ratio"],
) -> Dict[str, int | str]:
    result = optimise_parameters(
        price_df=price_df,
        param_space=param_space,
        n_calls=n_calls,
        random_state=random_state,
        method=method,
        optimisation_template=optimisation_template,
        factor_template=factor_template,
        logger=logger,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        folds=folds,
        min_train_ratio=min_train_ratio,
    )
    params = extract_parameters(result)
    save_parameters(params)
    return params


def split_dataset(
    price_df: pd.DataFrame,
    train_ratio: float = DATA_SPLIT_CONFIG["train_ratio"],
    val_ratio: float = DATA_SPLIT_CONFIG["val_ratio"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    sorted_df = price_df.sort_index()
    n = len(sorted_df)
    if n == 0:
        return sorted_df, sorted_df.iloc[0:0], sorted_df.iloc[0:0]

    train_end = max(int(n * train_ratio), 1)
    val_end = train_end + max(int(n * val_ratio), 1)
    val_end = min(val_end, n)

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]
    return train_df, val_df, test_df


def evaluate_on_splits(
    price_df: pd.DataFrame,
    params: Dict[str, int | str],
    factor_config: Optional[Dict] = None,
    train_ratio: float = DATA_SPLIT_CONFIG["train_ratio"],
    val_ratio: float = DATA_SPLIT_CONFIG["val_ratio"],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, str]]:
    logger = logger or setup_logging()
    price_df_sorted = price_df.sort_index()
    train_df, val_df, test_df = split_dataset(
        price_df_sorted,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    results: Dict[str, Dict[str, str]] = {}

    # Training evaluation
    if train_df.empty:
        logger.warning("Training split empty; unable to evaluate.")
        results["train"] = {}
    else:
        train_strategy = MeanVarianceStrategy(
            index_df=train_df,
            optimisation_config={**OPTIMISATION_CONFIG, **params},
            factor_config=factor_config or FACTOR_CONFIG,
            logger=logger,
        )
        train_future_position, _ = train_strategy.generate_future_position()
        train_backtester = PositionBacktest(
            position_series=train_future_position.shift(1).fillna(0),
            price_df=train_df,
            cost_rate=BACKTEST_CONFIG["cost_rate"],
            risk_free_rate=BACKTEST_CONFIG["risk_free_rate"],
            logger=logger,
        )
        train_metrics = train_backtester.run_full_backtest().loc["全样本"].to_dict()
        results["train"] = train_metrics

    # Validation evaluation (train + val, metrics on val)
    if val_df.empty:
        logger.warning("Validation split empty; skipping validation metrics.")
        results["val"] = {}
    else:
        combined_val = pd.concat([train_df, val_df])
        val_strategy = MeanVarianceStrategy(
            index_df=combined_val,
            optimisation_config={**OPTIMISATION_CONFIG, **params},
            factor_config=factor_config or FACTOR_CONFIG,
            logger=logger,
        )
        val_future_position, _ = val_strategy.generate_future_position()
        val_backtester = PositionBacktest(
            position_series=val_future_position.shift(1).fillna(0),
            price_df=combined_val,
            cost_rate=BACKTEST_CONFIG["cost_rate"],
            risk_free_rate=BACKTEST_CONFIG["risk_free_rate"],
            logger=logger,
        )
        val_backtester.run_backtest()
        val_returns = val_backtester.daily_returns.loc[val_df.index]
        val_metrics = val_backtester.calculate_metrics(val_returns)
        results["val"] = val_metrics

    # Test evaluation (train + val + test, metrics on test)
    if test_df.empty:
        logger.warning("Test split empty; skipping test metrics.")
        results["test"] = {}
    else:
        combined_test = pd.concat([train_df, val_df, test_df])
        test_strategy = MeanVarianceStrategy(
            index_df=combined_test,
            optimisation_config={**OPTIMISATION_CONFIG, **params},
            factor_config=factor_config or FACTOR_CONFIG,
            logger=logger,
        )
        test_future_position, _ = test_strategy.generate_future_position()
        test_backtester = PositionBacktest(
            position_series=test_future_position.shift(1).fillna(0),
            price_df=combined_test,
            cost_rate=BACKTEST_CONFIG["cost_rate"],
            risk_free_rate=BACKTEST_CONFIG["risk_free_rate"],
            logger=logger,
        )
        test_backtester.run_backtest()
        test_returns = test_backtester.daily_returns.loc[test_df.index]
        test_metrics = test_backtester.calculate_metrics(test_returns)
        results["test"] = test_metrics

    return results
