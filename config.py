"""
Central configuration for the mean variance strategy pipeline.
Defaults are intended as sane starting points and can be overridden
by user specific settings.
"""

from pathlib import Path
import json

OPTIMISED_PARAMS_PATH = Path(__file__).with_name("optimised_params.json")

# Factor generation configuration
FACTOR_CONFIG: dict = {
    # Optionally limit the set of indicators to evaluate; None uses all.
    "active_factors": None,
    # Minimum IC threshold to keep a factor in optimisation universe.
    "ic_threshold": 0.01,
    # Factor operator enhancement pipeline. Each entry describes an operator.
    "operators": [
        {"name": "zscore", "window": 20, "suffix": "z20"},
        {"name": "rolling_mean", "window": 5, "suffix": "ma5"},
        {"name": "rolling_std", "window": 5, "suffix": "std5"},
        {"name": "ema", "span": 10, "suffix": "ema10"},
        {"name": "pct_change", "periods": 5, "suffix": "pct5"},
        {"name": "identity", "suffix": "id"},
        {"name": "sin", "frequency": 1, "suffix": "sin"},
        {"name": "sin", "frequency": 2, "suffix": "sin2"},
        {"name": "sin", "frequency": 3, "suffix": "sin3"},
        {"name": "cos", "frequency": 1, "suffix": "cos"},
        {"name": "cos", "frequency": 2, "suffix": "cos2"},
        {"name": "cos", "frequency": 3, "suffix": "cos3"},
        {"name": "power", "power": 2, "suffix": "pow2"},
        {"name": "power", "power": 3, "suffix": "pow3"},
        {"name": "power", "power": 4, "suffix": "pow4"},
        {"name": "power", "power": 5, "suffix": "pow5"},
        {"name": "power", "power": 6, "suffix": "pow6"},
        {"name": "tanh", "suffix": "tanh"},
        {"name": "sigmoid", "suffix": "sigmoid01"},
        {"name": "sigmoid_symmetric", "suffix": "sigmoid11"},
        {"name": "arctan", "suffix": "atan"},
        {"name": "log1p_xsq", "suffix": "log1px2"},
        {"name": "scaled_log1p_xsq", "suffix": "log1px2s", "k": 0.5},
        {"name": "signed_log1p_abs", "suffix": "slog1p", "k": 0.5},
        {"name": "log1p_abs_pow", "suffix": "logpow", "k": 0.5, "p": 1.2},
        {"name": "softsign", "suffix": "softsign"},
        {"name": "softsign_scaled", "suffix": "softsigns", "c": 1.0, "p": 1.0},
        {"name": "arctan_scaled", "suffix": "atan_c", "c": 1.0},
        {"name": "algebraic_sigmoid", "suffix": "alsig", "c": 1.0},
        {"name": "exp_linear", "suffix": "exp001", "alpha": 0.01},
    ],
    # Operators can be selectively applied to specific factors.
    # Empty dict applies all operators to all factors.
    "operator_scope": {},
}

# Portfolio optimisation settings
OPTIMISATION_CONFIG: dict = {
    "rebalance_freq": "10D",
    "weight_bounds": (0, 0.3),
    "min_periods": 100,
    "max_periods": None,
    "fallback_strategy": "min_volatility",
    "ic_lookback_period": 20,
    "future_days": 1,
    "max_optimizer_attempts": 3,
}

# Backtest defaults
BACKTEST_CONFIG: dict = {
    "cost_rate": 0.0005,
    "risk_free_rate": 0.02,
}

# Reporting output defaults
REPORTING_CONFIG: dict = {
    "output_dir": Path("reports"),
    "docx_name": "均值方差策略结果.docx",
    "performance_filename": "绩效指标.csv",
    "trade_record_filename": "交易记录.csv",
    "latest_signal_filename": "行业指数最新信号.xlsx",
    "latest_trade_filename": "行业指数最新交易记录.csv",
}

DATA_SPLIT_CONFIG: dict = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
}

CROSS_VALIDATION_CONFIG: dict = {
    "folds": 3,
    "min_train_ratio": 0.5,
}

if OPTIMISED_PARAMS_PATH.exists():
    try:
        with OPTIMISED_PARAMS_PATH.open("r", encoding="utf-8") as f:
            optimised_params = json.load(f)
        if isinstance(optimised_params, dict):
            for key in ("rebalance_freq", "min_periods", "future_days", "ic_lookback_period"):
                if key in optimised_params:
                    OPTIMISATION_CONFIG[key] = optimised_params[key]
    except (json.JSONDecodeError, OSError):
        # Ignore malformed optimisation overrides and retain defaults.
        pass
