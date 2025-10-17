from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


OperatorFunc = Callable[[pd.Series, Mapping[str, float]], pd.Series]


def _series_scale(series: pd.Series) -> float:
    scale = float(series.std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return scale


def _zscore(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    window = int(params.get("window", 20))
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)


def _rolling_mean(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    window = int(params.get("window", 5))
    return series.rolling(window).mean()


def _rolling_std(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    window = int(params.get("window", 5))
    return series.rolling(window).std()


def _ema(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    span = params.get("span")
    if span is None:
        raise ValueError("EMA operator requires 'span' parameter")
    return series.ewm(span=span, adjust=False).mean()


def _pct_change(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    periods = int(params.get("periods", 1))
    return series.pct_change(periods)


def _identity(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    return series


def _sin(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    frequency = float(params.get("frequency", 1.0))
    return np.sin(frequency * series)


def _cos(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    frequency = float(params.get("frequency", 1.0))
    return np.cos(frequency * series)


def _power(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    exponent = params.get("power", 2)
    return np.power(series, exponent)


def _tanh(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    return np.tanh(series)


def _sigmoid(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    clipped = np.clip(series, -50, 50)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sigmoid_symmetric(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    clipped = np.clip(series, -50, 50)
    return 2.0 / (1.0 + np.exp(-clipped)) - 1.0


def _arctan(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    return np.arctan(series)


def _log1p_xsq(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    return np.log1p(np.square(series))


def _scaled_log1p_xsq(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    k = float(params.get("k", 0.5))
    scale = _series_scale(series)
    return np.log1p(np.square(series / (k * scale)))


def _signed_log1p_abs(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    k = float(params.get("k", 0.5))
    scale = _series_scale(series)
    return np.sign(series) * np.log1p(np.abs(series) / (k * scale))


def _log1p_abs_pow(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    k = float(params.get("k", 0.5))
    p = float(params.get("p", 1.2))
    scale = _series_scale(series)
    return np.log1p(np.power(np.abs(series) / (k * scale), p))


def _softsign(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    return series / (1.0 + np.abs(series))


def _softsign_scaled(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    c = float(params.get("c", 1.0))
    p = float(params.get("p", 1.0))
    denominator = 1.0 + np.power(np.abs(series) / c, p)
    return series / denominator


def _arctan_scaled(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    c = float(params.get("c", 1.0))
    return (2.0 / np.pi) * np.arctan(series / c)


def _algebraic_sigmoid(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    c = float(params.get("c", 1.0))
    return series / np.sqrt(1.0 + np.square(series / c))


def _exp_linear(series: pd.Series, params: Mapping[str, float]) -> pd.Series:
    alpha = float(params.get("alpha", 0.01))
    limit = 700 / max(abs(alpha), 1e-8)
    clipped = np.clip(series, -limit, limit)
    return np.exp(alpha * clipped)


DEFAULT_OPERATORS: Dict[str, OperatorFunc] = {
    "zscore": _zscore,
    "rolling_mean": _rolling_mean,
    "rolling_std": _rolling_std,
    "ema": _ema,
    "pct_change": _pct_change,
    "identity": _identity,
    "sin": _sin,
    "cos": _cos,
    "power": _power,
    "tanh": _tanh,
    "sigmoid": _sigmoid,
    "sigmoid_symmetric": _sigmoid_symmetric,
    "arctan": _arctan,
    "log1p_xsq": _log1p_xsq,
    "scaled_log1p_xsq": _scaled_log1p_xsq,
    "signed_log1p_abs": _signed_log1p_abs,
    "log1p_abs_pow": _log1p_abs_pow,
    "softsign": _softsign,
    "softsign_scaled": _softsign_scaled,
    "arctan_scaled": _arctan_scaled,
    "algebraic_sigmoid": _algebraic_sigmoid,
    "exp_linear": _exp_linear,
}


@dataclass
class OperatorSpec:
    name: str
    params: Dict[str, float]
    suffix: str


class FactorEnhancer:
    """
    Apply operator based transformations to base factor signals.
    """

    def __init__(
        self,
        operator_specs: Iterable[Mapping[str, float]],
        operator_scope: Mapping[str, Iterable[str]] | None = None,
        operators: Mapping[str, OperatorFunc] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("mean_variance")
        self.operators = dict(DEFAULT_OPERATORS)
        if operators:
            self.operators.update(operators)
        self.operator_specs = [self._parse_spec(spec) for spec in operator_specs]
        self.operator_scope = {
            k: set(v) for k, v in (operator_scope or {}).items()
        }

    def _parse_spec(self, spec: Mapping[str, float]) -> OperatorSpec:
        name = spec.get("name")
        if not name:
            raise ValueError("Operator specification must include 'name'")
        suffix = spec.get("suffix") or name
        params = {k: v for k, v in spec.items() if k not in {"name", "suffix"}}
        if name not in self.operators:
            raise KeyError(f"Operator '{name}' is not registered")
        return OperatorSpec(name=name, params=params, suffix=str(suffix))

    def apply(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Apply configured operators to provided factor DataFrame.

        Parameters
        ----------
        factors : pd.DataFrame
            Base factor dataframe.

        Returns
        -------
        pd.DataFrame
            Enhanced DataFrame containing originals plus derived features.
        """
        if factors.empty:
            self.logger.warning("No factors provided for enhancement")
            return factors

        enhanced = factors.copy()

        for factor_name in factors.columns:
            series = factors[factor_name]
            specs = self._specs_for_factor(factor_name)
            if not specs:
                continue

            for spec in specs:
                operator = self.operators[spec.name]
                transformed = operator(series, spec.params)
                new_col = f"{factor_name}_{spec.suffix}"
                enhanced[new_col] = transformed
                self.logger.debug(
                    "Applied operator %s to factor %s -> %s",
                    spec.name,
                    factor_name,
                    new_col,
                )

        enhanced = enhanced.replace([np.inf, -np.inf], np.nan)
        return enhanced

    def _specs_for_factor(self, factor_name: str) -> Iterable[OperatorSpec]:
        if not self.operator_scope:
            return self.operator_specs

        names = self.operator_scope.get(factor_name)
        if names is None:
            names = self.operator_scope.get("__all__", set())

        if not names:
            return []

        return [spec for spec in self.operator_specs if spec.name in names]
