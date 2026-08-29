"""Simple interpretable forecasting utilities for trade data."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

SUPPORTED_METRICS = (
    "imports",
    "exports",
    "trade_total",
    "import_dependency_ratio",
    "export_dependency_ratio",
)
SUPPORTED_MODELS = ("auto", "linear", "arima", "hybrid")


def build_country_time_series(data: pd.DataFrame, country: str) -> pd.DataFrame:
    """Aggregate yearly trade metrics for a single country."""
    relevant = data[(data["exporter"] == country) | (data["importer"] == country)].copy()
    if relevant.empty:
        raise ValueError(f"No trade history is available for country '{country}'.")

    records: List[Dict[str, float]] = []
    for year, year_frame in relevant.groupby("year"):
        total_imports = float(year_frame.loc[year_frame["importer"] == country, "trade_value"].sum())
        total_exports = float(year_frame.loc[year_frame["exporter"] == country, "trade_value"].sum())
        trade_total = total_imports + total_exports

        records.append(
            {
                "year": int(year),
                "imports": total_imports,
                "exports": total_exports,
                "trade_total": trade_total,
                "import_dependency_ratio": total_imports / trade_total if trade_total else 0.0,
                "export_dependency_ratio": total_exports / trade_total if trade_total else 0.0,
            }
        )

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def forecast_country_metric(
    data: pd.DataFrame,
    country: str,
    metric: str = "trade_total",
    periods: int = 1,
    model: str = "auto",
) -> Dict[str, object]:
    """Forecast a country metric from a normalized trade dataframe."""
    time_series = build_country_time_series(data, country)
    return forecast_metric_from_series(
        time_series=time_series,
        country=country,
        metric=metric,
        periods=periods,
        model=model,
    )


def forecast_metric_from_series(
    time_series: pd.DataFrame,
    country: str,
    metric: str = "trade_total",
    periods: int = 1,
    model: str = "auto",
) -> Dict[str, object]:
    """Forecast a metric from a prebuilt yearly time series."""
    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(SUPPORTED_METRICS)
        raise ValueError(f"Unsupported forecast metric '{metric}'. Supported metrics: {supported}")

    selected_model = str(model).strip().lower()
    if selected_model not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported forecast model '{model}'. Supported models: {supported}")

    if "year" not in time_series.columns or metric not in time_series.columns:
        raise ValueError("Time series must contain a 'year' column and the requested metric column.")

    ordered = time_series.sort_values("year").reset_index(drop=True)
    years = ordered["year"].to_numpy(dtype=float)
    values = ordered[metric].to_numpy(dtype=float)
    forecast_periods = max(1, int(periods))

    if len(ordered) < 2:
        predictions = np.repeat(float(values[-1]), forecast_periods)
        future_years = np.arange(int(years[-1]) + 1, int(years[-1]) + forecast_periods + 1, dtype=int)
        prediction_records = [
            {"year": int(year), "predicted_value": _bound_metric(metric, float(predicted))}
            for year, predicted in zip(future_years, predictions)
        ]
        return {
            "country": country,
            "metric": metric,
            "method": "naive",
            "slope": 0.0,
            "intercept": float(values[-1]),
            "history": ordered.to_dict(orient="records"),
            "forecast": prediction_records,
            "confidence": 0.3,
            "model_confidence": 0.3,
            "model_scores": {},
        }

    model_scores: Dict[str, float] = {}
    if selected_model == "auto":
        selected_model, model_scores = _select_best_model(values)

    if selected_model == "linear":
        predictions, slope, intercept = _linear_forecast(values=values, periods=forecast_periods)
        method = "linear_regression"
        confidence = _estimate_linear_confidence(values=values, slope=slope, intercept=intercept)
    elif selected_model == "arima":
        predictions, slope, intercept = _arima_direct_forecast(values=values, periods=forecast_periods)
        method = "arima(1,1,1)"
        confidence = _estimate_from_rolling_mape(_rolling_mape(values, _arima_direct_forecast_only))
    else:
        predictions, slope, intercept = _hybrid_forecast(values=values, periods=forecast_periods)
        method = "linear_plus_arima_residual"
        confidence = _estimate_from_rolling_mape(_rolling_mape(values, _hybrid_forecast_only))

    future_years = np.arange(int(years.max()) + 1, int(years.max()) + forecast_periods + 1, dtype=int)
    prediction_records = [
        {"year": int(year), "predicted_value": _bound_metric(metric, float(predicted))}
        for year, predicted in zip(future_years, predictions)
    ]

    return {
        "country": country,
        "metric": metric,
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "history": ordered.to_dict(orient="records"),
        "forecast": prediction_records,
        "confidence": float(confidence),
        "model_confidence": float(confidence),
        "model_scores": model_scores,
    }


def _linear_forecast(values: np.ndarray, periods: int) -> Tuple[np.ndarray, float, float]:
    """Forecast using a linear trend fit."""
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    xf = np.arange(len(values), len(values) + periods, dtype=float)
    return slope * xf + intercept, float(slope), float(intercept)


def _fit_arima(values: np.ndarray, order: Tuple[int, int, int], periods: int) -> Optional[np.ndarray]:
    """Fit an ARIMA model and forecast, or return None if it cannot.

    Short, flat, and zero-heavy trade series make maximum likelihood fail to
    converge often -- roughly one series in twenty here. That is a handled
    condition: callers fall back to the linear trend, and rolling-MAPE model
    selection scores the fit quality anyway. Left unsuppressed the warning is
    written to stderr on every such request, which fills the API log with
    noise an operator cannot act on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        try:
            fit = ARIMA(
                values,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            return np.asarray(fit.forecast(steps=periods), dtype=float)
        except Exception:
            return None


def _arima_direct_forecast(values: np.ndarray, periods: int) -> Tuple[np.ndarray, float, float]:
    """Forecast directly with a simple ARIMA model."""
    if len(values) < 8:
        return _linear_forecast(values=values, periods=periods)

    pred = _fit_arima(values, order=(1, 1, 1), periods=periods)
    if pred is None:
        pred, _, _ = _linear_forecast(values=values, periods=periods)

    slope, intercept = np.polyfit(np.arange(len(values), dtype=float), values, 1)
    return pred, float(slope), float(intercept)


def _hybrid_forecast(values: np.ndarray, periods: int) -> Tuple[np.ndarray, float, float]:
    """Forecast with linear trend plus ARIMA residual correction."""
    trend_pred, slope, intercept = _linear_forecast(values=values, periods=periods)
    x = np.arange(len(values), dtype=float)
    trend_hist = slope * x + intercept
    residuals = values - trend_hist

    if len(values) < 12:
        return trend_pred, slope, intercept

    residual_pred = _fit_arima(residuals, order=(1, 0, 1), periods=periods)
    if residual_pred is None:
        # No usable residual model means no correction, which leaves the plain
        # linear trend rather than a worse one.
        residual_pred = np.zeros(periods, dtype=float)

    return trend_pred + residual_pred, slope, intercept


def _arima_direct_forecast_only(values: np.ndarray, periods: int) -> np.ndarray:
    """Helper for rolling validation."""
    return _arima_direct_forecast(values=values, periods=periods)[0]


def _hybrid_forecast_only(values: np.ndarray, periods: int) -> np.ndarray:
    """Helper for rolling validation."""
    return _hybrid_forecast(values=values, periods=periods)[0]


def _linear_forecast_only(values: np.ndarray, periods: int) -> np.ndarray:
    """Helper for rolling validation."""
    return _linear_forecast(values=values, periods=periods)[0]


def _rolling_mape(
    values: np.ndarray,
    predictor: Callable[[np.ndarray, int], np.ndarray],
    min_train: int = 10,
    window_points: int = 8,
) -> float:
    """Estimate one-step rolling MAPE for model selection."""
    if len(values) <= min_train:
        return float("inf")

    start_index = max(min_train, len(values) - window_points)
    errors: List[float] = []

    for index in range(start_index, len(values)):
        train = values[:index]
        actual = float(values[index])
        try:
            forecast = float(predictor(train, 1)[0])
        except Exception:
            continue
        denominator = max(abs(actual), 1e-9)
        errors.append(abs(forecast - actual) / denominator)

    if not errors:
        return float("inf")
    return float(np.mean(errors))


def _select_best_model(values: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Select the best small-data model using rolling MAPE."""
    model_scores: Dict[str, float] = {}
    model_scores["linear"] = _rolling_mape(values, _linear_forecast_only)
    model_scores["arima"] = _rolling_mape(values, _arima_direct_forecast_only)
    if len(values) >= 14:
        model_scores["hybrid"] = _rolling_mape(values, _hybrid_forecast_only)

    finite_scores = {name: score for name, score in model_scores.items() if np.isfinite(score)}
    if not finite_scores:
        return "linear", model_scores
    selected = min(finite_scores, key=finite_scores.get)
    return selected, model_scores


def _estimate_linear_confidence(values: np.ndarray, slope: float, intercept: float) -> float:
    """Estimate a simple confidence score from fit quality and history length."""
    x = np.arange(len(values), dtype=float)
    fitted_values = (slope * x) + intercept
    total_variance = float(np.sum((values - values.mean()) ** 2))
    residual_variance = float(np.sum((values - fitted_values) ** 2))
    r_squared = 1.0 if total_variance == 0 else max(0.0, 1.0 - (residual_variance / total_variance))
    coverage = min(1.0, len(values) / 8.0)
    return float(np.clip(0.25 + (0.5 * r_squared) + (0.25 * coverage), 0.1, 0.99))


def _estimate_from_rolling_mape(mape: float) -> float:
    """Convert rolling MAPE into a bounded confidence score."""
    if not np.isfinite(mape):
        return 0.35
    # Practical mapping: 0-5% => high confidence, 20%+ => lower confidence.
    return float(np.clip(1.0 - (mape / 25.0), 0.2, 0.95))


def _bound_metric(metric: str, value: float) -> float:
    """Bound forecast values to plausible ranges for the requested metric."""
    if "dependency_ratio" in metric:
        return float(max(0.0, min(1.0, value)))
    return float(max(0.0, value))
