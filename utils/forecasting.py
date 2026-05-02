"""# 8. Time Series Forecasting (SARIMA)

SARIMA forecasts future dangerous cases from daily risk counts. Forecasting is
important because rising interaction risk should trigger monitoring before harm
accumulates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_daily_risk_series(enriched_df: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Create daily dangerous-case counts when source data has no timestamps."""
    df = enriched_df.copy()
    df["pseudo_label"] = labels.values
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=180, freq="D")
    df["event_date"] = [dates[i % len(dates)] for i in range(len(df))]
    dangerous = df[df["pseudo_label"] == "High Risk"].groupby("event_date").size()
    series = dangerous.reindex(dates, fill_value=0).astype(float)
    # A tiny rolling signal avoids all-zero training in sparse samples.
    return series.rolling(3, min_periods=1).mean()


def train_sarima_forecast(series: pd.Series, horizon: int = 14) -> tuple[pd.Series, pd.Series, float]:
    """Train SARIMA and forecast future dangerous cases."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        train = series.iloc[:-horizon]
        test = series.iloc[-horizon:]
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        predicted = fitted.forecast(steps=horizon)
        future = fitted.forecast(steps=horizon * 2).iloc[-horizon:]
        error = float(np.mean(np.abs(test.values - predicted.values)))
        combined_forecast = pd.concat([predicted, future])
    except Exception:
        predicted = series.rolling(7, min_periods=1).mean().iloc[-horizon:]
        future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        future = pd.Series(float(predicted.mean()), index=future_index)
        error = float(np.mean(np.abs(series.iloc[-horizon:].values - predicted.values)))
        combined_forecast = pd.concat([predicted, future])
    return combined_forecast, series.iloc[-horizon:], error


def forecast_is_increasing(forecast: pd.Series, actual_recent: pd.Series) -> bool:
    """Determine whether future dangerous cases are trending up."""
    future_half = forecast.iloc[len(forecast) // 2 :]
    return bool(future_half.mean() > actual_recent.mean())


def forecast_plot(series: pd.Series, forecast: pd.Series):
    """Plot actual vs predicted dangerous risk counts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Predicted"))
    fig.update_layout(
        template="plotly_dark",
        title="Actual vs Predicted Dangerous Cases",
        xaxis_title="Date",
        yaxis_title="Daily dangerous risk count",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

