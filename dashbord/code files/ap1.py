# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:58:52 2026

@author: adiya
"""


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")
st.caption("Daily Revenue Prediction using Random Forest Regression")

# -------------------------------
# LOAD MODEL AND DATA
# -------------------------------

@st.cache_resource
def load_model():
    return joblib.load("final_rf.pkl")


@st.cache_data
def load_data():
    data = pd.read_csv("daily_sales.xls")

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date")

    history = (
        data.groupby("Date")["Sales"]
        .sum()
        .sort_index()
        .asfreq("D", fill_value=0)
    )

    return history


try:
    model = load_model()
    sales_history = load_data()

except Exception as e:
    st.error(f"Unable to load model or dataset: {e}")
    st.stop()


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.header("⚙️ Forecast Settings")

forecast_days = st.sidebar.selectbox(
    "Select Forecast Period",
    [30, 60, 90],
    index=1
)

st.sidebar.write("Forecast period:", forecast_days, "days")


# -------------------------------
# FORECASTING FUNCTION
# -------------------------------

def generate_forecast(model, history, days):

    history = history.copy()

    future_dates = pd.date_range(
        start=history.index.max() + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    results = []

    for date in future_dates:

        features = {
            "Day": date.day,
            "Month": date.month,
            "Year": date.year,
            "DayOfWeek": date.dayofweek,
            "WeekOfYear": date.isocalendar().week,

            "Lag_1": history.iloc[-1],
            "Lag_7": history.iloc[-7],
            "Lag_14": history.iloc[-14],
            "Lag_28": history.iloc[-28],

            "MA_7": history.tail(7).mean(),
            "MA_14": history.tail(14).mean(),
            "MA_30": history.tail(30).mean()
        }

        input_data = pd.DataFrame([features])

        predicted_sales = model.predict(input_data)[0]

        predicted_sales = max(0, predicted_sales)

        results.append({
            "Date": date,
            "Predicted_Sales": predicted_sales
        })

        history.loc[date] = predicted_sales

    return pd.DataFrame(results)


# -------------------------------
# GENERATE FORECAST
# -------------------------------

if len(sales_history) < 30:
    st.error("At least 30 days of historical sales data are required.")
    st.stop()

forecast_df = generate_forecast(
    model,
    sales_history,
    forecast_days
)


# -------------------------------
# KPI SECTION
# -------------------------------

total_revenue = sales_history.sum()

forecast_revenue = forecast_df["Predicted_Sales"].sum()

average_sales = forecast_df["Predicted_Sales"].mean()

max_sales = forecast_df["Predicted_Sales"].max()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Historical Revenue",
    f"₹{total_revenue:,.2f}"
)

col2.metric(
    "Forecast Revenue",
    f"₹{forecast_revenue:,.2f}"
)

col3.metric(
    "Average Daily Forecast",
    f"₹{average_sales:,.2f}"
)

col4.metric(
    "Maximum Predicted Sales",
    f"₹{max_sales:,.2f}"
)


# -------------------------------
# FORECAST INFORMATION
# -------------------------------

st.divider()

st.subheader("📅 Forecast Information")

col1, col2, col3 = st.columns(3)

col1.write("**Last Historical Date**")
col1.write(sales_history.index.max().strftime("%d %b %Y"))

col2.write("**Forecast Start Date**")
col2.write(forecast_df["Date"].min().strftime("%d %b %Y"))

col3.write("**Forecast End Date**")
col3.write(forecast_df["Date"].max().strftime("%d %b %Y"))


# -------------------------------
# HISTORICAL VS FORECAST CHART
# -------------------------------

st.divider()

st.subheader("📊 Historical Sales vs Future Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=sales_history.tail(90).index,
        y=sales_history.tail(90).values,
        mode="lines",
        name="Historical Sales",
        line=dict(width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Predicted_Sales"],
        mode="lines",
        name="Forecasted Sales",
        line=dict(width=2, dash="dash")
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales Revenue (₹)",
    hovermode="x unified",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# FORECAST TABLE
# -------------------------------

st.divider()

st.subheader("📋 Forecast Details")

display_df = forecast_df.copy()

display_df["Date"] = display_df["Date"].dt.strftime("%d-%m-%Y")

display_df["Predicted_Sales"] = display_df["Predicted_Sales"].round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# -------------------------------
# DAILY FORECAST CHART
# -------------------------------

st.subheader("📈 Daily Predicted Revenue")

bar_fig = go.Figure()

bar_fig.add_trace(
    go.Bar(
        x=forecast_df["Date"],
        y=forecast_df["Predicted_Sales"],
        name="Predicted Revenue"
    )
)

bar_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Predicted Revenue (₹)",
    template="plotly_white",
    height=400
)

st.plotly_chart(bar_fig, use_container_width=True)


# -------------------------------
# DOWNLOAD FORECAST
# -------------------------------

st.divider()

st.subheader("📥 Download Forecast")

csv_data = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

excel_data = None

import io

buffer = io.BytesIO()

with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    forecast_df.to_excel(
        writer,
        index=False,
        sheet_name="Sales Forecast"
    )

st.download_button(
    label="Download Forecast Excel",
    data=buffer.getvalue(),
    file_name="sales_forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success("Sales forecasting completed successfully!")

st.caption(
    "Note: Forecasted values are model estimates, not actual future sales."
)
