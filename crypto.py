import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crypto Volatility Visualizer", page_icon="⚡", layout="wide")
st.title("⚡ Crypto Volatility Visualizer – Fast Public Edition")

# =========================
# SIDEBAR SETTINGS
# =========================
crypto_options = ["BTC-USD", "ETH-USD", "SOL-USD"]
symbol = st.sidebar.selectbox("Select Crypto", crypto_options, index=0)

# Safety check for date input (prevents crash if user only clicks one date)
date_range = st.sidebar.date_input("Date Range", [pd.to_datetime("2020-01-01"), datetime.today().date()])
if len(date_range) != 2:
    st.sidebar.warning("Please select both a start and end date.")
    st.stop()

vol_window = st.sidebar.slider("Volatility Window", 5, 50, 20)
sim_toggle = st.sidebar.checkbox("Enable Simulation Mode")
sim_mode = st.sidebar.selectbox("Simulation Pattern", ["Sine wave", "Random noise", "Drift"])
amp = st.sidebar.slider("Amplitude", 1000, 20000, 5000)
freq = st.sidebar.slider("Frequency", 0.5, 20.0, 5.0)
drift = st.sidebar.slider("Drift slope", -100.0, 100.0, 10.0)
noise = st.sidebar.slider("Noise intensity", 500, 10000, 2000)

# =========================
# DATA LOADING (CACHED)
# =========================
@st.cache_data(ttl=86400)  # cache 1 day
def load_large_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    
    # Safely flatten MultiIndex from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "Price"}, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

df = load_large_data(symbol, date_range[0], date_range[1])

# =========================
# CALCULATE INDICATORS
# =========================
df["Daily_Return"] = df["Price"].pct_change()
df["Rolling_Mean"] = df["Price"].rolling(vol_window).mean()
df["Rolling_Std"] = df["Daily_Return"].rolling(vol_window).std()
df["Rolling_Volatility"] = df["Rolling_Std"] * np.sqrt(252)

# Bollinger Bands
rolling_std = df["Price"].rolling(vol_window).std()
df["BB_Upper"] = df["Rolling_Mean"] + (rolling_std * 2)
df["BB_Lower"] = df["Rolling_Mean"] - (rolling_std * 2)

# RSI
delta = df["Price"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# LIGHTWEIGHT AI VOLATILITY
# =========================
latest_vol = df["Rolling_Volatility"].iloc[-1]
if latest_vol < 0.4:
    vol_state = "Low 🟢"
elif latest_vol < 0.7:
    vol_state = "Medium 🟡"
else:
    vol_state = "High 🔴"

st.subheader(f"Current Volatility: {vol_state} ({latest_vol*100:.1f}%)")

# =========================
# METRICS
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
c2.metric("Daily Return", f"{df['Daily_Return'].iloc[-1]*100:.2f}%")
c3.metric("Annualized Volatility", f"{latest_vol*100:.2f}%")

# =========================
# SIMULATION
# =========================
if sim_toggle:
    st.subheader("Mathematical Price Simulation")
    t = np.arange(len(df))
    base = df["Price"].iloc[0].item() if hasattr(df["Price"].iloc[0], 'item') else float(df["Price"].iloc[0])
    
    if sim_mode == "Sine wave":
        df["Simulated"] = base + amp * np.sin(2 * np.pi * freq * t / len(t))
    elif sim_mode == "Random noise":
        df["Simulated"] = base + np.random.normal(0, noise, len(t))
    else:
        df["Simulated"] = base + drift * t
        
    # Use Date as index so the X-axis shows dates
    st.line_chart(df.set_index("Date")[["Price", "Simulated"]])

# =========================
# VISUALIZATIONS
# =========================
st.subheader("Price & Rolling Volatility")
# Dual-axis Altair chart to handle different scales
base = alt.Chart(df).encode(x="Date:T")
price_line = base.mark_line(color="#00ffcc").encode(y=alt.Y("Price:Q", title="Price ($)"))
vol_line = base.mark_line(color="orange").encode(y=alt.Y("Rolling_Volatility:Q", title="Volatility"))
dual_chart = alt.layer(price_line, vol_line).resolve_scale(y='independent')
st.altair_chart(dual_chart, use_container_width=True)

st.subheader("Daily Return Distribution")
st.bar_chart(df.set_index("Date")["Daily_Return"].dropna())

st.subheader("Bollinger Bands")
st.line_chart(df.set_index("Date")[["Price", "BB_Upper", "BB_Lower"]])

st.subheader("RSI (14-day)")
st.line_chart(df.set_index("Date")["RSI"])

# =========================
# MONTE CARLO SIMULATION (LIGHT)
# =========================
st.subheader("Monte Carlo Simulation (10 paths, 30 days)")
if st.button("Run Fast Monte Carlo Simulation"):
    returns = df["Daily_Return"].dropna()
    mean_return, std_return = returns.mean(), returns.std()
    last_price = float(df["Price"].iloc[-1])
    
    simulations = []
    for _ in range(10):  
        path = [last_price]
        for _ in range(30):
            path.append(path[-1] * (1 + np.random.normal(mean_return, std_return)))
        simulations.append(path)
    
    sim_df = pd.DataFrame(simulations).T
    # Set dates as the index for the X-axis
    sim_df.index = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(31)]
    st.line_chart(sim_df)

# =========================
# SIMPLE TREND FORECAST (7 DAYS)
# =========================
st.subheader("7-Day Trend Forecast")
recent_data = df.iloc[-30:].copy()
recent_data["Day_Num"] = np.arange(len(recent_data))
slope, intercept = np.polyfit(recent_data["Day_Num"], recent_data["Price"], 1)

future_days = np.arange(len(recent_data), len(recent_data)+7)
predicted_prices = slope * future_days + intercept
future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]

# Format data so the chart shows two distinct lines (Actual vs Forecast)
plot_df = pd.DataFrame(index=pd.concat([recent_data["Date"], pd.Series(future_dates)]))
plot_df["Actual Price"] = recent_data.set_index("Date")["Price"]
plot_df["Trend Forecast"] = pd.Series(predicted_prices, index=future_dates)

st.line_chart(plot_df)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>FinTechLab Pvt. Ltd. | Optimized FA-2 Project</div>", unsafe_allow_html=True)
