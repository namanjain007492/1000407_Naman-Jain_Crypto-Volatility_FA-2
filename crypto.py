import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt
import openai  # Gemini API works like OpenAI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crypto AI Assistant & Volatility Visualizer", page_icon="₿", layout="wide")
st.title("⚡ Crypto AI Assistant & Volatility Visualizer")

# =========================
# GEMINI / AI API KEY
# =========================
api_key = st.secrets.get("GEMINI_API_KEY")  # Use Streamlit secrets for safety
if not api_key:
    st.warning("⚠️ Please set your Gemini API Key in Streamlit Secrets.")
openai.api_key = api_key

# =========================
# SIDEBAR SETTINGS
# =========================
crypto_options = ["BTC-USD", "ETH-USD", "SOL-USD"]
symbol = st.sidebar.selectbox("Select Crypto", crypto_options, index=0)
date_range = st.sidebar.date_input("Date Range", [pd.to_datetime("2020-01-01"), datetime.today()])
vol_window = st.sidebar.slider("Volatility Window", 5, 50, 20)
sim_toggle = st.sidebar.checkbox("Enable Simulation Mode")
sim_mode = st.sidebar.selectbox("Simulation Pattern", ["Sine wave", "Random noise", "Drift", "Combined mode"])
amp = st.sidebar.slider("Amplitude", 1000, 20000, 5000)
freq = st.sidebar.slider("Frequency", 0.5, 20.0, 5.0)
drift = st.sidebar.slider("Drift slope", -100.0, 100.0, 10.0)
noise = st.sidebar.slider("Noise intensity", 500, 10000, 2000)

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=86400)
def load_large_data(symbol="BTC-USD", start="2020-01-01", end=datetime.today().strftime("%Y-%m-%d")):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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

price_series = df["Price"].squeeze()
rolling_std = price_series.rolling(vol_window).std()
df["BB_Upper"] = df["Rolling_Mean"] + (rolling_std * 2)
df["BB_Lower"] = df["Rolling_Mean"] - (rolling_std * 2)

delta = df["Price"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# =========================
# AI Recommendations
# =========================
def ai_recommendation(df):
    latest_vol = df["Rolling_Volatility"].iloc[-1]
    latest_rsi = df["RSI"].iloc[-1]
    recs = []
    if latest_vol < 0.4:
        recs.append("Market stable 🟢 – good for accumulation.")
    elif latest_vol < 0.7:
        recs.append("Normal market fluctuations 🟡 – trade cautiously.")
    else:
        recs.append("High volatility 🔴 – avoid large positions.")
    if latest_rsi > 70:
        recs.append(f"RSI {latest_rsi:.1f} → Overbought, possible pullback.")
    elif latest_rsi < 30:
        recs.append(f"RSI {latest_rsi:.1f} → Oversold, possible bounce.")
    return recs

st.subheader("🤖 AI Recommendations")
recs = ai_recommendation(df)
for r in recs:
    st.info(r)

# =========================
# METRICS
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
c2.metric("Daily Return", f"{df['Daily_Return'].iloc[-1]*100:.2f}%")
c3.metric("Rolling Volatility", f"{df['Rolling_Volatility'].iloc[-1]*100:.2f}%")

# =========================
# SIMULATION
# =========================
if sim_toggle:
    t = np.arange(len(df))
    base = df["Price"].iloc[0]
    if sim_mode == "Sine wave":
        df["Simulated"] = base + amp * np.sin(2 * np.pi * freq * t / len(t))
    elif sim_mode == "Random noise":
        df["Simulated"] = base + np.random.normal(0, noise, len(t))
    elif sim_mode == "Drift":
        df["Simulated"] = base + drift * t
    else:
        df["Simulated"] = base + drift*t + amp*np.sin(2*np.pi*freq*t/len(t)) + np.random.normal(0, noise, len(t))
    st.subheader(f"Simulation Mode: {sim_mode}")
    st.line_chart(df[["Price", "Simulated"]])

# =========================
# DOWNSAMPLE FOR PLOTTING
# =========================
def downsample(df, factor=10):
    return df.iloc[::factor, :].copy()

df_plot = downsample(df, factor=10)

# =========================
# VISUALIZATIONS
# =========================
st.subheader("Price & Rolling Volatility")
price_chart = alt.Chart(df_plot).mark_line().encode(
    x="Date", y="Price", tooltip=["Date", "Price"]
)
vol_chart = alt.Chart(df_plot).mark_line(color="orange").encode(
    x="Date", y="Rolling_Volatility"
)
st.altair_chart(price_chart + vol_chart, use_container_width=True)

st.subheader("Daily Return Distribution")
st.bar_chart(df_plot["Daily_Return"].dropna())

st.subheader("Bollinger Bands")
st.line_chart(df_plot[["Price", "BB_Upper", "BB_Lower"]])

st.subheader("RSI (14-day)")
st.line_chart(df_plot["RSI"])

# =========================
# MONTE CARLO SIMULATION
# =========================
st.subheader("Monte Carlo Simulation (10 paths, 30 days)")
if st.button("Run Monte Carlo Simulation"):
    returns = df["Daily_Return"].dropna()
    mean_return, std_return = returns.mean(), returns.std()
    last_price = df["Price"].iloc[-1]
    sims = []
    for _ in range(10):
        path = [last_price]
        for _ in range(30):
            path.append(path[-1] * (1 + np.random.normal(mean_return, std_return)))
        sims.append(path)
    sim_df = pd.DataFrame(sims).T
    sim_df.index = [datetime.today() + timedelta(days=i) for i in range(31)]
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
future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1,8)]
forecast_df = pd.DataFrame({"Date": future_dates, "Price": predicted_prices})
st.line_chart(pd.concat([recent_data[["Date","Price"]], forecast_df], ignore_index=True))

# =========================
# GEMINI AI QUESTION ANSWERING
# =========================
st.subheader("💬 Ask Crypto Questions (AI Powered)")
sample_questions = [
    "Should I buy BTC today?",
    "What is the current volatility of ETH?",
    "Is SOL overbought or oversold?",
    "Explain Bitcoin market trend in simple words."
]
st.markdown("**Sample Questions:** " + ", ".join(sample_questions))
user_question = st.text_input("Ask a question about your selected crypto:")

def ask_gemini(question, symbol):
    prompt = f"You are a crypto AI assistant. Answer this question for {symbol}: {question}"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

if user_question:
    answer = ask_gemini(user_question, symbol)
    st.info(answer)

st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>FinTechLab Pvt. Ltd. | Crypto AI FA-2 Project</div>", unsafe_allow_html=True)
