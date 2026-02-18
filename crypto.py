import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt
import google.generativeai as genai

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crypto Volatility Visualizer", page_icon="⚡", layout="wide")
st.title("⚡ Crypto Volatility Visualizer – AI Public Edition")

# =========================
# SIDEBAR SETTINGS & SECRETS
# =========================
with st.sidebar:
    st.header("🔑 AI Integration")
    
    # Securely fetch the API key from secrets.toml
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    
    if gemini_api_key:
        st.success("✅ API Key loaded securely!")
    else:
        st.error("⚠️ API Key not found in secrets.toml")
        
    st.header("⚙️ Settings")
    crypto_options = ["BTC-USD", "ETH-USD", "SOL-USD"]
    symbol = st.selectbox("Select Crypto", crypto_options, index=0)

    # Date Range with safety check
    date_range = st.date_input("Date Range", [pd.to_datetime("2020-01-01"), datetime.today().date()])
    if len(date_range) != 2:
        st.warning("Please select both a start and end date.")
        st.stop()

    vol_window = st.slider("Volatility Window", 5, 50, 20)
    
    st.markdown("---")
    st.subheader("📐 Math Simulation")
    sim_toggle = st.checkbox("Enable Simulation Mode")
    sim_mode = st.selectbox("Simulation Pattern", ["Sine wave", "Random noise", "Drift"])
    amp = st.slider("Amplitude", 1000, 20000, 5000)
    freq = st.slider("Frequency", 0.5, 20.0, 5.0)
    drift = st.slider("Drift slope", -100.0, 100.0, 10.0)
    noise = st.slider("Noise intensity", 500, 10000, 2000)

# =========================
# DATA LOADING (CACHED)
# =========================
@st.cache_data(ttl=86400)  
def load_large_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
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

rolling_std = df["Price"].rolling(vol_window).std()
df["BB_Upper"] = df["Rolling_Mean"] + (rolling_std * 2)
df["BB_Lower"] = df["Rolling_Mean"] - (rolling_std * 2)

delta = df["Price"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

latest_vol = float(df["Rolling_Volatility"].iloc[-1])
latest_price = float(df['Price'].iloc[-1])
latest_ret = float(df['Daily_Return'].iloc[-1])

# =========================
# METRICS DASHBOARD
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Latest Price", f"${latest_price:,.2f}")
c2.metric("Daily Return", f"{latest_ret*100:.2f}%")
c3.metric("Annualized Volatility", f"{latest_vol*100:.2f}%")

st.markdown("---")

# =========================
# VISUALIZATIONS & TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Core Charts", "📐 Math Simulation", "🔮 Forecasts", "🤖 Gemini AI Assistant"])

with tab1:
    st.subheader("Price & Rolling Volatility")
    base = alt.Chart(df).encode(x="Date:T")
    price_line = base.mark_line(color="#00ffcc").encode(y=alt.Y("Price:Q", title="Price ($)"))
    vol_line = base.mark_line(color="orange").encode(y=alt.Y("Rolling_Volatility:Q", title="Volatility"))
    dual_chart = alt.layer(price_line, vol_line).resolve_scale(y='independent')
    st.altair_chart(dual_chart, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Bollinger Bands")
        st.line_chart(df.set_index("Date")[["Price", "BB_Upper", "BB_Lower"]])
    with c_right:
        st.subheader("RSI (14-day)")
        st.line_chart(df.set_index("Date")["RSI"])

with tab2:
    if sim_toggle:
        st.subheader("Mathematical Price Simulation")
        t = np.arange(len(df))
        base_val = df["Price"].iloc[0].item() if hasattr(df["Price"].iloc[0], 'item') else float(df["Price"].iloc[0])
        
        if sim_mode == "Sine wave":
            df["Simulated"] = base_val + amp * np.sin(2 * np.pi * freq * t / len(t))
        elif sim_mode == "Random noise":
            df["Simulated"] = base_val + np.random.normal(0, noise, len(t))
        else:
            df["Simulated"] = base_val + drift * t
            
        st.line_chart(df.set_index("Date")[["Price", "Simulated"]])
    else:
        st.info("👈 Enable Simulation Mode in the sidebar to view mathematical overlays.")

with tab3:
    st.subheader("Monte Carlo Simulation (10 paths, 30 days)")
    if st.button("Run Fast Monte Carlo"):
        returns = df["Daily_Return"].dropna()
        mean_return, std_return = returns.mean(), returns.std()
        
        simulations = []
        for _ in range(10):  
            path = [latest_price]
            for _ in range(30):
                path.append(path[-1] * (1 + np.random.normal(mean_return, std_return)))
            simulations.append(path)
        
        sim_df = pd.DataFrame(simulations).T
        sim_df.index = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(31)]
        st.line_chart(sim_df)

    st.subheader("7-Day Trend Forecast")
    recent_data = df.iloc[-30:].copy()
    recent_data["Day_Num"] = np.arange(len(recent_data))
    slope, intercept = np.polyfit(recent_data["Day_Num"], recent_data["Price"], 1)

    future_days = np.arange(len(recent_data), len(recent_data)+7)
    predicted_prices = slope * future_days + intercept
    future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]

    plot_df = pd.DataFrame(index=pd.concat([recent_data["Date"], pd.Series(future_dates)]))
    plot_df["Actual Price"] = recent_data.set_index("Date")["Price"]
    plot_df["Trend Forecast"] = pd.Series(predicted_prices, index=future_dates)
    st.line_chart(plot_df)

# =========================
# 🤖 GEMINI AI ASSISTANT
# =========================
with tab4:
    st.subheader("🧠 Senior AI Quant Assistant")
    st.write("Ask the AI about the current market, volatility formulas, or investment risks.")

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("E.g., Based on the current volatility, is it a good time to buy?"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                context = f"""
                You are a Senior Quantitative Analyst AI. Answer concisely.
                CURRENT MARKET CONTEXT:
                - Asset: {symbol}
                - Latest Price: ${latest_price:,.2f}
                - Daily Return: {latest_ret*100:.2f}%
                - Annualized Volatility: {latest_vol*100:.2f}%
                """
                full_prompt = f"{context}\n\nUser Question: {prompt}"

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing market data..."):
                        response = model.generate_content(full_prompt)
                        st.markdown(response.text)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"API Error: {e}. Please check your API key configuration.")
    else:
        st.warning("⚠️ The AI Assistant is disabled. Please configure your API Key in Streamlit secrets.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>FinTechLab Pvt. Ltd. | Optimized FA-2 Project</div>", unsafe_allow_html=True)
