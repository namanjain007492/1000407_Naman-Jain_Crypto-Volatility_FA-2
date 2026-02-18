import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import base64
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Bitcoin Volatility Visualizer", page_icon="⚡", layout="wide")

# CSS for UI
st.markdown("""
    <style>
    .metric-card { background-color: #1e1e24; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #333; }
    .metric-value { font-size: 26px; font-weight: bold; color: #00ffcc; margin-top: 5px; }
    .metric-label { font-size: 14px; color: #b0b0b0; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 FAST DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_fast_data(symbol="BTC-USD", period="1y"):
    """Fetches data efficiently with caching."""
    df = yf.download(symbol, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    
    # Fast cleaning
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={"Close": "Price"}, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Price", "Volume"]]
    df.ffill(inplace=True)
    return df

@st.cache_data
def calculate_metrics(df):
    """Vectorized calculation of indicators (Instant)."""
    # Price Changes
    df["Daily_Return"] = df["Price"].pct_change()
    
    # Vectorized Rolling Window Operations
    indexer = df["Price"].rolling(window=20)
    df["Rolling_Mean"] = indexer.mean()
    df["Rolling_Std"] = df["Daily_Return"].rolling(window=20).std()
    df["Rolling_Volatility"] = df["Rolling_Std"] * np.sqrt(252)
    
    # Bollinger Bands
    df["BB_Upper"] = df["Rolling_Mean"] + (indexer.std() * 2)
    df["BB_Lower"] = df["Rolling_Mean"] - (indexer.std() * 2)
    
    # RSI (Vectorized)
    delta = df["Price"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Drawdown
    cum_max = df["Price"].cummax()
    df["Drawdown"] = (df["Price"] - cum_max) / cum_max
    
    return df.dropna()

# ==========================================
# 🔹 LIGHTWEIGHT FEATURES
# ==========================================
def render_mascot(volatility):
    color = "#00ff00" if volatility < 0.5 else "#ff0000"
    # Simple HTML/CSS Pulse Animation (Lighter than Three.js)
    html = f"""
    <div style="display:flex; justify-content:center; align-items:center; height:200px;">
        <div style="width:100px; height:100px; background:{color}; border-radius:50%; 
        box-shadow: 0 0 20px {color}; animation: pulse 2s infinite;"></div>
    </div>
    <style>@keyframes pulse {{ 0% {{ transform: scale(0.95); opacity: 0.7; }} 50% {{ transform: scale(1.05); opacity: 1; }} 100% {{ transform: scale(0.95); opacity: 0.7; }} }}</style>
    <div style="text-align:center; color:white;">AI Status: <b>{volatility*100:.1f}% Volatility</b></div>
    """
    components.html(html, height=220)

def fast_simulation(df):
    """Vectorized Monte Carlo (100x Faster than Loops)."""
    last_price = df["Price"].iloc[-1]
    returns = df["Daily_Return"].dropna()
    mean = returns.mean()
    std = returns.std()
    
    # Generate 100 paths of 30 days INSTANTLY using Matrix Math
    # Shape: (30 days, 100 simulations)
    daily_returns = np.random.normal(mean, std, (30, 100))
    price_paths = last_price * (1 + daily_returns).cumprod(axis=0)
    
    fig = go.Figure()
    # Plot only first 50 paths to save rendering memory
    fig.add_trace(go.Scatter(y=price_paths[:, 0], mode='lines', line=dict(color='rgba(0,255,200,0.3)'), name='Simulations'))
    for i in range(1, 50):
        fig.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(color='rgba(0,255,200,0.1)'), showlegend=False))
        
    fig.update_layout(title="⚡ Instant Monte Carlo (Vectorized)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🔹 MAIN APP
# ==========================================
def main():
    st.title("⚡ Bitcoin Volatility Visualizer")
    
    if "symbol" not in st.session_state:
        st.session_state.symbol = "BTC-USD"

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.caption("Lightweight Public Edition")
        
        # Simulated Voice Control (Instant State Change)
        if st.button("🎙️ Voice: 'Switch to ETH'"):
            st.session_state.symbol = "ETH-USD"
            st.rerun()

        # Crypto Selector
        options = ["BTC-USD", "ETH-USD", "SOL-USD"]
        idx = options.index(st.session_state.symbol) if st.session_state.symbol in options else 0
        new_symbol = st.selectbox("Crypto", options, index=idx)
        if new_symbol != st.session_state.symbol:
            st.session_state.symbol = new_symbol
            st.rerun()

        period = st.selectbox("Data Period", ["3mo", "6mo", "1y"], index=1)
        
        st.markdown("---")
        st.subheader("Simulate")
        show_sim = st.checkbox("Show Math Model")
        amp = st.slider("Amplitude", 100, 5000, 2000)

    # Load Data (Cached)
    raw_df = fetch_fast_data(st.session_state.symbol, period)
    df = calculate_metrics(raw_df)

    # Metrics Row
    curr_price = df["Price"].iloc[-1]
    volatility = df["Rolling_Volatility"].iloc[-1]
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">${curr_price:,.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Volatility</div><div class="metric-value">{volatility*100:.1f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Risk</div><div class="metric-value" style="color:{"red" if volatility > 0.6 else "green"}">{"HIGH" if volatility > 0.6 else "LOW"}</div></div>', unsafe_allow_html=True)

    # Main Visuals
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        # Main Chart
        fig = px.line(df, x="Date", y="Price", title=f"{st.session_state.symbol} Price Action")
        
        # Math Simulation Overlay (Instant Calculation)
        if show_sim:
            t = np.arange(len(df))
            # Vectorized Sine Wave
            sim_price = df["Price"].iloc[0] + amp * np.sin(0.1 * t) + (t * 5)
            fig.add_trace(go.Scatter(x=df["Date"], y=sim_price, name="Math Model", line=dict(dash='dot', color='yellow')))
            
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicators Tab
        tab1, tab2 = st.tabs(["📉 Indicators", "🔮 Predictions"])
        with tab1:
            st.plotly_chart(px.line(df, x="Date", y="RSI", title="RSI Indicator"), use_container_width=True)
        with tab2:
            if st.button("Run Fast Monte Carlo"):
                fast_simulation(df)

    with c_right:
        # Lightweight Mascot
        render_mascot(volatility)
        
        # Replay (Optimized)
        st.subheader("Playback")
        if st.button("▶️ Play (Last 30 Days)"):
            chart_spot = st.empty()
            subset = df.iloc[-30:]
            # Reduced frames for speed (skip every 2nd day)
            for i in range(5, 30, 2):
                fig_anim = px.line(subset.iloc[:i], x="Date", y="Price")
                fig_anim.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0))
                chart_spot.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.01) # Ultra fast sleep

    st.markdown("---")
    
    # Auto-Report
    txt = f"Report for {st.session_state.symbol}\nVolatility: {volatility:.2%}\nPrice: {curr_price}"
    b64 = base64.b64encode(txt.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64}" download="report.txt">📄 Download Summary</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
