import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import base64
import time
import os
import google.generativeai as genai

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING (Institutional UI)
# ==============================================================================
st.set_page_config(page_title="Aegis Crypto Quant Terminal", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #1e1e24; padding: 20px; border-radius: 12px; border: 1px solid #333; text-align: center; }
    .metric-value { font-size: 24px; font-weight: bold; color: #00ffcc; }
    .metric-label { font-size: 14px; color: #b0b0b0; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. STAGE 4: DATA PREPARATION ENGINE (Requirement )
# ==============================================================================
class DataEngine:
    """Handles loading, cleaning, and preparation of delivery data."""
    FILE_NAME = 'btcusd_1-min_data.csv.crdownload' # Replace with your actual file path

    @staticmethod
    @st.cache_data
    def load_and_prepare():
        # Check if local file exists, otherwise generate high-fidelity dummy data for deployment
        if not os.path.exists(DataEngine.FILE_NAME):
            dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
            df = pd.DataFrame({
                'Timestamp': dates.view(np.int64) // 10**9,
                'Open': np.random.randn(500).cumsum() + 45000,
                'High': np.random.randn(500).cumsum() + 45500,
                'Low': np.random.randn(500).cumsum() + 44500,
                'Close': np.random.randn(500).cumsum() + 45000,
                'Volume': np.random.uniform(10, 100, 500)
            })
        else:
            df = pd.read_csv(DataEngine.FILE_NAME)

        # Stage 4 Steps: Clean and Subset 
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.rename(columns={'Close': 'Price'}, inplace=True)
        df.ffill(inplace=True) # Handle missing values 
        df = df.dropna().tail(300) # Subset for performance 
        return df

# ==============================================================================
# 3. STAGE 5: TECHNICAL ANALYSIS & VISUALIZATION ENGINE (Requirement )
# ==============================================================================
class TechnicalEngine:
    @staticmethod
    def calculate_indicators(df, window=20):
        df = df.copy()
        # Returns & Volatility
        df["Daily_Return"] = df["Price"].pct_change()
        df["Volatility"] = df["Daily_Return"].rolling(window).std() * np.sqrt(252)
        
        # Indicators
        df["SMA_20"] = df["Price"].rolling(window).mean()
        df["BB_Upper"] = df["SMA_20"] + (df["Price"].rolling(window).std() * 2)
        df["BB_Lower"] = df["SMA_20"] - (df["Price"].rolling(window).std() * 2)
        
        # MACD
        ema12 = df["Price"].ewm(span=12).mean()
        ema26 = df["Price"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        
        # RSI
        delta = df["Price"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + gain / loss))
        
        # Drawdown
        df["Drawdown"] = (df["Price"] - df["Price"].cummax()) / df["Price"].cummax()
        return df.dropna()

# ==============================================================================
# 4. QUANT & AI CORE
# ==============================================================================
def render_mascot(volatility):
    color = "#00ff00" if volatility < 0.4 else "#ff0000"
    html = f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:200px; border:1px solid #333; border-radius:12px;">
        <div style="width:60px; height:60px; background:{color}; border-radius:50%; box-shadow: 0 0 20px {color}; animation: pulse 2s infinite;"></div>
        <p style="color:white; font-family:sans-serif; margin-top:15px; font-weight:bold;">SYSTEM RISK: {('HIGH' if volatility > 0.4 else 'LOW')}</p>
    </div>
    <style>@keyframes pulse {{ 0% {{transform:scale(0.95);}} 50% {{transform:scale(1.05);}} 100% {{transform:scale(0.95);}} }}</style>
    """
    components.html(html, height=210)

def run_monte_carlo(df):
    last_price = df['Price'].iloc[-1]
    returns = df['Daily_Return'].dropna()
    sims = []
    for _ in range(50):
        path = [last_price]
        for _ in range(30):
            path.append(path[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
        sims.append(path)
    fig = go.Figure()
    for s in sims: fig.add_trace(go.Scatter(y=s, mode='lines', line=dict(width=1, color='rgba(0,255,204,0.1)'), showlegend=False))
    fig.update_layout(title="Monte Carlo 30-Day Path Simulation", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 6. STAGE 6: MAIN INTERFACE (Requirement )
# ==============================================================================
def main():
    st.title("⚡ Aegis Crypto Quant Terminal")
    
    # 🔹 SIDEBAR CONTROLS 
    with st.sidebar:
        st.header("Terminal Settings")
        vol_window = st.slider("Volatility Window", 5, 50, 20)
        st.divider()
        st.subheader("Math Simulation Controls ")
        sim_mode = st.selectbox("Pattern", ["Sine wave", "Random noise", "Drift"])
        amp = st.slider("Amplitude", 100, 5000, 1000)
        drift_val = st.slider("Drift", -10.0, 10.0, 2.0)
        st.divider()
        # Secure API Key from st.secrets
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")

    # 🔹 DATA PIPELINE
    df_raw = DataEngine.load_and_prepare()
    df = TechnicalEngine.calculate_indicators(df_raw, vol_window)
    
    # 🔹 TOP METRICS
    m1, m2, m3, m4 = st.columns(4)
    last_p, last_v = df['Price'].iloc[-1], df['Volatility'].iloc[-1]
    m1.markdown(f'<div class="metric-card"><div class="metric-label">Market Price</div><div class="metric-value">${last_p:,.2f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">Ann. Volatility</div><div class="metric-value">{last_v*100:.2f}%</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-label">RSI Index</div><div class="metric-value">{df["RSI"].iloc[-1]:.1f}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value">{df["Drawdown"].min()*100:.2f}%</div></div>', unsafe_allow_html=True)

    # 🔹 MAIN TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Visuals", "📐 Simulation Lab", "🔬 Quant Analytics", "🧠 AI Analyst"])

    with tab1:
        st.subheader("Interactive Market Visualizations ")
        # 1. Price Line Graph 
        st.plotly_chart(px.line(df, x='Date', y='Price', title="Bitcoin Price Over Time"), use_container_width=True)
        # 2. High vs Low 
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Scatter(x=df['Date'], y=df['High'], name="High", line=dict(color='green')))
        fig_hl.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name="Low", line=dict(color='red')))
        fig_hl.update_layout(title="High vs Low Comparison", template="plotly_dark")
        st.plotly_chart(fig_hl, use_container_width=True)
        # 3. Volume Analysis 
        st.plotly_chart(px.bar(df, x='Date', y='Volume', title="Trading Volume Analysis"), use_container_width=True)
        # 4. Bollinger Bands (Stable vs Volatile Periods )
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(dash='dot'), name="Upper Band"))
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="Price"))
        fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(dash='dot'), name="Lower Band"))
        fig_bb.update_layout(title="Bollinger Bands: Volatility Channels", template="plotly_dark")
        st.plotly_chart(fig_bb, use_container_width=True)

    with tab2:
        st.subheader("Mathematical Simulation Engine ")
        t = np.arange(len(df))
        if sim_mode == "Sine wave": sim_p = df['Price'].iloc[0] + amp * np.sin(0.1 * t)
        elif sim_mode == "Random noise": sim_p = df['Price'].iloc[0] + np.random.normal(0, amp, len(t))
        else: sim_p = df['Price'].iloc[0] + drift_val * t
        
        fig_sim = px.line(x=df['Date'], y=sim_p, title=f"Simulated {sim_mode} Pattern")
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab3:
        col_q1, col_q2 = st.columns(2)
        with col_q1: 
            run_monte_carlo(df)
            st.plotly_chart(px.line(df, x='Date', y='RSI', title="Relative Strength Index (RSI)"), use_container_width=True)
        with col_q2:
            st.plotly_chart(px.area(df, x='Date', y='Drawdown', title="Historical Drawdown Impact"), use_container_width=True)
            st.plotly_chart(px.histogram(df, x="Daily_Return", title="Daily Return Distribution"), use_container_width=True)

    with tab4:
        c_a1, c_a2 = st.columns([1, 2])
        with c_a1: render_mascot(last_v)
        with c_a2:
            st.subheader("Institutional AI Consultant")
            if not gemini_key:
                st.info("To activate the AI, add your `GEMINI_API_KEY` to the Streamlit Secrets tab.")
            else:
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-pro')
                if "msgs" not in st.session_state: st.session_state.msgs = []
                for m in st.session_state.msgs:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
                if prompt := st.chat_input("Analyze market volatility..."):
                    st.session_state.msgs.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    ctx = f"Context: BTC Price ${last_p}, Volatility {last_v*100:.1f}%."
                    response = model.generate_content(f"{ctx}\n\nUser: {prompt}")
                    with st.chat_message("assistant"): st.markdown(response.text)
                    st.session_state.msgs.append({"role": "assistant", "content": response.text})

if __name__ == "__main__":
    main()
