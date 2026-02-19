import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit.components.v1 as components
import base64
import time
import os
import google.generativeai as genai
from scipy.stats import norm

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Aegis Crypto Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #0d1117; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
        .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
        .metric-value { font-size: 1.8rem; font-weight: bold; color: #58a6ff; }
        .stTabs [data-baseweb="tab-list"] { background-color: #161b22; border-radius: 8px; padding: 5px; }
        .stTabs [data-baseweb="tab"] { color: #8b949e; transition: 0.3s; }
        .stTabs [data-baseweb="tab"]:hover { color: #58a6ff; }
        .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #58a6ff !important; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. DATA MANAGEMENT ENGINE
# ==============================================================================
class DataEngine:
    FILE_NAME = 'btcusd_1-min_data.csv.crdownload'

    @staticmethod
    @st.cache_data
    def load_and_resample(resample_rule='1H'):
        if not os.path.exists(DataEngine.FILE_NAME):
            # Simulation for development if file is missing
            dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
            return pd.DataFrame({
                'Date': dates,
                'Open': np.random.randn(1000).cumsum() + 50000,
                'High': np.random.randn(1000).cumsum() + 50100,
                'Low': np.random.randn(1000).cumsum() + 49900,
                'Price': np.random.randn(1000).cumsum() + 50000,
                'Volume': np.random.randint(100, 5000, 1000)
            })

        df = pd.read_csv(DataEngine.FILE_NAME)
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Date', inplace=True)
        resampled = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        resampled.reset_index(inplace=True)
        resampled.rename(columns={'Close': 'Price'}, inplace=True)
        return resampled

# ==============================================================================
# 3. TECHNICAL ANALYSIS ENGINE
# ==============================================================================
class TechnicalEngine:
    @staticmethod
    def calculate_all(df):
        df = df.copy()
        
        # MACD with Histogram
        exp1 = df['Price'].ewm(span=12, adjust=False).mean()
        exp2 = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # Bollinger Bands %B & Bandwidth
        df['BB_Mid'] = df['Price'].rolling(window=20).mean()
        df['BB_Std'] = df['Price'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        
        # Returns
        df['Daily_Return'] = df['Price'].pct_change()
        df['Cum_Return'] = (1 + df['Daily_Return']).cumprod()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        df['Drawdown'] = (df['Price'] - df['Price'].cummax()) / df['Price'].cummax()
        
        return df.dropna()

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    local_css()
    
    # 1. AI Setup from Secrets
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except:
        pass

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚡ Aegis Controls")
        tf_label = st.selectbox("Timeframe", ["15T (15 Min)", "1H (Hourly)", "D (Daily)"], index=1)
        resample_rule = tf_label.split(" ")[0]
        st.divider()
        st.subheader("Indicator Toggles")
        show_macd = st.checkbox("Show MACD Histogram", True)
        show_equity = st.checkbox("Show Equity Curve", True)
        show_corr = st.checkbox("Correlation Matrix", False)

    # --- DATA PIPELINE ---
    df_raw = DataEngine.load_and_resample(resample_rule)
    df = TechnicalEngine.calculate_all(df_raw)
    
    # --- METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")
    m2.metric("Volatility (Ann)", f"{df['Volatility'].iloc[-1]*100:.2f}%")
    m3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
    m4.metric("Strategy Growth", f"{((df['Cum_Return'].iloc[-1]-1)*100):.2f}%")

    tab_terminal, tab_quant, tab_chat = st.tabs(["📊 Market Terminal", "🔬 Quant Analytics", "💬 AI Assistant"])

    with tab_terminal:
        # Main Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Price'], name="Price")])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        col_bot1, col_bot2 = st.columns(2)
        with col_bot1:
            if show_macd:
                # Essential MACD Histogram
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name="Histogram", marker_color='gray'))
                fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='cyan', width=1.5)))
                fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal", line=dict(color='orange', width=1.5)))
                fig_macd.update_layout(title="MACD Momentum", height=300, template="plotly_dark", margin=dict(t=30))
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with col_bot2:
            if show_equity:
                # Cumulative Returns (Equity Curve)
                fig_cum = px.area(df, x='Date', y='Cum_Return', title="Cumulative Growth (Buy & Hold)")
                fig_cum.update_layout(height=300, template="plotly_dark", margin=dict(t=30))
                st.plotly_chart(fig_cum, use_container_width=True)

    with tab_quant:
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.subheader("Volatility Bandwidth")
            st.plotly_chart(px.line(df, x='Date', y='BB_Width', title="BB Width (Squeeze Indicator)"), use_container_width=True)
        with q_col2:
            st.subheader("Risk Heatmap")
            if show_corr:
                corr = df[['Price', 'Volume', 'Volatility', 'RSI']].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
            else:
                st.info("Enable Correlation Matrix in Sidebar.")

    with tab_chat:
        st.subheader("Institutional AI Consultant")
        if "GEMINI_API_KEY" not in st.secrets:
            st.warning("Add `GEMINI_API_KEY` to your Streamlit Secrets to enable.")
        else:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Analyze the current RSI and MACD for me..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                model = genai.GenerativeModel('gemini-pro')
                context = f"Current Market Data: Price ${df['Price'].iloc[-1]}, RSI {df['RSI'].iloc[-1]:.1f}, MACD Hist {df['MACD_Hist'].iloc[-1]:.4f}."
                response = model.generate_content(f"{context}\n\nUser Question: {prompt}")
                
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

if __name__ == "__main__":
    main()
