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
import os
import google.generativeai as genai

# ==============================================================================
# 1. PAGE CONFIGURATION & INSTITUTIONAL STYLING
# ==============================================================================
st.set_page_config(page_title="Aegis Crypto Quant Terminal", page_icon="⚡", layout="wide")

def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main { background-color: #0d1117; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
        .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }
        .metric-value { font-size: 1.8rem; font-weight: bold; color: #58a6ff; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. STAGE 4: DATA PREPARATION ENGINE
# ==============================================================================
class DataEngine:
    FILE_NAME = 'btcusd_1-min_data.csv.crdownload'

    @staticmethod
    @st.cache_data
    def load_data(resample_rule='1H'):
        if not os.path.exists(DataEngine.FILE_NAME):
            # Fallback for demo/deployment if file is missing
            dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
            df = pd.DataFrame({
                'Timestamp': dates.view(np.int64) // 10**9,
                'Open': np.random.randn(500).cumsum() + 50000,
                'High': np.random.randn(500).cumsum() + 50500,
                'Low': np.random.randn(500).cumsum() + 49500,
                'Close': np.random.randn(500).cumsum() + 50000,
                'Volume': np.random.uniform(10, 1000, 500)
            })
        else:
            df = pd.read_csv(DataEngine.FILE_NAME)

        # Stage 4 Requirements
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Date', inplace=True)
        resampled = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        resampled.reset_index(inplace=True)
        resampled.rename(columns={'Close': 'Price'}, inplace=True)
        return resampled

# ==============================================================================
# 3. STAGE 5: TECHNICAL ANALYSIS (Essential Metrics)
# ==============================================================================
class TechnicalEngine:
    @staticmethod
    def calculate_all(df):
        df = df.copy()
        # MACD with Histogram
        ema12 = df['Price'].ewm(span=12).mean()
        ema26 = df['Price'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Squeeze
        df['SMA20'] = df['Price'].rolling(20).mean()
        df['BB_Std'] = df['Price'].rolling(20).std()
        df['BB_Width'] = (df['BB_Std'] * 4) / df['SMA20']
        
        # Risk Indicators
        df['Return'] = df['Price'].pct_change()
        df['Equity_Curve'] = (1 + df['Return'].fillna(0)).cumprod()
        df['Volatility'] = df['Return'].rolling(20).std() * np.sqrt(252)
        return df.dropna()

# ==============================================================================
# 4. MAIN INTERFACE LOGIC
# ==============================================================================
def main():
    local_css()
    
    # Secure API Key from Secrets
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    
    with st.sidebar:
        st.title("⚡ Terminal Settings")
        timeframe = st.selectbox("Frequency", ["15T", "1H", "4H", "D"], index=1)
        st.divider()
        st.subheader("Indicator Toggles")
        show_macd = st.checkbox("MACD Histogram", True)
        show_equity = st.checkbox("Equity Curve", True)

    # Data Pipeline
    df = TechnicalEngine.calculate_all(DataEngine.load_data(timeframe))
    
    # Key Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")
    m2.metric("Ann. Volatility", f"{df['Volatility'].iloc[-1]*100:.2f}%")
    m3.metric("BB Width", f"{df['BB_Width'].iloc[-1]:.4f}")
    m4.metric("Growth", f"{((df['Equity_Curve'].iloc[-1]-1)*100):.2f}%")

    tab_terminal, tab_ai = st.tabs(["📊 Market Terminal", "🧠 AI Analyst"])

    with tab_terminal:
        # Main Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Price'])])
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        col_bot1, col_bot2 = st.columns(2)
        with col_bot1:
            if show_macd:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name="Momentum", marker_color='gray'))
                fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='cyan')))
                fig_macd.update_layout(title="MACD Momentum Histogram", height=300, template="plotly_dark")
                st.plotly_chart(fig_macd, use_container_width=True)
        with col_bot2:
            if show_equity:
                fig_eq = px.area(df, x='Date', y='Equity_Curve', title="Cumulative Growth (Equity Curve)")
                fig_eq.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_eq, use_container_width=True)

    with tab_ai:
        st.subheader("Institutional AI Consultant")
        if not gemini_api_key:
            st.error("⚠️ Gemini API Key missing! Add it to your Streamlit Secrets.")
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                # --- AUTO-DISCOVERY FIX ---
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                target_model = available_models[0] if available_models else "gemini-pro"
                model = genai.GenerativeModel(target_model)

                if "messages" not in st.session_state: st.session_state.messages = []
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])

                if prompt := st.chat_input("Analyze these indicators for me..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)

                    context = f"Market Status: Price ${df['Price'].iloc[-1]}, MACD Hist {df['MACD_Hist'].iloc[-1]:.4f}, Volatility {df['Volatility'].iloc[-1]*100:.1f}%."
                    response = model.generate_content(f"{context}\n\nUser: {prompt}")
                    
                    with st.chat_message("assistant"): st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"API Connection Error: {e}")

if __name__ == "__main__":
    main()
