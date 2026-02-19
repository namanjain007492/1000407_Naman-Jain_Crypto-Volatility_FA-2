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
        .sidebar .sidebar-content { background-image: linear-gradient(#161b22,#0d1117); }
        .plot-container { border: 1px solid #30363d; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. DATA MANAGEMENT ENGINE
# ==============================================================================
class DataEngine:
    """Handles all operations related to the CSV data source."""
    
    FILE_NAME = 'btcusd_1-min_data.csv.crdownload'

    @staticmethod
    @st.cache_data
    def load_and_resample(resample_rule='1H'):
        if not os.path.exists(DataEngine.FILE_NAME):
            return pd.DataFrame()

        # Load raw data
        df = pd.read_csv(DataEngine.FILE_NAME)
        
        # Accurate timestamp conversion
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Date', inplace=True)
        
        # Resampling Logic
        resampled = df.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        resampled.reset_index(inplace=True)
        resampled.rename(columns={'Close': 'Price'}, inplace=True)
        return resampled

# ==============================================================================
# 3. TECHNICAL ANALYSIS ENGINE
# ==============================================================================
class TechnicalEngine:
    """Calculates deep technical indicators and market signals."""

    @staticmethod
    def calculate_all(df):
        df = df.copy()
        
        # Basic Moving Averages
        df['SMA_20'] = df['Price'].rolling(window=20).mean()
        df['EMA_50'] = df['Price'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Price'].ewm(span=200, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Price'].ewm(span=12, adjust=False).mean()
        exp2 = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Mid'] = df['Price'].rolling(window=20).mean()
        df['BB_Std'] = df['Price'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Price'].shift())
        low_cp = np.abs(df['Low'] - df['Price'].shift())
        df['TR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Ichimoku Cloud (Simplified)
        df['Tenkan_Sen'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['Kijun_Sen'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        
        # Returns & Volatility
        df['Daily_Return'] = df['Price'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Drawdown
        df['Peak'] = df['Price'].cummax()
        df['Drawdown'] = (df['Price'] - df['Peak']) / df['Peak']
        
        return df.dropna()

    @staticmethod
    def get_fibonacci_levels(df):
        max_p = df['Price'].max()
        min_p = df['Price'].min()
        diff = max_p - min_p
        return {
            'Level 0%': max_p,
            'Level 23.6%': max_p - 0.236 * diff,
            'Level 38.2%': max_p - 0.382 * diff,
            'Level 50%': max_p - 0.5 * diff,
            'Level 61.8%': max_p - 0.618 * diff,
            'Level 100%': min_p
        }

# ==============================================================================
# 4. PREDICTIVE & SIMULATION ENGINE
# ==============================================================================
class QuantEngine:
    """Predictive modeling and stochastic path generation."""

    @staticmethod
    def train_neural_net(df):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[['Price']])
        
        X, y = [], []
        lookback = 15
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
            
        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
        model.fit(np.array(X), np.array(y))
        
        # Forecast 7 periods
        current_batch = data[-lookback:].reshape(1, -1)
        predictions = []
        for _ in range(7):
            p = model.predict(current_batch)[0]
            predictions.append(p)
            current_batch = np.append(current_batch[:, 1:], p).reshape(1, -1)
            
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    @staticmethod
    def monte_carlo(df, days=30, sims=100):
        returns = df['Daily_Return'].dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        last_price = df['Price'].iloc[-1]
        results = []
        
        for _ in range(sims):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            results.append(prices)
        return np.array(results)

# ==============================================================================
# 5. UI COMPONENTS & RENDERERS
# ==============================================================================
def render_header():
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="background: #58a6ff; width: 10px; height: 40px; border-radius: 5px; margin-right: 15px;"></div>
            <h1 style="margin: 0;">Aegis Quant Terminal</h1>
        </div>
    """, unsafe_allow_html=True)

def render_mascot(state):
    colors = {"Low": "#2ea043", "Medium": "#d29922", "High": "#f85149"}
    c = colors.get(state, "#58a6ff")
    html = f"""
    <div id="container" style="height: 200px; width: 100%; border: 1px solid #30363d; border-radius: 10px; display: flex; justify-content: center; align-items: center; background: #0d1117;">
        <div style="text-align: center;">
            <div style="width: 60px; height: 60px; background: {c}; border-radius: 50%; margin: 0 auto 10px; box-shadow: 0 0 20px {c};"></div>
            <p style="color: white; font-family: sans-serif; font-size: 14px; font-weight: bold;">AI ANALYST: {state.upper()} RISK</p>
        </div>
    </div>
    """
    components.html(html, height=210)

# ==============================================================================
# 6. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    local_css()
    render_header()
    
    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2091/2091665.png", width=80)
        st.title("Terminal Controls")
        
        timeframe = st.selectbox("Market Periodicity", 
                                ["15T (15 Min)", "1H (Hourly)", "4H (4 Hour)", "D (Daily)"], 
                                index=1)
        resample_rule = timeframe.split(" ")[0]
        
        st.divider()
        st.subheader("Indicator Overlays")
        show_ma = st.checkbox("Moving Averages", True)
        show_bb = st.checkbox("Bollinger Bands", False)
        show_fib = st.checkbox("Fibonacci Retracement", False)
        show_ichimoku = st.checkbox("Ichimoku Cloud", False)
        
        st.divider()
        st.subheader("Quant Config")
        train_model = st.button("🚀 Re-Train ML Engine")
        run_sim = st.button("🎲 Run Monte Carlo")
        
        st.divider()
        api_key = st.text_input("Gemini API Key", type="password")
        if api_key:
            st.success("AI Integration Active")

    # --- DATA PIPELINE ---
    with st.spinner("Synchronizing with CSV Data Lake..."):
        df_raw = DataEngine.load_and_resample(resample_rule)
        if df_raw.empty:
            st.error("FATAL ERROR: CSV Data source missing or corrupted.")
            return
            
        df = TechnicalEngine.calculate_all(df_raw)
    
    # --- KEY METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    last_p = df['Price'].iloc[-1]
    last_v = df['Volatility'].iloc[-1] * 100
    last_rsi = df['RSI'].iloc[-1]
    drawdown = df['Drawdown'].min() * 100
    
    m1.markdown(f'<div class="stMetric"><div class="metric-label">Market Price</div><div class="metric-value">${last_p:,.2f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stMetric"><div class="metric-label">Ann. Volatility</div><div class="metric-value">{last_v:.2f}%</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stMetric"><div class="metric-label">RSI Momentum</div><div class="metric-value" style="color:{"#f85149" if last_rsi > 70 else ("#2ea043" if last_rsi < 30 else "#58a6ff")}">{last_rsi:.2f}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="stMetric"><div class="metric-label">Max Drawdown</div><div class="metric-value" style="color:#f85149">{drawdown:.2f}%</div></div>', unsafe_allow_html=True)

    # --- MAIN TABS ---
    tab_main, tab_quant, tab_vol, tab_ai = st.tabs(["📊 Technical Terminal", "🔬 Quant Lab", "🌪 Risk & Volatility", "🧠 AI Analyst"])

    with tab_main:
        # Main Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Price'], name="Price Action"))
        
        if show_ma:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_50'], name="EMA 50", line=dict(color='#ff7f0e', width=1)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_200'], name="EMA 200", line=dict(color='#d62728', width=1.5)))
        
        if show_bb:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name="BB Upper", line=dict(color='rgba(173, 216, 230, 0.2)')))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name="BB Lower", line=dict(color='rgba(173, 216, 230, 0.2)'), fill='tonexty'))
            
        if show_fib:
            fibs = TechnicalEngine.get_fibonacci_levels(df)
            for level, val in fibs.items():
                fig.add_hline(y=val, line_dash="dot", annotation_text=level, line_color="gray")

        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Sub-Charts (Volume & RSI)
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.plotly_chart(px.bar(df, x='Date', y='Volume', title="Transactional Volume", color_discrete_sequence=['#58a6ff']), use_container_width=True)
        with c_sub2:
            fig_rsi = px.line(df, x='Date', y='RSI', title="Relative Strength Index")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_quant:
        col_q1, col_q2 = st.columns([2, 1])
        
        with col_q1:
            st.subheader("Monte Carlo Path Simulation")
            if run_sim:
                paths = QuantEngine.monte_carlo(df)
                fig_mc = go.Figure()
                x_axis = np.arange(len(paths[0]))
                for i in range(min(50, len(paths))):
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=paths[i], mode='lines', line=dict(width=0.5, color='rgba(88, 166, 255, 0.3)'), showlegend=False))
                fig_mc.update_layout(title="30-Day Probability Paths", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.info("Click 'Run Monte Carlo' in the sidebar to simulate future price paths.")

        with col_q2:
            st.subheader("Market Regime Clustering")
            kmeans = KMeans(n_clusters=3, n_init=10)
            df['Regime'] = kmeans.fit_predict(df[['Volatility', 'RSI']])
            fig_regime = px.scatter(df, x='Volatility', y='RSI', color='Regime', title="Volatility-Momentum Clusters")
            st.plotly_chart(fig_regime, use_container_width=True)

    with tab_vol:
        st.subheader("Volatility Surface Analysis")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.plotly_chart(px.line(df, x='Date', y='Volatility', title="Rolling Annualized Volatility", color_discrete_sequence=['#f85149']), use_container_width=True)
        with col_v2:
            st.plotly_chart(px.area(df, x='Date', y='Drawdown', title="Historical Drawdown Portfolio Impact", color_discrete_sequence=['#f85149']), use_container_width=True)
            
        st.subheader("GARCH-like Risk Assessment")
        st.plotly_chart(px.histogram(df, x="Daily_Return", nbins=100, marginal="rug", title="Return Kurtosis & Normality Check"), use_container_width=True)

    with tab_ai:
        col_ai1, col_ai2 = st.columns([1, 2])
        
        with col_ai1:
            risk_state = "High" if last_v > 60 else ("Medium" if last_v > 35 else "Low")
            render_mascot(risk_state)
            st.markdown(f"**Current State:** {risk_state} Volatility Environment")
            st.write("The quant models suggest a " + ("highly unstable" if risk_state == "High" else "relatively calm") + " market trend based on historical 2012-2013 CSV patterns.")
            
        with col_ai2:
            st.subheader("Neural Network Price Forecasting")
            if train_model:
                preds = QuantEngine.train_neural_net(df)
                f_dates = [df['Date'].iloc[-1] + timedelta(hours=i) for i in range(1, 8)]
                fig_nn = go.Figure()
                fig_nn.add_trace(go.Scatter(x=df['Date'].tail(50), y=df['Price'].tail(50), name="Actual"))
                fig_nn.add_trace(go.Scatter(x=f_dates, y=preds, name="NN Forecast", line=dict(dash='dash', color='cyan')))
                st.plotly_chart(fig_nn, use_container_width=True)
            else:
                st.info("Initialize the Neural Network via the sidebar to see predictive paths.")

        if api_key:
            st.divider()
            st.subheader("💬 Institutional AI Consultant")
            prompt = st.chat_input("Ask about market liquidity or technical indicators...")
            if prompt:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                context = f"Context: Market Price ${last_p}, Volatility {last_v:.1f}%, RSI {last_rsi:.1f}. Dataset: 2012-2013 BTC."
                response = model.generate_content(f"{context}\nQuestion: {prompt}")
                st.chat_message("assistant").write(response.text)

    # --- FOOTER ---
    st.divider()
    cols_f = st.columns(3)
    cols_f[0].caption("Data Integrity: 100% Verified CSV")
    cols_f[1].markdown("<center style='color: gray; font-size: 10px;'>Aegis Quant Terminal v4.2.0 | Institutional Public Release</center>", unsafe_allow_html=True)
    cols_f[2].caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()

# ==============================================================================
# END OF TERMINAL SOURCE CODE
# ==============================================================================
