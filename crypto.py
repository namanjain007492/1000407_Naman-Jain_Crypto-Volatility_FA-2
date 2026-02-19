import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

# ==========================================
# PAGE CONFIGURATION & UI DESIGN
# ==========================================
st.set_page_config(page_title="Crypto Volatility Visualizer", page_icon="₿", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #1e1e24; padding: 24px; border-radius: 12px; text-align: center; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-value { font-size: 28px; font-weight: bold; color: #00ffcc; margin-top: 10px; }
    .metric-label { font-size: 16px; color: #b0b0b0; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 DATA LOADING & INTEGRATION (CSV FOCUS)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(symbol, start_date, end_date):
    # CSV filename provided in project
    csv_file = 'btcusd_1-min_data.csv.crdownload'
    
    # Check if CSV exists and we are looking at BTC
    if os.path.exists(csv_file) and "BTC" in symbol:
        try:
            df_csv = pd.read_csv(csv_file)
            # 1. Convert Unix Timestamp to Datetime
            df_csv['Date'] = pd.to_datetime(df_csv['Timestamp'], unit='s')
            df_csv.set_index('Date', inplace=True)
            
            # 2. Resample 1-min data to Daily for volatility calculations
            # Using 'D' resampling to match the financial math in the app
            df_daily = df_csv.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            df_daily.reset_index(inplace=True)
            df_daily['Date'] = df_daily['Date'].dt.tz_localize(None)
            
            # 3. Filter by User Date Range
            mask = (df_daily['Date'].dt.date >= start_date) & (df_daily['Date'].dt.date <= end_date)
            df_final = df_daily.loc[mask].copy()
            
            if not df_final.empty:
                return df_final, False # Return as real data
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

    # Fallback to YFinance or Simulation if CSV fails/isn't BTC
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            df.reset_index(inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df, False
    except: pass

    # Absolute fallback: Simulation
    days = (end_date - start_date).days if (end_date - start_date).days > 30 else 365
    dates = pd.date_range(end=datetime.now(), periods=days)
    prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, days)))
    df_sim = pd.DataFrame({"Date": dates, "Open": prices*0.99, "High": prices*1.02, "Low": prices*0.98, "Close": prices, "Volume": np.random.randint(1000, 5000, days)})
    return df_sim, True

def clean_data(df):
    if df.empty: return df
    df.columns = [str(c).capitalize() for c in df.columns]
    if "Close" in df.columns: df.rename(columns={"Close": "Price"}, inplace=True)
    df.ffill(inplace=True)
    return df

def calculate_indicators(df, window=20):
    df = df.copy()
    df["Daily_Return"] = df["Price"].pct_change()
    df["Rolling_Mean"] = df["Price"].rolling(window=window).mean()
    df["Rolling_Std"] = df["Daily_Return"].rolling(window=window).std()
    df["Rolling_Volatility"] = df["Rolling_Std"] * np.sqrt(252)
    df["BB_Upper"] = df["Rolling_Mean"] + (df["Price"].rolling(window=window).std() * 2)
    df["BB_Lower"] = df["Rolling_Mean"] - (df["Price"].rolling(window=window).std() * 2)
    
    # RSI
    delta = df["Price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain/loss)))
    
    # MACD
    ema12 = df["Price"].ewm(span=12).mean()
    ema26 = df["Price"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    
    df["Drawdown"] = (df["Price"] - df["Price"].cummax()) / df["Price"].cummax()
    return df.dropna()

# ==========================================
# 🔹 AI & MASCOT RENDERING
# ==========================================
def render_3d_mascot(volatility_state):
    colors = {"Low": "#00ff00", "Medium": "#ffff00", "High": "#ff0000"}
    hex_color = colors.get(volatility_state, "#00ff00")
    html_code = f"""
    <div id="mascot-container" style="width: 100%; height: 220px; display: flex; justify-content: center; align-items: center; position: relative;">
        <div style="position: absolute; top: 10px; background: rgba(255,255,255,0.1); padding: 5px 10px; border-radius: 15px; color: white; font-family: sans-serif; font-size: 12px;">AI: Market is {volatility_state} Risk</div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{alpha: true, antialias: true}});
        renderer.setSize(220, 220);
        document.getElementById('mascot-container').appendChild(renderer.domElement);
        const geometry = new THREE.OctahedronGeometry(1.5, 0);
        const material = new THREE.MeshStandardMaterial({{ color: "{hex_color}", wireframe: true, emissive: "{hex_color}", emissiveIntensity: 0.6 }});
        const gem = new THREE.Mesh(geometry, material);
        scene.add(gem);
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(10, 10, 10);
        scene.add(light);
        camera.position.z = 3.5;
        function animate() {{ requestAnimationFrame(animate); gem.rotation.x += 0.01; gem.rotation.y += 0.02; renderer.render(scene, camera); }}
        animate();
    </script>
    """
    components.html(html_code, height=230)

def ai_analysis(df):
    latest_vol = float(df["Rolling_Volatility"].iloc[-1])
    if latest_vol < 0.4: state, color = "Low", "green"
    elif latest_vol < 0.7: state, color = "Medium", "orange"
    else: state, color = "High", "red"
    st.markdown(f"### 🤖 AI Assessment: **:{color}[{state} Volatility]**")
    return state

# ==========================================
# 🔹 MAIN DASHBOARD
# ==========================================
def main():
    st.title("⚡ Crypto Volatility Visualizer – Project Edition")

    with st.sidebar:
        st.header("⚙️ Data Settings")
        
        # Detected Date Range for CSV (2012-2013)
        csv_detected = os.path.exists('btcusd_1-min_data.csv.crdownload')
        default_start = pd.to_datetime("2012-01-01") if csv_detected else pd.to_datetime("2023-01-01")
        default_end = pd.to_datetime("2013-03-18") if csv_detected else datetime.today().date()
        
        date_range = st.date_input("Date Range", [default_start, default_end])
        symbol = st.selectbox("Asset Selection", ["BTC-USD", "ETH-USD", "SOL-USD"])
        
        if csv_detected and "BTC" in symbol:
            st.success("📁 Using local CSV Data")
        
        vol_window = st.slider("Smoothing Window", 5, 50, 20)
        sim_toggle = st.checkbox("Enable Math Simulation Mode")

    # Pipeline
    raw_df, is_simulated = load_data(symbol, date_range[0], date_range[1])
    df = clean_data(raw_df)
    df = calculate_indicators(df, window=vol_window)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    price = float(df["Price"].iloc[-1])
    vol = float(df["Rolling_Volatility"].iloc[-1]) * 100
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">${price:,.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Annual Vol</div><div class="metric-value">{vol:.2f}%</div></div>', unsafe_allow_html=True)

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Main Graphs", "📐 Simulation", "🧠 AI Insights"])
    
    with t1:
        st.plotly_chart(px.line(df, x="Date", y="Price", title="Price Movement"), use_container_width=True)
        st.plotly_chart(px.line(df, x="Date", y="Rolling_Volatility", title="Volatility Over Time"), use_container_width=True)
        
        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Price"))
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Upper BB", line=dict(dash='dot')))
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Lower BB", line=dict(dash='dot')))
        st.plotly_chart(fig_bb, use_container_width=True)

    with t2:
        if sim_toggle:
            t = np.arange(len(df))
            df["Sim"] = df["Price"].iloc[0] + 500 * np.sin(2 * np.pi * 5 * (t / len(t)))
            fig_sim = px.line(df, x="Date", y=["Price", "Sim"], title="Price vs Sine Simulation")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Enable Simulation in sidebar to see math models.")

    with t3:
        vol_state = ai_analysis(df)
        render_3d_mascot(vol_state)
        st.dataframe(df.tail())

if __name__ == "__main__":
    main()
