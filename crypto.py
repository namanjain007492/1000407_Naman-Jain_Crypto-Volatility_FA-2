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
# 🔹 HELPER FUNCTIONS 
# ==========================================
@st.cache_data(ttl=3600)
def load_data(symbol, start_date, end_date):
    # --- FILE INTEGRATION LOGIC ---
    file_path = 'btcusd_1-min_data.csv.crdownload'
    if os.path.exists(file_path) and "BTC" in symbol:
        try:
            df_raw = pd.read_csv(file_path)
            # Convert Unix Timestamp to Datetime
            df_raw['Date'] = pd.to_datetime(df_raw['Timestamp'], unit='s')
            df_raw.set_index('Date', inplace=True)
            
            # Resample 1-minute data to Daily for the app's indicators
            df_daily = df_raw.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            df_daily.reset_index(inplace=True)
            
            # Filter by selected date range
            mask = (df_daily['Date'].dt.date >= start_date) & (df_daily['Date'].dt.date <= end_date)
            df_filtered = df_daily.loc[mask].copy()
            
            if not df_filtered.empty:
                return df_filtered, False  # Data is real CSV data
        except Exception as e:
            st.warning(f"CSV loading failed: {e}. Falling back to API/Simulation.")

    # --- ORIGINAL FALLBACK LOGIC ---
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        df = yf.download(symbol, start=start_str, end=end_str, auto_adjust=False, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.reset_index(inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df, False  
    except Exception:
        pass

    # Simulation fallback
    days = (end_date - start_date).days
    if days < 30: days = 365
    np.random.seed(42 if "BTC" in symbol else 43)
    dates = pd.date_range(start=start_str, periods=days, freq="D")
    base_price = 50000 if "BTC" in symbol else 3000
    volatility = 0.03
    returns = np.random.normal(0.001, volatility, days)
    prices = base_price * np.exp(np.cumsum(returns))
    df_fallback = pd.DataFrame({
        "Date": dates,
        "Open": prices * np.random.uniform(0.98, 1.01, days),
        "High": prices * np.random.uniform(1.01, 1.05, days),
        "Low": prices * np.random.uniform(0.95, 0.99, days),
        "Close": prices,
        "Volume": np.random.uniform(10000, 500000, days)
    })
    return df_fallback, True  

def clean_data(df):
    if df.empty: return df
    df.columns = [str(c).capitalize() for c in df.columns]
    if "Close" in df.columns:
        df.rename(columns={"Close": "Price"}, inplace=True)
    df.ffill(inplace=True)
    if "Price" in df.columns:
        df.dropna(subset=["Price"], inplace=True) 
    return df

def calculate_indicators(df, window=20):
    df["Daily_Return"] = df["Price"].pct_change()
    df["Rolling_Mean"] = df["Price"].rolling(window=window).mean()
    df["Rolling_Std"] = df["Daily_Return"].rolling(window=window).std()
    df["Rolling_Volatility"] = df["Rolling_Std"] * np.sqrt(252)
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    
    df["BB_Upper"] = df["Rolling_Mean"] + (df["Price"].rolling(window=window).std() * 2)
    df["BB_Lower"] = df["Rolling_Mean"] - (df["Price"].rolling(window=window).std() * 2)
    
    ema_12 = df["Price"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    delta = df["Price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    cumulative_max = df["Price"].cummax()
    df["Drawdown"] = (df["Price"] - cumulative_max) / cumulative_max
    
    return df.dropna()

def simulate_patterns(df, mode, amp, freq, drift, noise_int):
    t = np.arange(len(df))
    base = float(df["Price"].iloc[0])
    if mode == "Sine wave": sim = base + amp * np.sin(2 * np.pi * freq * (t / len(t)))
    elif mode == "Cosine wave": sim = base + amp * np.cos(2 * np.pi * freq * (t / len(t)))
    elif mode == "Random noise": sim = base + np.random.normal(0, noise_int, len(t))
    elif mode == "Drift (integral effect)": sim = base + drift * t 
    else: sim = base + drift * t + amp * np.sin(2 * np.pi * freq * (t / len(t))) + np.random.normal(0, noise_int, len(t))
    return sim

def render_3d_mascot(volatility_state):
    colors = {"Low": "#00ff00", "Medium": "#ffff00", "High": "#ff0000"}
    hex_color = colors.get(volatility_state, "#00ff00")
    html_code = f"""
    <div id="mascot-container" style="width: 100%; height: 220px; display: flex; justify-content: center; align-items: center; position: relative;">
        <div style="position: absolute; top: 10px; background: rgba(255,255,255,0.1); padding: 5px 10px; border-radius: 15px; color: white; font-family: sans-serif; font-size: 12px;">
            AI: Market is {volatility_state} Risk
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{alpha: true, antialias: true}});
        renderer.setSize(220, 220);
        document.getElementById('mascot-container').appendChild(renderer.domElement);
        const geometry = new THREE.OctahedronGeometry(1.5, 0);
        const material = new THREE.MeshStandardMaterial({{ 
            color: "{hex_color}", wireframe: true, emissive: "{hex_color}", emissiveIntensity: 0.6 
        }});
        const gem = new THREE.Mesh(geometry, material);
        scene.add(gem);
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(10, 10, 10);
        scene.add(light);
        camera.position.z = 3.5;
        function animate() {{
            requestAnimationFrame(animate);
            gem.rotation.x += 0.01;
            gem.rotation.y += 0.02;
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """
    components.html(html_code, height=230)

def ai_analysis(df):
    latest_vol = float(df["Rolling_Volatility"].iloc[-1])
    if latest_vol < 0.4: state, color, explanation = "Low", "green", "Prices are relatively stable."
    elif latest_vol < 0.7: state, color, explanation = "Medium", "orange", "Normal market fluctuations occurring."
    else: state, color, explanation = "High", "red", "Expect large swings. Capital preservation advised."
    st.markdown(f"### 🤖 AI Assessment: **:{color}[{state} Volatility]**")
    st.info(f"**Analysis:** {explanation}")
    return state

def generate_pdf_report(df, vol_state, symbol, currency_sym):
    report_content = f"CRYPTO VOLATILITY REPORT: {symbol}\nGenerated: {datetime.today().strftime('%Y-%m-%d')}\nFinal Price: {currency_sym}{float(df['Price'].iloc[-1]):,.2f}"
    b64 = base64.b64encode(report_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{symbol}_Report.txt" class="metric-card" style="color:#00ffcc; text-decoration:none;">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# ==========================================
# 🔹 MAIN DASHBOARD 
# ==========================================
def main():
    st.title("⚡ Crypto Volatility Visualizer – Elite Public Edition")
    
    if "selected_crypto" not in st.session_state:
        st.session_state.selected_crypto = "BTC-USD"
    if "currency" not in st.session_state:
        st.session_state.currency = "USD"
    
    with st.sidebar:
        st.header("⚙️ Settings Panel")
        
        # --- ADJUST DEFAULT DATES IF LOCAL CSV EXISTS ---
        file_path = 'btcusd_1-min_data.csv.crdownload'
        default_start = pd.to_datetime("2012-01-01") if os.path.exists(file_path) else pd.to_datetime("2023-01-01")
        default_end = pd.to_datetime("2013-03-18") if os.path.exists(file_path) else datetime.today().date()
        
        date_range = st.date_input("Date Range", [default_start, default_end])
        if len(date_range) != 2: st.stop()
            
        symbol = st.selectbox("Crypto Selector", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
        st.session_state.selected_crypto = symbol
        
        vol_window = st.slider("Volatility Window", 5, 50, 20)
        sim_toggle = st.checkbox("Enable Simulation Mode")

    # --- DATA PIPELINE ---
    raw_df, is_simulated = load_data(st.session_state.selected_crypto, date_range[0], date_range[1])
    
    if raw_df.empty:
        st.error("⚠️ No data found for the selected range.")
        st.stop()
        
    df = clean_data(raw_df)
    df = calculate_indicators(df, window=vol_window)
    
    # --- UI DISPLAY ---
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">${float(df["Price"].iloc[-1]):,.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Volatility</div><div class="metric-value">{float(df["Rolling_Volatility"].iloc[-1])*100:.2f}%</div></div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📊 Charts", "📐 Simulation", "🧠 AI Analysis"])
    
    with t1:
        fig1 = px.line(df, x="Date", y="Price", title=f"Price Trend ({symbol})")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig_vol = px.line(df, x="Date", y="Rolling_Volatility", title="Rolling Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

    with t2:
        if sim_toggle:
            df["Simulated"] = simulate_patterns(df, "Combined", 5000, 5, 10, 2000)
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Real Price"))
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Simulated"], name="Simulated"))
            st.plotly_chart(fig_sim, use_container_width=True)

    with t3:
        vol_state = ai_analysis(df)
        render_3d_mascot(vol_state)
        generate_pdf_report(df, vol_state, symbol, "$")

if __name__ == "__main__":
    main()
