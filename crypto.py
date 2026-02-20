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
    .math-card { background-color: #2b1d3d; border-left: 5px solid #a855f7; padding: 20px; border-radius: 8px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 DATA PIPELINE (Reading from Local CSV) 
# ==========================================
@st.cache_data(ttl=3600)
def load_base_csv():
    """Loads the 1-minute CSV and resamples to Daily Data for performance."""
    file_path = "btcusd_1-min_data.csv.crdownload"
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}. Make sure the file is in the same folder as app.py")
        return pd.DataFrame()

    # Convert Unix Timestamp to Datetime
    if 'Timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.drop(columns=['Timestamp'], inplace=True)
    
    # Resample 1-min data to Daily to prevent browser crashing
    df.set_index('Date', inplace=True)
    df_daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    return df_daily

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
    
    # Bollinger Bands
    df["BB_Upper"] = df["Rolling_Mean"] + (df["Price"].rolling(window=window).std() * 2)
    df["BB_Lower"] = df["Rolling_Mean"] - (df["Price"].rolling(window=window).std() * 2)
    
    # RSI
    delta = df["Price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Drawdown
    cumulative_max = df["Price"].cummax()
    df["Drawdown"] = (df["Price"] - cumulative_max) / cumulative_max
    
    return df.dropna()

# ==========================================
# 🔹 MATHEMATICS SIMULATION ENGINE (FA-2 RUBRIC)
# ==========================================
def simulate_patterns(df, mode, amp, freq, drift, noise_int):
    """Applies Sine, Cosine, Drift (Integral), and Noise to the baseline price."""
    t = np.arange(len(df))
    base = float(df["Price"].iloc[0]) # Starting price from your data
    
    if mode == "Sine wave":
        sim = base + amp * np.sin(2 * np.pi * freq * (t / len(t)))
    elif mode == "Cosine wave":
        sim = base + amp * np.cos(2 * np.pi * freq * (t / len(t)))
    elif mode == "Random noise":
        sim = base + np.random.normal(0, noise_int, len(t))
    elif mode == "Drift (integral effect)":
        sim = base + drift * t 
    else: 
        # Combined
        sim = base + drift * t + amp * np.sin(2 * np.pi * freq * (t / len(t))) + np.random.normal(0, noise_int, len(t))
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
    if latest_vol < 0.4:
        state, color = "Low", "green"
    elif latest_vol < 0.7:
        state, color = "Medium", "orange"
    else:
        state, color = "High", "red"
        
    st.markdown(f"### 🤖 AI Assessment: **:{color}[{state} Volatility]**")
    return state

# ==========================================
# 🔹 MAIN DASHBOARD 
# ==========================================
def main():
    st.title("⚡ Crypto Volatility Visualizer – FA-2 Math Edition")
    
    if "selected_crypto" not in st.session_state:
        st.session_state.selected_crypto = "BTC-USD (Local Data)"
    if "currency" not in st.session_state:
        st.session_state.currency = "USD"
        
    base_df = load_base_csv()
    if base_df.empty:
        st.stop()
        
    min_csv_date = base_df['Date'].min().date()
    max_csv_date = base_df['Date'].max().date()
    default_start = max(min_csv_date, max_csv_date - timedelta(days=365)) 
    
    with st.sidebar:
        st.header("🔑 AI Assistant API")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
            
        st.header("⚙️ Data Settings")
        curr_options = ["USD", "EUR"]
        selected_curr = st.radio("Display Currency", curr_options, index=curr_options.index(st.session_state.currency), horizontal=True)
        if selected_curr != st.session_state.currency:
            st.session_state.currency = selected_curr
            st.rerun()
            
        currency_sym = "€" if st.session_state.currency == "EUR" else "$"
        exchange_rate = 0.92 if st.session_state.currency == "EUR" else 1.0

        st.text_input("Current Dataset:", value="btcusd_1-min_data.csv", disabled=True)
            
        date_range = st.date_input("Date Range", [default_start, max_csv_date], min_value=min_csv_date, max_value=max_csv_date)
        if len(date_range) != 2:
            st.warning("Please select an end date.")
            st.stop()
            
        vol_window = st.slider("Volatility Smoothing Window", 5, 50, 20)
        
        st.markdown("---")
        st.subheader("📐 Math Parameters (FA-2)")
        sim_mode = st.selectbox("Mathematical Pattern", ["Combined mode", "Sine wave", "Cosine wave", "Random noise", "Drift (integral effect)"])
        amp = st.slider("Wave Amplitude", 1000, 20000, 5000)
        freq = st.slider("Wave Frequency", 0.5, 20.0, 5.0)
        drift = st.slider("Drift slope (Integral)", -100.0, 100.0, 10.0)
        noise = st.slider("Noise intensity", 500, 10000, 2000)

    # --- DATA FILTERING ---
    mask = (base_df['Date'].dt.date >= date_range[0]) & (base_df['Date'].dt.date <= date_range[1])
    raw_df = base_df.loc[mask].copy()
    
    if raw_df.empty:
        st.error("⚠️ Selected date range contains no data. Please select a different range.")
        st.stop()
        
    df = clean_data(raw_df)
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col] * exchange_rate
            
    df = calculate_indicators(df, window=vol_window)

    # --- METRICS UI ---
    latest_price = float(df["Price"].iloc[-1])
    latest_ret = float(df["Daily_Return"].iloc[-1]) * 100
    latest_vol = float(df["Rolling_Volatility"].iloc[-1]) * 100
    sharpe = float((df["Daily_Return"].mean() / df["Daily_Return"].std()) * np.sqrt(252))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Latest Price</div><div class="metric-value">{currency_sym}{latest_price:,.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Daily Return</div><div class="metric-value" style="color:{"#00ff00" if latest_ret>0 else "#ff0000"}">{latest_ret:.2f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Annualized Volatility</div><div class="metric-value">{latest_vol:.2f}%</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)

    col_ai1, col_ai2 = st.columns([1, 2])
    with col_ai2:
        vol_state = ai_analysis(df)
        st.progress(min(int(latest_vol), 100), text=f"Risk Meter: {latest_vol:.1f}%")
    with col_ai1:
        if latest_vol < 40: mascot_color = "Low"
        elif latest_vol < 70: mascot_color = "Medium"
        else: mascot_color = "High"
        render_3d_mascot(mascot_color)

    st.markdown("---")

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📐 FA-2 Math Simulation", "📊 Core Visualizations", "🧠 AI Quant Tools", "🤖 Gemini Chat"])
    
    with t1:
        st.header("FA-2 Rubric: Mathematical Functions & Price Swings")
        st.write("This section demonstrates how mathematical functions are used to synthesize wave-like price swings, sudden random noise jumps, and long-term integral drift against your actual local CSV data.")
        
        # Calculate simulation
        df["Simulated"] = simulate_patterns(df, sim_mode, amp, freq, drift, noise)
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Actual Market Price (Baseline)", line=dict(color='gray', width=1)))
        fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Simulated"], name=f"Mathematical Model ({sim_mode})", line=dict(color='#00ffcc', width=2)))
        fig_sim.update_layout(title="Real Data vs Synthesized Mathematical Pattern", template="plotly_dark")
        st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown("""<div class="math-card">
        <h3>🧮 How the Mathematics Work</h3>
        """, unsafe_allow_html=True)
        
        if sim_mode in ["Sine wave", "Combined mode"]:
            st.write("**1. Sine / Cosine (Cyclical Market Swings)**")
            st.latex(r"P(t) = P_0 + A \cdot \sin\left(2\pi f \frac{t}{N}\right)")
            st.write(f"Where Amplitude ($A$) = {amp} and Frequency ($f$) = {freq}.")
            
        if sim_mode in ["Random noise", "Combined mode"]:
            st.write("**2. Random Noise (Sudden Price Jumps)**")
            st.latex(r"P(t) = P_0 + \mathcal{N}(0, \sigma^2)")
            st.write(f"Where Noise Intensity ($\sigma$) = {noise}. This simulates unpredictable market shocks.")
            
        if sim_mode in ["Drift (integral effect)", "Combined mode"]:
            st.write("**3. Drift & Integrals (Long-Term Trends)**")
            st.latex(r"P(t) = P_0 + \int_{0}^{t} \text{drift} \, dt = P_0 + (\text{drift} \times t)")
            st.write(f"Where Drift = {drift}. This calculates the cumulative area under the curve representing long-term upward or downward momentum.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with t2:
        st.subheader("Advanced Price Charts")
        
        fig1 = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Price'])])
        fig1.update_layout(title=f"Candlestick Chart ({currency_sym})", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.plotly_chart(px.line(df, x="Date", y="Rolling_Volatility", title="Rolling Volatility (Annualized)"), use_container_width=True)
        
        df["Vol_Color"] = np.where(df["Rolling_Volatility"] > 0.6, "Volatile", "Stable")
        st.plotly_chart(px.scatter(df, x="Date", y="Price", color="Vol_Color", title="Stable vs Volatile Market Regions"), use_container_width=True)

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Price"))
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Upper Band", line=dict(dash='dot')))
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Lower Band", line=dict(dash='dot')))
        fig7.update_layout(title="Bollinger Bands (Volatility Ranges)")
        st.plotly_chart(fig7, use_container_width=True)
            
    with t3:
        if st.button("Run Monte Carlo Simulation", key="btn_mc"):
            returns = df["Daily_Return"].dropna()
            mean_return, std_return = returns.mean(), returns.std()
            last_price = float(df["Price"].iloc[-1])
            simulations = []
            for _ in range(100): 
                path = [last_price]
                for _ in range(30): 
                    path.append(path[-1] * (1 + np.random.normal(mean_return, std_return)))
                simulations.append(path)
            fig_mc = go.Figure()
            for sim in simulations:
                fig_mc.add_trace(go.Scatter(y=sim, mode='lines', line=dict(width=1, color='rgba(0, 255, 204, 0.05)')))
            fig_mc.update_layout(title="🔮 Monte Carlo Simulation (100 Paths, 30 Days)", showlegend=False, template="plotly_dark")
            st.plotly_chart(fig_mc, use_container_width=True)
            
        if st.button("Run Neural Network Forecast", key="btn_nn"):
            st.write("Training Simple Neural Network on Local Data...")
            data = df[["Price"]].values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            X, y = [], []
            for i in range(14, len(scaled_data)):
                X.append(scaled_data[i-14:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=300, random_state=42)
            model.fit(X, y)
            last_14 = scaled_data[-14:].reshape(1, -1)
            predictions = []
            for _ in range(7):
                next_pred = model.predict(last_14)[0]
                predictions.append(next_pred)
                last_14 = np.append(last_14[:, 1:], next_pred).reshape(1, -1)
            pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
            fig_nn = go.Figure()
            fig_nn.add_trace(go.Scatter(x=df["Date"].iloc[-30:], y=df["Price"].iloc[-30:], name="Recent Actual Price"))
            fig_nn.add_trace(go.Scatter(x=future_dates, y=pred_prices.flatten(), name="7-Day NN Forecast", line=dict(dash='dot', color='red')))
            fig_nn.update_layout(title="🤖 Neural Network 7-Day Forecast", template="plotly_dark")
            st.plotly_chart(fig_nn, use_container_width=True)

    with t4:
        st.subheader("🧠 Senior AI Quant Assistant")

        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                if not available_models:
                    st.error("API Error: Your API key is valid but doesn't have text generation access.")
                else:
                    target_model = available_models[0]
                    for m in available_models:
                        if 'flash' in m or 'pro' in m:
                            target_model = m
                            break
                            
                    model = genai.GenerativeModel(target_model)

                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if prompt := st.chat_input("Ask about Sine Waves, Integrals, or Volatility..."):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        context = f"""
                        You are a Senior Quantitative Analyst AI. 
                        CURRENT MARKET CONTEXT:
                        - Asset: BTC (Local CSV)
                        - Latest Price: {currency_sym}{latest_price:,.2f}
                        - Annualized Volatility: {latest_vol:.2f}%
                        """
                        full_prompt = f"{context}\n\nUser Question: {prompt}"

                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing..."):
                                response = model.generate_content(full_prompt)
                                st.markdown(response.text)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response.text})

            except Exception as e:
                st.error(f"API Connection Error. Details: {e}")
        else:
            st.warning("⚠️ Enter your API Key in the sidebar to chat with the AI.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>BTEC CRS AI-II FA-2 Project | Educational Purposes Only</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
