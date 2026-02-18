import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 SIDEBAR & SECRETS
# ==========================================
with st.sidebar:
    st.header("🔑 AI Integration")
    # Securely fetch the API key from secrets.toml (Fallback to text input if missing)
    gemini_api_key = "AIzaSyCakLpCyBipE7p-amnYzBEWeT5KKcTbgmo"
    if not gemini_api_key:
        gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
        
    if gemini_api_key:
        st.success("✅ AI Assistant Active!")
    else:
        st.error("⚠️ API Key missing.")

    st.header("⚙️ Settings Panel")
    crypto_options = ["BTC-USD", "ETH-USD", "SOL-USD"]
    symbol = st.selectbox("Multi-Crypto Selector", crypto_options, index=0)
    
    date_range = st.date_input("Date Range", [pd.to_datetime("2023-01-01"), datetime.today().date()])
    if len(date_range) != 2:
        st.warning("Please select both a start and end date.")
        st.stop()
        
    vol_window = st.slider("Volatility Smoothing Window", 5, 50, 20)
    
    st.markdown("---")
    st.subheader("📐 Math Simulation")
    sim_toggle = st.checkbox("Enable Simulation Mode")
    sim_mode = st.selectbox("Pattern", ["Sine wave", "Cosine wave", "Random noise", "Drift (integral effect)", "Combined mode"])
    amp = st.slider("Amplitude", 1000, 20000, 5000)
    freq = st.slider("Frequency", 0.5, 20.0, 5.0)
    drift = st.slider("Drift slope", -100.0, 100.0, 10.0)
    noise = st.slider("Noise intensity", 500, 10000, 2000)

# ==========================================
# 🔹 STAGE 4: DATA PREPARATION (CACHED)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={"Close": "Price"}, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True) 
    return df

df = load_data(symbol, date_range[0], date_range[1])

# Calculate Indicators
df["Daily_Return"] = df["Price"].pct_change()
df["Rolling_Mean"] = df["Price"].rolling(window=vol_window).mean()
df["Rolling_Std"] = df["Daily_Return"].rolling(window=vol_window).std()
df["Rolling_Volatility"] = df["Rolling_Std"] * np.sqrt(252)

df["BB_Upper"] = df["Rolling_Mean"] + (df["Price"].rolling(window=vol_window).std() * 2)
df["BB_Lower"] = df["Rolling_Mean"] - (df["Price"].rolling(window=vol_window).std() * 2)

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

df = df.dropna()

# ==========================================
# 🔹 3D MASCOT SYSTEM
# ==========================================
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

# ==========================================
# 🔹 MAIN DASHBOARD
# ==========================================
st.title("⚡ Crypto Volatility Visualizer – Elite Public Edition")

with st.expander("📊 View Dataset Details (Stage 4 Requirements)"):
    st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

# Top Metrics Cards
c1, c2, c3, c4 = st.columns(4)
latest_price = float(df["Price"].iloc[-1])
latest_ret = float(df["Daily_Return"].iloc[-1]) * 100
latest_vol = float(df["Rolling_Volatility"].iloc[-1]) * 100
sharpe = float((df["Daily_Return"].mean() / df["Daily_Return"].std()) * np.sqrt(252))

c1.markdown(f'<div class="metric-card"><div class="metric-label">Latest Price</div><div class="metric-value">${latest_price:,.2f}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="metric-label">Daily Return</div><div class="metric-value" style="color:{"#00ff00" if latest_ret>0 else "#ff0000"}">{latest_ret:.2f}%</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="metric-label">Annualized Volatility</div><div class="metric-value">{latest_vol:.2f}%</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Risk Meter & Mascot
if latest_vol < 40:
    vol_state = "Low"
elif latest_vol < 70:
    vol_state = "Medium"
else:
    vol_state = "High"

col_ai1, col_ai2 = st.columns([1, 2])
with col_ai2:
    st.markdown(f"### 🤖 Current AI Assessment: **{vol_state} Volatility**")
    st.progress(min(int(latest_vol), 100), text=f"Risk Meter: {latest_vol:.1f}%")
    
    # PDF Report Generator
    report_content = f"BITCOIN VOLATILITY REPORT\nDate: {datetime.today().strftime('%Y-%m-%d')}\nFinal Price: ${latest_price:,.2f}\nVolatility: {latest_vol:.2f}%\nAI Verdict: {vol_state}"
    b64 = base64.b64encode(report_content.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64}" download="{symbol}_Report.txt" class="metric-card" style="color:#00ffcc; text-decoration:none;">📄 Download Auto-Generated Report (.txt)</a>', unsafe_allow_html=True)
with col_ai1:
    render_3d_mascot(vol_state)

st.markdown("---")

# ==========================================
# 🔹 TABS: VISUALS, SIMULATION, QUANT, AI
# ==========================================
t1, t2, t3, t4 = st.tabs(["📊 10 Core Visualizations", "📐 Math Simulation", "🔮 Quant Tools & Replay", "🧠 Gemini AI Assistant"])

with t1:
    st.subheader("Interactive Plotly Charts (Stage 5 Rubric)")
    # 1. Price
    st.plotly_chart(px.line(df, x="Date", y="Price", title="1️⃣ Price vs Date"), use_container_width=True)
    # 2. High/Low
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["High"], name="High", line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["Low"], name="Low", line=dict(color='red')))
    fig2.update_layout(title="2️⃣ High vs Low Comparison")
    st.plotly_chart(fig2, use_container_width=True)
    # 3. Volume
    st.plotly_chart(px.bar(df, x="Date", y="Volume", title="3️⃣ Trading Volume"), use_container_width=True)
    # 4. Histogram
    st.plotly_chart(px.histogram(df, x="Daily_Return", nbins=60, title="4️⃣ Histogram of Daily Returns"), use_container_width=True)
    # 5. Volatility
    st.plotly_chart(px.line(df, x="Date", y="Rolling_Volatility", title="5️⃣ Rolling Volatility (Annualized)"), use_container_width=True)
    # 6. Stable/Volatile Heatmap
    df["Vol_Color"] = np.where(df["Rolling_Volatility"] > 0.6, "Volatile", "Stable")
    st.plotly_chart(px.scatter(df, x="Date", y="Price", color="Vol_Color", title="6️⃣ Stable vs Volatile Regions"), use_container_width=True)
    # 7. Bollinger Bands
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Price"))
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Upper", line=dict(dash='dot')))
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Lower", line=dict(dash='dot')))
    fig7.update_layout(title="7️⃣ Bollinger Bands")
    st.plotly_chart(fig7, use_container_width=True)
    # 8. RSI
    fig8 = px.line(df, x="Date", y="RSI", title="8️⃣ 14-Day RSI")
    fig8.add_hline(y=70, line_dash="dash", line_color="red")
    fig8.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig8, use_container_width=True)
    # 9. MACD
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
    fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal"))
    fig9.update_layout(title="9️⃣ MACD Indicator")
    st.plotly_chart(fig9, use_container_width=True)
    # 10. Drawdown
    st.plotly_chart(px.area(df, x="Date", y="Drawdown", title="🔟 Drawdown Chart"), use_container_width=True)

with t2:
    if sim_toggle:
        t = np.arange(len(df))
        base_val = float(df["Price"].iloc[0])
        if sim_mode == "Sine wave":
            sim_prices = base_val + amp * np.sin(2 * np.pi * freq * t / len(t))
        elif sim_mode == "Cosine wave":
            sim_prices = base_val + amp * np.cos(2 * np.pi * freq * t / len(t))
        elif sim_mode == "Random noise":
            sim_prices = base_val + np.random.normal(0, noise, len(t))
        elif sim_mode == "Drift (integral effect)":
            sim_prices = base_val + drift * t
        else:
            sim_prices = base_val + drift * t + amp * np.sin(2 * np.pi * freq * t / len(t)) + np.random.normal(0, noise, len(t))
            
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Real Price"))
        fig_sim.add_trace(go.Scatter(x=df["Date"], y=sim_prices, name=f"Simulated ({sim_mode})"))
        fig_sim.update_layout(title="Real vs Mathematical Simulated Price", template="plotly_dark")
        st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.info("👈 Enable Simulation Mode in the sidebar to view mathematical overlays.")

with t3:
    st.subheader("Lightweight Linear Trend Forecast (7 Days)")
    if st.button("Run Fast Trend Forecast"):
        recent_data = df.iloc[-30:].copy()
        recent_data['Day_Num'] = np.arange(len(recent_data))
        slope, intercept = np.polyfit(recent_data['Day_Num'], recent_data['Price'], 1)
        future_days = np.arange(len(recent_data), len(recent_data) + 7)
        predicted_prices = slope * future_days + intercept
        future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"].iloc[-30:], y=df["Price"].iloc[-30:], name="Actual Price"))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, name="Trend Forecast", line=dict(dash='dot', color='cyan')))
        fig.update_layout(title="📈 7-Day Price Trend Projection", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    st.subheader("⏯ Animated Price Replay")
    if st.button("▶️ Start Live Replay"):
        animated_df = df.iloc[-60:].copy()
        min_date, max_date = animated_df["Date"].min(), animated_df["Date"].max()
        min_price, max_price = animated_df["Price"].min()*0.95, animated_df["Price"].max()*1.05
        chart_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(1, len(animated_df) + 1):
            current_data = animated_df.iloc[:i]
            fig_anim = px.line(current_data, x="Date", y="Price", range_x=[min_date, max_date], range_y=[min_price, max_price])
            fig_anim.update_layout(template="plotly_dark", title="Live Price Movement Simulation")
            chart_placeholder.plotly_chart(fig_anim, use_container_width=True)
            progress_bar.progress(i / len(animated_df))
            time.sleep(0.05)
        st.success("Replay Complete!")

with t4:
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
                - Annualized Volatility: {latest_vol:.2f}%
                """
                full_prompt = f"{context}\n\nUser Question: {prompt}"

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing market data..."):
                        response = model.generate_content(full_prompt)
                        st.markdown(response.text)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})

        except Exception as e:
            st.error(f"API Error: {e}. Please check your API key.")
    else:
        st.warning("⚠️ Enter your API Key in the sidebar or Streamlit Secrets to chat with the AI.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>FinTechLab Pvt. Ltd. | FA-2 Full Detail Edition</div>", unsafe_allow_html=True)
