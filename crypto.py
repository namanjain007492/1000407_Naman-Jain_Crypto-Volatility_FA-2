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
import google.generativeai as genai

# ==========================================
# PAGE CONFIGURATION & UI DESIGN
# ==========================================
st.set_page_config(page_title="Bitcoin Volatility Visualizer", page_icon="₿", layout="wide")

# Modern, clean, slightly oversized aesthetic for FinTech UI
st.markdown("""
    <style>
    .metric-card { background-color: #1e1e24; padding: 24px; border-radius: 12px; text-align: center; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-value { font-size: 28px; font-weight: bold; color: #00ffcc; margin-top: 10px; }
    .metric-label { font-size: 16px; color: #b0b0b0; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 STAGE 4: BULLETPROOF DATA PREPARATION
# ==========================================
@st.cache_data(ttl=3600)
def load_data(symbol="BTC-USD", start_date="2023-01-01", end_date=datetime.today().strftime('%Y-%m-%d')):
    """Fetches real Bitcoin dataset using yfinance."""
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Catch empty dataframes from Yahoo Finance
    if df.empty:
        return pd.DataFrame()
        
    # Handle yfinance multi-index columns safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
        
    df.reset_index(inplace=True)
    return df

def clean_data(df):
    """Cleans and formats data safely."""
    if df.empty:
        return df
        
    df["Date"] = pd.to_datetime(df["Date"])
    
    if "Close" in df.columns:
        df.rename(columns={"Close": "Price"}, inplace=True)
    
    df.ffill(inplace=True)
    if "Price" in df.columns:
        df.dropna(subset=["Price"], inplace=True) 
    return df

def calculate_indicators(df, window=20):
    """Calculates all required rolling metrics and technical indicators."""
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

# ==========================================
# 🔹 MATHEMATICAL SIMULATION
# ==========================================
def simulate_patterns(df, mode, amp, freq, drift, noise_int):
    """Generates mathematical price simulations."""
    t = np.arange(len(df))
    base = df["Price"].iloc[0].item() if hasattr(df["Price"].iloc[0], 'item') else float(df["Price"].iloc[0])
    
    if mode == "Sine wave":
        sim = base + amp * np.sin(2 * np.pi * freq * (t / len(t)))
    elif mode == "Cosine wave":
        sim = base + amp * np.cos(2 * np.pi * freq * (t / len(t)))
    elif mode == "Random noise":
        sim = base + np.random.normal(0, noise_int, len(t))
    elif mode == "Drift (integral effect)":
        sim = base + drift * t 
    else: 
        sim = base + drift * t + amp * np.sin(2 * np.pi * freq * (t / len(t))) + np.random.normal(0, noise_int, len(t))
        
    return sim

# ==========================================
# 🔹 3D MASCOT SYSTEM & AI ANALYSIS
# ==========================================
def render_3d_mascot(volatility_state):
    """Renders a 3D Three.js rotating object."""
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
        explanation = "The current volatility is Low. Prices are relatively stable. Sharpe ratio indicates steady risk-adjusted returns."
    elif latest_vol < 0.7:
        state, color = "Medium", "orange"
        explanation = "The current volatility is Medium. Normal market fluctuations are occurring. Watch RSI for overbought/oversold conditions."
    else:
        state, color = "High", "red"
        explanation = "The current volatility is High Risk. Expect large price swings. Capital preservation is strongly advised."
        
    st.markdown(f"### 🤖 AI Assessment: **:{color}[{state} Volatility]**")
    st.info(f"**Analysis:** {explanation}")
    st.caption("*Educational Suggestion: High volatility increases both potential reward and risk (standard deviation).*")
    return state

# ==========================================
# 🔹 ADVANCED FEATURES
# ==========================================
def monte_carlo_simulation(df):
    returns = df["Daily_Return"].dropna()
    mean_return, std_return = returns.mean(), returns.std()
    last_price = float(df["Price"].iloc[-1])
    
    simulations = []
    for _ in range(100): 
        path = [last_price]
        for _ in range(30): 
            path.append(path[-1] * (1 + np.random.normal(mean_return, std_return)))
        simulations.append(path)
        
    fig = go.Figure()
    for sim in simulations:
        fig.add_trace(go.Scatter(y=sim, mode='lines', line=dict(width=1, color='rgba(0, 255, 204, 0.05)')))
    fig.update_layout(title="🔮 Monte Carlo Simulation (100 Paths, 30 Days)", showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def neural_network_prediction(df):
    st.write("Training Simple Neural Network (LSTM Alternative)...")
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"].iloc[-30:], y=df["Price"].iloc[-30:].values.flatten(), name="Recent Actual Price"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_prices.flatten(), name="7-Day NN Forecast", line=dict(dash='dot', color='red')))
    fig.update_layout(title="🤖 Neural Network 7-Day Forecast", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def generate_pdf_report(df, vol_state):
    report_content = f"""
    BITCOIN VOLATILITY REPORT - ELITE EDITION
    =========================================
    Date Generated: {datetime.today().strftime('%Y-%m-%d')}
    Dataset Shape: {df.shape}
    
    KEY METRICS:
    - Final Price: ${float(df["Price"].iloc[-1]):,.2f}
    - Annualized Volatility: {float(df["Rolling_Volatility"].iloc[-1])*100:.2f}%
    - Max Drawdown: {float(df["Drawdown"].min())*100:.2f}%
    
    AI CLASSIFICATION:
    The current market is exhibiting {vol_state} volatility. 
    
    Disclaimer: Generated for BTEC CRS AI-II. Not financial advice.
    """
    b64 = base64.b64encode(report_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="BTC_Volatility_Report.txt" class="metric-card" style="color:#00ffcc; text-decoration:none;">📄 Download Auto-Generated Report (.txt)</a>'
    st.markdown(href, unsafe_allow_html=True)

def portfolio_optimizer():
    st.write("Fetching multi-crypto correlation data (BTC, ETH, SOL)...")
    try:
        data = yf.download(["BTC-USD", "ETH-USD", "SOL-USD"], period="6mo", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            returns = data['Close'].pct_change().dropna()
        else:
            returns = data.pct_change().dropna()
            
        corr = returns.corr()
        fig = px.imshow(corr, text_auto=True, title="💼 Multi-Crypto Correlation Matrix", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load multi-crypto data. Error: {str(e)}")

def build_visualizations(df):
    fig1 = px.line(df, x="Date", y="Price", title="1️⃣ Bitcoin Price vs Date")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["High"].values.flatten(), name="High", line=dict(color='rgba(0,255,0,0.6)')))
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["Low"].values.flatten(), name="Low", line=dict(color='rgba(255,0,0,0.6)')))
    fig2.update_layout(title="2️⃣ High vs Low Comparison")
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.bar(df, x="Date", y="Volume", title="3️⃣ Trading Volume")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df, x="Daily_Return", nbins=60, title="4️⃣ Histogram of Daily Returns")
    st.plotly_chart(fig4, use_container_width=True)
    
    fig5 = px.line(df, x="Date", y="Rolling_Volatility", title="5️⃣ Rolling Volatility (Annualized)")
    st.plotly_chart(fig5, use_container_width=True)
    
    df["Vol_Color"] = np.where(df["Rolling_Volatility"] > 0.6, "Volatile", "Stable")
    fig6 = px.scatter(df, x="Date", y="Price", color="Vol_Color", title="6️⃣ Stable vs Volatile Market Regions")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["Price"].values.flatten(), name="Price"))
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"].values.flatten(), name="Upper Band", line=dict(dash='dot')))
    fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"].values.flatten(), name="Lower Band", line=dict(dash='dot')))
    fig7.update_layout(title="7️⃣ Bollinger Bands")
    st.plotly_chart(fig7, use_container_width=True)
    
    fig8 = px.line(df, x="Date", y="RSI", title="8️⃣ 14-Day RSI")
    fig8.add_hline(y=70, line_dash="dash", line_color="red")
    fig8.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig8, use_container_width=True)
    
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD"].values.flatten(), name="MACD"))
    fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"].values.flatten(), name="Signal"))
    fig9.update_layout(title="9️⃣ MACD Indicator")
    st.plotly_chart(fig9, use_container_width=True)
    
    fig10 = px.area(df, x="Date", y="Drawdown", title="🔟 Drawdown Chart")
    st.plotly_chart(fig10, use_container_width=True)


# ==========================================
# 🔹 MAIN DASHBOARD
# ==========================================
def main():
    st.title("⚡ Crypto Volatility Visualizer – Elite Public Edition")
    
    if "selected_crypto" not in st.session_state:
        st.session_state.selected_crypto = "BTC-USD"
    
    with st.sidebar:
        st.header("🔑 AI Assistant API")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not gemini_api_key:
            gemini_api_key = st.text_input("Enter Gemini API Key", type="password")
            
        if gemini_api_key:
            st.success("✅ AI Ready")
        else:
            st.error("⚠️ AI Key missing")

        st.header("⚙️ Settings Panel")
        st.markdown("**🎤 Voice Control**")
        st.caption("Simulator Mode (No Mic Required)")
        if st.button("🎙️ Simulate Voice: 'Switch to ETH'"):
            st.session_state.selected_crypto = "ETH-USD"
            st.toast("✅ Voice Command Recognized: Switching to Ethereum...")
            st.rerun()
            
        crypto_options = ["BTC-USD", "ETH-USD", "SOL-USD"]
        try:
            default_index = crypto_options.index(st.session_state.selected_crypto)
        except ValueError:
            default_index = 0
            
        symbol = st.selectbox("Multi-Crypto Selector", crypto_options, index=default_index)
        if symbol != st.session_state.selected_crypto:
            st.session_state.selected_crypto = symbol
            st.rerun() 
            
        date_range = st.date_input("Date Range", [pd.to_datetime("2023-01-01"), datetime.today().date()])
        if len(date_range) != 2:
            st.warning("Please select an end date.")
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

    # 🔹 LOAD DATA
    raw_df = load_data(st.session_state.selected_crypto, date_range[0], date_range[1])
    if raw_df.empty or "Close" not in raw_df.columns:
        st.error("⚠️ No price data found for selected dates. Try a wider range.")
        st.stop()
        
    df = clean_data(raw_df)
    if df.empty or "Price" not in df.columns:
        st.error("⚠️ Data cleaning failed. Try selecting different dates.")
        st.stop()
        
    df = calculate_indicators(df, window=vol_window)
    if df.empty:
        st.error("⚠️ Not enough data points to calculate indicators. Select a wider date range.")
        st.stop()

    with st.expander("📊 View Dataset Details (Stage 4 Requirements)"):
        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
        st.markdown("Missing values handled using `ffill()` to prevent look-ahead bias.")

    latest_price = float(df["Price"].iloc[-1])
    latest_ret = float(df["Daily_Return"].iloc[-1]) * 100
    latest_vol = float(df["Rolling_Volatility"].iloc[-1]) * 100
    sharpe = float((df["Daily_Return"].mean() / df["Daily_Return"].std()) * np.sqrt(252))
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Latest Price</div><div class="metric-value">${latest_price:,.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Daily Return</div><div class="metric-value" style="color:{"#00ff00" if latest_ret>0 else "#ff0000"}">{latest_ret:.2f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Annualized Volatility</div><div class="metric-value">{latest_vol:.2f}%</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_ai1, col_ai2 = st.columns([1, 2])
    with col_ai2:
        vol_state = ai_analysis(df)
        st.progress(min(int(latest_vol), 100), text=f"Risk Meter: {latest_vol:.1f}%")
        generate_pdf_report(df, vol_state)
    with col_ai1:
        render_3d_mascot(vol_state)

    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["📊 Core Visualizations", "📐 Simulation Mode", "🧠 AI Quant Tools", "🤖 Gemini Chat"])
    
    with t1:
        build_visualizations(df)
        
    with t2:
        if sim_toggle:
            df["Simulated"] = simulate_patterns(df, sim_mode, amp, freq, drift, noise)
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Price"].values.flatten(), name="Real Price"))
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Simulated"], name=f"Simulated ({sim_mode})"))
            fig_sim.update_layout(title="Real vs Mathematical Simulated Price", template="plotly_dark")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("👈 Enable Simulation Mode in the sidebar to view mathematical models.")
            
    with t3:
        if st.button("Run Monte Carlo Simulation"):
            monte_carlo_simulation(df)
        if st.button("Run Neural Network Forecast"):
            neural_network_prediction(df)
        if st.button("Run Multi-Crypto Optimizer"):
            portfolio_optimizer()
            
        st.markdown("---")
        st.subheader("⏯ Animated Price Replay")
        animated_df = df.iloc[-60:].copy()
        min_date, max_date = animated_df["Date"].min(), animated_df["Date"].max()
        min_price, max_price = animated_df["Price"].min()*0.95, animated_df["Price"].max()*1.05
        chart_placeholder = st.empty()
        
        if st.button("▶️ Start Live Replay"):
            progress_bar = st.progress(0)
            for i in range(1, len(animated_df) + 1):
                current_data = animated_df.iloc[:i]
                fig_anim = px.line(current_data, x="Date", y="Price", range_x=[min_date, max_date], range_y=[min_price, max_price])
                fig_anim.update_layout(template="plotly_dark", title="Live Price Movement Simulation")
                chart_placeholder.plotly_chart(fig_anim, use_container_width=True)
                progress_bar.progress(i / len(animated_df))
                time.sleep(0.05)
            st.success("Replay Complete!")
        else:
            fig_anim = px.line(animated_df, x="Date", y="Price", range_x=[min_date, max_date], range_y=[min_price, max_price])
            fig_anim.update_layout(template="plotly_dark", title="Ready for Replay (Click Start)")
            chart_placeholder.plotly_chart(fig_anim, use_container_width=True)

    with t4:
        st.subheader("🧠 Senior AI Quant Assistant")
        st.write(f"Ask the AI about {symbol} volatility, formulas, or investment risks.")

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

                    # Secretly inject live dashboard data into the prompt
                    context = f"""
                    You are a Senior Quantitative Analyst AI. Answer concisely.
                    CURRENT MARKET CONTEXT:
                    - Asset: {st.session_state.selected_crypto}
                    - Latest Price: ${latest_price:,.2f}
                    - Daily Return: {latest_ret:.2f}%
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
    st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>FinTechLab Pvt. Ltd. | BTEC CRS AI-II FA-2 Project | Educational Purposes Only</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
