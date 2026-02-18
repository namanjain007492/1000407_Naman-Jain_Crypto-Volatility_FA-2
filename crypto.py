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
    """Fetches real data. If Yahoo API blocks the Cloud IP, it generates fallback data."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # Attempt 1: Fetch Real Data
        df = yf.download(symbol, start=start_str, end=end_str, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.reset_index(inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df, False  # False means "Real Data"
    except Exception:
        pass

    # Attempt 2: THE FALLBACK (If Yahoo blocks the IP, generate realistic market data)
    days = (end_date - start_date).days
    if days < 30: days = 365
    
    np.random.seed(42 if "BTC" in symbol else (43 if "ETH" in symbol else 44))
    dates = pd.date_range(start=start_str, periods=days, freq="D")
    base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 100)
    volatility = 0.03 if "BTC" in symbol else 0.04
    
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
    return df_fallback, True  # True means "Simulated Fallback Data"

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

def generate_pdf_report(df, vol_state, symbol):
    latest_price = float(df["Price"].iloc[-1])
    latest_vol = float(df["Rolling_Volatility"].iloc[-1])
    report_content = f"""
    CRYPTO VOLATILITY REPORT: {symbol}
    =========================================
    Date Generated: {datetime.today().strftime('%Y-%m-%d')}
    Dataset Shape: {df.shape}
    
    KEY METRICS:
    - Final Price: ${latest_price:,.2f}
    - Annualized Volatility: {latest_vol*100:.2f}%
    - Max Drawdown: {float(df["Drawdown"].min())*100:.2f}%
    
    AI CLASSIFICATION:
    The current market is exhibiting {vol_state} volatility. 
    
    Disclaimer: Generated for BTEC CRS AI-II. Not financial advice.
    """
    b64 = base64.b64encode(report_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{symbol}_Volatility_Report.txt" class="metric-card" style="color:#00ffcc; text-decoration:none;">📄 Download Auto-Generated Report (.txt)</a>'
    st.markdown(href, unsafe_allow_html=True)

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
            
        symbol = st.selectbox("Multi-Crypto Selector", crypto_options, index=default_index, key="crypto_selector_widget")
        if symbol != st.session_state.selected_crypto:
            st.session_state.selected_crypto = symbol
            st.rerun() 
            
        date_range = st.date_input("Date Range", [pd.to_datetime("2023-01-01"), datetime.today().date()], key="date_range_widget")
        if len(date_range) != 2:
            st.warning("Please select an end date.")
            st.stop()
            
        vol_window = st.slider("Volatility Smoothing Window", 5, 50, 20, key="vol_window_widget")
        
        st.markdown("---")
        st.subheader("📐 Math Simulation")
        sim_toggle = st.checkbox("Enable Simulation Mode", key="sim_toggle_widget")
        sim_mode = st.selectbox("Pattern", ["Sine wave", "Cosine wave", "Random noise", "Drift (integral effect)", "Combined mode"], key="sim_mode_widget")
        amp = st.slider("Amplitude", 1000, 20000, 5000, key="amp_widget")
        freq = st.slider("Frequency", 0.5, 20.0, 5.0, key="freq_widget")
        drift = st.slider("Drift slope", -100.0, 100.0, 10.0, key="drift_widget")
        noise = st.slider("Noise intensity", 500, 10000, 2000, key="noise_widget")

    # --- DATA PIPELINE ---
    raw_df, is_simulated = load_data(st.session_state.selected_crypto, date_range[0], date_range[1])
    
    if raw_df.empty:
        st.error("⚠️ Data pipeline failed entirely. Please refresh.")
        st.stop()
        
    if is_simulated:
        st.warning("📡 **Network Alert:** Yahoo Finance API blocked the cloud connection. Loaded high-fidelity simulated market data to maintain functionality.")
        
    df = clean_data(raw_df)
    df = calculate_indicators(df, window=vol_window)
    
    if df.empty:
        st.error("⚠️ Not enough data points to calculate indicators. Select a wider date range.")
        st.stop()

    # --- METRICS & AI UI ---
    with st.expander("📊 View Dataset Details (Stage 4 Requirements)"):
        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Data Source:** {'Real Market Data' if not is_simulated else 'Simulated Fallback Data'}")
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
        generate_pdf_report(df, vol_state, symbol)
    with col_ai1:
        if latest_vol < 40:
            mascot_color = "Low"
        elif latest_vol < 70:
            mascot_color = "Medium"
        else:
            mascot_color = "High"
        render_3d_mascot(mascot_color)

    st.markdown("---")

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 Core Visualizations", "📐 Simulation Mode", "🧠 AI Quant Tools", "🤖 Gemini Chat"])
    
    with t1:
        st.plotly_chart(px.line(df, x="Date", y="Price", title="1️⃣ Price vs Date"), use_container_width=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["High"], name="High", line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["Low"], name="Low", line=dict(color='red')))
        fig2.update_layout(title="2️⃣ High vs Low Comparison")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.plotly_chart(px.bar(df, x="Date", y="Volume", title="3️⃣ Trading Volume"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="Daily_Return", nbins=60, title="4️⃣ Histogram of Daily Returns"), use_container_width=True)
        st.plotly_chart(px.line(df, x="Date", y="Rolling_Volatility", title="5️⃣ Rolling Volatility (Annualized)"), use_container_width=True)
        
        df["Vol_Color"] = np.where(df["Rolling_Volatility"] > 0.6, "Volatile", "Stable")
        st.plotly_chart(px.scatter(df, x="Date", y="Price", color="Vol_Color", title="6️⃣ Stable vs Volatile Market Regions"), use_container_width=True)

        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Price"))
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="Upper Band", line=dict(dash='dot')))
        fig7.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="Lower Band", line=dict(dash='dot')))
        fig7.update_layout(title="7️⃣ Bollinger Bands")
        st.plotly_chart(fig7, use_container_width=True)
        
        fig8 = px.line(df, x="Date", y="RSI", title="8️⃣ 14-Day RSI")
        fig8.add_hline(y=70, line_dash="dash", line_color="red")
        fig8.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig8, use_container_width=True)
        
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
        fig9.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal"))
        fig9.update_layout(title="9️⃣ MACD Indicator")
        st.plotly_chart(fig9, use_container_width=True)
        
        st.plotly_chart(px.area(df, x="Date", y="Drawdown", title="🔟 Drawdown Chart"), use_container_width=True)
        
    with t2:
        if sim_toggle:
            df["Simulated"] = simulate_patterns(df, sim_mode, amp, freq, drift, noise)
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Real Price"))
            fig_sim.add_trace(go.Scatter(x=df["Date"], y=df["Simulated"], name=f"Simulated ({sim_mode})"))
            fig_sim.update_layout(title="Real vs Mathematical Simulated Price", template="plotly_dark")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("👈 Enable Simulation Mode in the sidebar to view mathematical models.")
            
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
            fig_nn = go.Figure()
            fig_nn.add_trace(go.Scatter(x=df["Date"].iloc[-30:], y=df["Price"].iloc[-30:], name="Recent Actual Price"))
            fig_nn.add_trace(go.Scatter(x=future_dates, y=pred_prices.flatten(), name="7-Day NN Forecast", line=dict(dash='dot', color='red')))
            fig_nn.update_layout(title="🤖 Neural Network 7-Day Forecast", template="plotly_dark")
            st.plotly_chart(fig_nn, use_container_width=True)
            
        if st.button("Run Multi-Crypto Optimizer", key="btn_opt"):
            st.write("Fetching multi-crypto correlation data (BTC, ETH, SOL)...")
            try:
                # Forcing string format here just in case yfinance gets stuck
                d1 = date_range[0].strftime('%Y-%m-%d')
                d2 = (date_range[1] + timedelta(days=1)).strftime('%Y-%m-%d')
                data = yf.download(["BTC-USD", "ETH-USD", "SOL-USD"], start=d1, end=d2, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    returns = data['Close'].pct_change().dropna()
                else:
                    returns = data.pct_change().dropna()
                corr = returns.corr()
                fig_corr = px.imshow(corr, text_auto=True, title="💼 Multi-Crypto Correlation Matrix", color_continuous_scale="Viridis")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error("⚠️ Multi-Crypto correlation failed due to Yahoo Finance rate limits.")
            
        st.markdown("---")
        st.subheader("⏯ Animated Price Replay")
        if st.button("▶️ Start Live Replay", key="btn_replay"):
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
