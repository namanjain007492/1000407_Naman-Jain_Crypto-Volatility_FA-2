⚡ Crypto Volatility Visualizer – Elite Public Edition

An advanced AI-powered cryptocurrency analytics dashboard built using Streamlit, Machine Learning, Mathematical Modeling, and Quantitative Finance concepts.

This project analyzes real Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) market data, calculates volatility using statistical formulas, simulates price behavior using mathematical functions, and integrates AI-based forecasting tools.

Developed for:
BTEC CRS – Mathematics for AI-II (FA-2 Project)

🚀 Application Overview

This application combines:

📊 Real financial time-series analysis

📐 Mathematical volatility simulation

🤖 AI-based market risk classification

🔮 Monte Carlo forecasting

🧠 Neural network price prediction

🎤 Voice-controlled crypto selection

🎨 Interactive professional fintech UI

🧑‍🚀 3D animated volatility mascot

The goal is to demonstrate applied mathematics in AI-driven financial modeling.

📊 Core Features
1️⃣ Real-Time Cryptocurrency Data

Data Source: Yahoo Finance (via yfinance)

Supported assets:

BTC-USD

ETH-USD

SOL-USD

Dataset contains:

Date

Open

High

Low

Close (renamed to Price)

Volume

Data preprocessing includes:

Date conversion to datetime

Forward filling missing values

Removing initial NaNs

Rolling window smoothing

2️⃣ Mathematical Calculations
Daily Return Formula
𝑅
𝑒
𝑡
𝑢
𝑟
𝑛
=
𝑃
𝑡
−
𝑃
𝑡
−
1
𝑃
𝑡
−
1
Return=
P
t−1
	​

P
t
	​

−P
t−1
	​

	​

Rolling Standard Deviation

Measures short-term volatility.

Annualized Volatility Formula
𝑉
𝑜
𝑙
𝑎
𝑡
𝑖
𝑙
𝑖
𝑡
𝑦
=
𝜎
×
252
Volatility=σ×
252
	​


Where:

σ = standard deviation of daily returns

252 = trading days in a year

Additional Calculations

Rolling Mean

Cumulative Returns

Bollinger Bands

RSI (Relative Strength Index)

MACD Indicator

Drawdown

Sharpe Ratio

Sharpe Ratio Formula:

𝑆
ℎ
𝑎
𝑟
𝑝
𝑒
=
𝑀
𝑒
𝑎
𝑛
(
𝑅
𝑒
𝑡
𝑢
𝑟
𝑛
)
𝑆
𝑡
𝑑
(
𝑅
𝑒
𝑡
𝑢
𝑟
𝑛
)
×
252
Sharpe=
Std(Return)
Mean(Return)
	​

×
252
	​

📈 Interactive Visualizations (10 Required Charts)

The dashboard includes:

Price vs Date

High vs Low Comparison

Volume Bar Chart

Histogram of Returns

Rolling Volatility Chart

Stable vs Volatile Scatter Plot

Bollinger Bands

RSI Indicator

MACD Indicator

Drawdown Area Chart

All graphs:

Interactive (Plotly)

Hover tooltips enabled

Zoomable

Professionally styled (Dark fintech theme)

📐 Mathematical Simulation Mode

Users can simulate synthetic price behavior using:

Simulation formula:

𝑆
𝑖
𝑚
𝑢
𝑙
𝑎
𝑡
𝑒
𝑑
=
𝐵
𝑎
𝑠
𝑒
+
𝐴
sin
⁡
(
𝑓
𝑡
)
+
𝐷
𝑟
𝑖
𝑓
𝑡
⋅
𝑡
+
𝑁
𝑜
𝑖
𝑠
𝑒
Simulated=Base+Asin(ft)+Drift⋅t+Noise

Where:

A = Amplitude

f = Frequency

Drift = Linear trend

Noise = Random Gaussian variation

Simulation Modes:

Sine Wave

Cosine Wave

Random Noise

Drift (Integral Effect)

Combined Mode

Users can compare:
Real Price vs Simulated Price (side-by-side)

🤖 AI & Quantitative Tools
🔮 Monte Carlo Simulation

100 simulated future price paths

30-day projection

Uses Geometric Brownian Motion

🧠 Neural Network Forecast

Scikit-Learn MLPRegressor

Uses previous 14 days as input

Predicts next 7 days

MinMax scaling applied

💼 Multi-Crypto Correlation Matrix

BTC, ETH, SOL correlation heatmap

Portfolio diversification insight

🎤 Voice Control System

Using streamlit-mic-recorder

Users can say:

“Bitcoin”

“Ethereum”

“Solana”

The app automatically switches cryptocurrency.

🧑‍🚀 3D Volatility Mascot

Built using Three.js embedded in Streamlit.

Features:

Rotating 3D object

Color changes based on volatility:

Green → Low Risk
Yellow → Medium Risk
Red → High Risk

Displays AI market classification message.

📄 Auto-Generated Analytical Report

Users can download:

Text-based volatility summary

Final price

Annualized volatility

Max drawdown

AI classification

Generated dynamically using base64 encoding.

⏯ Animated Price Replay

Replays last 60 days of price movement

Frame-by-frame animation

Progress bar indicator

Fixed axes for smooth playback

🖥️ User Interface Design

Wide layout

Card-based metrics

Dark fintech theme

Sidebar control panel

Tabbed dashboard structure

Mobile responsive

🎓 Educational Objectives

This project demonstrates:

Time-series analysis

Volatility modeling

Statistical risk measurement

Financial mathematics

Simulation modeling

AI forecasting

Machine learning regression

Portfolio diversification analysis

⚠ Disclaimer

This application is developed for educational purposes under BTEC CRS Mathematics for AI-II.

It does not provide financial advice.

👨‍💻 Developer

Naman Jain
BTEC CRS – Mathematics for AI-II
FinTechLab Project Theme

🧠 Suggested Viva Questions

What is volatility and why is it annualized?

Why do we multiply by √252?

What does Sharpe ratio measure?

What is the purpose of Monte Carlo simulation?

How does the neural network predict prices?

What is look-ahead bias?

Why use rolling windows?

What does drawdown represent?

How does RSI indicate overbought conditions?

What is drift in financial modeling?
