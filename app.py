import streamlit as st
import torch
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import CryptoTransformer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Crypto Market Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- HIGH-CONTRAST CLEAN WHITE & SKY-BLUE UI ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e0f2fe 50%, #eef2ff 100%);
        color: #0f172a !important;
    }
    /* Force dark high-contrast text on all Streamlit labels, headings & metrics */
    label, p, span, h1, h2, h3, .stMarkdown {
        color: #0f172a !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 8px 20px -8px rgba(15, 23, 42, 0.12);
    }
    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    [data-testid="stMetricValue"] div {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 26px !important;
    }
    /* Input & Selectbox styling */
    input, [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .main-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 26px;
        box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.1);
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        background: #0284c7;
        color: #ffffff !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-radius: 20px;
        margin-bottom: 10px;
    }
    .title-text {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .subtitle-text {
        color: #475569 !important;
        font-size: 14px;
        margin-top: 4px;
        font-weight: 500;
    }
    .bullish-pill {
        background: #dcfce7;
        color: #15803d !important;
        border: 1.5px solid #86efac;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 13px;
        display: inline-block;
    }
    .bearish-pill {
        background: #fee2e2;
        color: #b91c1c !important;
        border: 1.5px solid #fca5a5;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 13px;
        display: inline-block;
    }
    div.stButton > button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        border: none;
        width: 100%;
    }
    div.stButton > button p {
        color: #ffffff !important;
    }
    div.stButton > button:hover {
        background-color: #0284c7 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD TRAINED PYTORCH TRANSFORMER ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = CryptoTransformer(feature_size=1, d_model=64, nhead=4, num_layers=2)
    model.load_state_dict(torch.load("crypto_transformer.pth", map_location=device, weights_only=True))
    model.eval()
    return model

# --- HEADER ---
st.markdown("""
<div class="main-card" style="text-align: center;">
    <span class="badge">Transformer Architecture</span>
    <h1 class="title-text">Market Predictor</h1>
    <p class="subtitle-text">PyTorch Self-Attention Time-Series Forecasting Engine</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT CONTROLS ---
col_input, col_quick = st.columns([2, 1])
with col_input:
    ticker = st.text_input("Asset Ticker (Yahoo Finance)", value="BTC-USD", placeholder="e.g. BTC-USD, ETH-USD, SOL-USD").strip().upper()
with col_quick:
    preset = st.selectbox("Quick Select", ["Custom", "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"])
    if preset != "Custom":
        ticker = preset

predict_clicked = st.button("Generate Prediction")

# --- PREDICTION PIPELINE ---
if predict_clicked:
    if not ticker:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner(f"Fetching 60-day live market window for {ticker}..."):
            try:
                model = load_model()
                df = yf.download(ticker, period="60d", interval="1d", progress=False)

                if df.empty or len(df) < 10:
                    st.error(f"No sufficient market data found for '{ticker}'. Check the ticker symbol.")
                else:
                    close_series = df["Close"].squeeze()
                    prices = close_series.values.reshape(-1, 1)

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_prices = scaler.fit_transform(prices)

                    X_input = torch.tensor(scaled_prices, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        scaled_pred = model(X_input).item()

                    predicted_price = float(scaler.inverse_transform([[scaled_pred]])[0][0])
                    current_price = float(prices[-1][0])
                    delta_val = predicted_price - current_price
                    delta_pct = (delta_val / current_price) * 100

                    is_bullish = predicted_price > current_price
                    trend_html = (
                        '<span class="bullish-pill">BULLISH &uarr;</span>'
                        if is_bullish
                        else '<span class="bearish-pill">BEARISH &darr;</span>'
                    )

                    # --- METRICS DISPLAY ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Asset", ticker)
                    m2.metric("Current Price", f"${current_price:,.2f}")
                    m3.metric("Next-Day Forecast", f"${predicted_price:,.2f}", f"{delta_pct:+.2f}%")

                    st.markdown(f"""
                    <div class="main-card" style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px;">
                        <span style="font-weight: 700; color: #0f172a; font-size: 15px;">Expected Market Direction</span>
                        {trend_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # --- 60-DAY PRICE HISTORY CHART ---
                    st.subheader(f"{ticker} — 60-Day Close Price Trajectory")
                    chart_df = pd.DataFrame({
                        "Close Price (USD)": close_series.values
                    }, index=df.index)
                    st.line_chart(chart_df, color="#0284c7")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")