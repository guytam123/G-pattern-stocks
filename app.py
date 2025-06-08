
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="זיהוי מניות עם תבנית W", layout="wide")
st.title("📈 מניות עם תבנית W – ניתוח טכני")

# הגדרות משתמש
st.sidebar.header("🎯 קריטריונים")
market_cap = st.sidebar.selectbox("גודל שוק", ["Large Cap", "Mid Cap", "Small Cap"])
price_range = st.sidebar.slider("טווח מחיר למניה ($)", 1, 500, (5, 100))
interval = st.sidebar.radio("טיימפריים", ["1d", "1wk"], index=1)
tickers_input = st.sidebar.text_area("מניות לבדיקה (מופרדות בפסיקים)", "AAPL,MSFT,NVDA,TSLA,AMD,PINS,F,PLTR")

tickers = [x.strip() for x in tickers_input.split(",") if x.strip()]

def get_data(ticker, interval='1wk'):
    try:
        df = yf.Ticker(ticker).history(period='1y', interval=interval)
        return df if not df.empty else None
    except:
        return None

def detect_W_pattern(df):
    closes = df['Close'].values
    if len(closes) < 10:
        return False
    minima = (np.diff(np.sign(np.diff(closes))) > 0).nonzero()[0] + 1
    if len(minima) >= 2:
        low1, low2 = closes[minima[-2]], closes[minima[-1]]
        if abs(low1 - low2) / low1 < 0.05:
            neck_index = max(minima[-2], minima[-1]) + 1
            if neck_index < len(closes):
                neck = closes[neck_index]
                if closes[-1] > neck:
                    return True
    return False

results = []

progress = st.progress(0)
for i, ticker in enumerate(tickers):
    progress.progress((i + 1) / len(tickers))
    try:
        info = yf.Ticker(ticker).info
        market_cap_val = info.get('marketCap', 0)
        current_price = info.get('currentPrice', 0)

        cap_type = "Large Cap" if market_cap_val >= 1e10 else "Mid Cap" if market_cap_val >= 2e9 else "Small Cap"
        if cap_type != market_cap or not (price_range[0] <= current_price <= price_range[1]):
            continue

        df = get_data(ticker, interval)
        if df is not None and detect_W_pattern(df):
            results.append((ticker, current_price, market_cap_val))
            st.success(f"{ticker} ✅ תבנית W מזוהה")
            st.line_chart(df['Close'])
    except Exception as e:
        st.warning(f"שגיאה ב-{ticker}: {e}")

if results:
    df_results = pd.DataFrame(results, columns=["Ticker", "Price", "Market Cap"])
    st.subheader("📋 מניות שמצאנו")
    st.dataframe(df_results)
    st.download_button("📥 הורד כ־CSV", df_results.to_csv(index=False), "w_pattern_stocks.csv")
else:
    st.info("לא נמצאו מניות עם תבנית W בקריטריונים שנבחרו.")
