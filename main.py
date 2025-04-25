# main.py
import os
import joblib
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
import time
import requests
import pyodbc
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import argrelextrema
import numpy as np

MODEL_DIR = "models"
CACHE_DIR = "cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt"),
        logging.StreamHandler()
    ]
)

def get_sql_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=gotbai;"
        "UID=sa;PWD=LEtoy_89"
    )

def read_symbols_from_file(filepath="listsyombol.txt"):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

def wait_until_next_candle():
    now = datetime.utcnow()
    next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    wait_seconds = (next_candle - now).total_seconds()
    for i in range(int(wait_seconds), 0, -1):
        print(f"⏳ Menuju candle baru (1h) dalam {i} detik...", end="\r")
        time.sleep(1)
    print()

def fetch_binance_klines_batch(symbol, interval, start_time=None, end_time=None):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1000,
    }
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()

        if isinstance(data, dict) and "code" in data and data["code"] == -1121:
            raise ValueError(f"Invalid symbol: {symbol}")

        if not isinstance(data, list):
            raise Exception(f"Unexpected response for {symbol}: {data}")

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
            "open": float, "high": float, "low": float, "close": float, "volume": float
        })
        return df

    except Exception as e:
        logging.warning(f"⚠️ Gagal fetch data {symbol}-{interval}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["signal_line"] = macd.macd_signal()

    df["support"], df["resistance"] = np.nan, np.nan

    if len(df) > 20:
        lows = df["low"].values
        highs = df["high"].values
        swing_lows = argrelextrema(lows, np.less_equal, order=3)[0]
        swing_highs = argrelextrema(highs, np.greater_equal, order=3)[0]

        support_levels = []
        resistance_levels = []

        for idx in swing_lows:
            level = lows[idx]
            hits = np.sum((lows >= level * 0.99) & (lows <= level * 1.01))
            if hits >= 2:
                support_levels.append(level)

        for idx in swing_highs:
            level = highs[idx]
            hits = np.sum((highs >= level * 0.99) & (highs <= level * 1.01))
            if hits >= 2:
                resistance_levels.append(level)

        df["support"] = support_levels[-1] if support_levels else np.nan
        df["resistance"] = resistance_levels[-1] if resistance_levels else np.nan

    return df

def analyze_symbol(symbol):
    logging.info(f"🔍 Menganalisis {symbol}...")

    model_cache_path = os.path.join(CACHE_DIR, f"{symbol}_model.pkl")
    model, features = None, None

    if os.path.exists(model_cache_path):
        model, features, _ = joblib.load(model_cache_path)
        logging.info(f"Model cache ditemukan untuk {symbol}. Menggunakan model yang ada.")
    else:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_sql_model.pkl")
        if os.path.exists(model_path):
            model, features, _ = joblib.load(model_path)
            logging.info(f"Model ditemukan untuk {symbol}. Memuat model.")
        else:
            logging.error(f"❌ Model tidak tersedia untuk {symbol}")
            return

    now = datetime.utcnow()
    history_minutes = 200 * 60
    start_time = now - timedelta(minutes=history_minutes)
    df = fetch_binance_klines_batch(symbol, "1h", start_time=start_time, end_time=now)

    if df is None or df.empty or len(df) < 50:
        logging.warning(f"⚠️ Data tidak cukup untuk {symbol}.")
        return

    df = calculate_technical_indicators(df)
    latest = df.iloc[[-1]]

    missing = [f for f in features if f not in latest.columns]
    if missing:
        logging.error(f"❌ Fitur hilang di {symbol}: {missing}")
        return

    X = latest[features]
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    label_map = {0: "BUY", 1: "SELL", 2: "WAIT"}
    pred_label = label_map.get(pred, "UNKNOWN")
    confidence = round(proba[pred] * 100, 2)
    price = float(latest['close'].values[0])
    volume = float(latest['volume'].values[0])

    logging.info(f"✅ Prediksi {symbol}: {pred_label} | Price: {price:.4f} | Confidence: {confidence:.2f}%")

    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        tgl, jam = now.date(), now.hour

        cursor.execute("""
            INSERT INTO predict_log (tgl, jam, symbol, label, interval, current_price, confidence, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tgl, jam, symbol, pred_label, "1h", price, confidence, volume)
        conn.commit()

        prev = now - timedelta(hours=1)
        cursor.execute("""
            UPDATE predict_log
            SET future_price = ?
            WHERE symbol = ? AND tgl = ? AND jam = ? AND (future_price IS NULL OR future_price = 0)
        """, price, symbol, prev.date(), prev.hour)
        conn.commit()

        if jam == 0:
            yesterday = tgl - timedelta(days=1)
            cursor.execute("""
                UPDATE predict_log
                SET future_price = ?
                WHERE symbol = ? AND tgl = ? AND jam = 23 AND (future_price IS NULL OR future_price = 0)
            """, price, symbol, yesterday)
            conn.commit()

        cursor.close()
        conn.close()
    except Exception as db_err:
        logging.error(f"❌ Gagal simpan prediksi DB: {db_err}")
        traceback.print_exc()

def run_all():
    symbols = read_symbols_from_file()

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(analyze_symbol, symbols)

    for symbol in symbols:
        for interval in ["1h", "4h"]:
            try:
                now = datetime.utcnow()
                last_period = now - timedelta(minutes={"1h": 60, "4h": 240}[interval])
                df = fetch_binance_klines_batch(symbol, interval, start_time=last_period, end_time=now)

                if df.empty:
                    if interval == "4h":
                        logging.warning(f"⚠️ Data 4h kosong untuk {symbol}. Kemungkinan candle belum terbentuk.")
                    continue

                if interval == "4h":
                    latest_ts = df["timestamp"].max()
                    if latest_ts.minute != 0 or latest_ts.hour % 4 != 0:
                        logging.warning(
                            f"⏳ Data 4h belum lengkap untuk {symbol}. Candle terakhir pada {latest_ts}. Menunggu selesai terbentuk."
                        )
                        continue

                conn = get_sql_connection()
                cursor = conn.cursor()
                for _, row in df.iterrows():
                    cursor.execute("""
                        IF NOT EXISTS (
                            SELECT 1 FROM historical_klines WHERE symbol = ? AND interval = ? AND timestamp = ?)
                        INSERT INTO historical_klines (
                            symbol, interval, timestamp, opened, high, low, closet, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, symbol, interval, row["timestamp"],
                          symbol, interval, row["timestamp"],
                          row["open"], row["high"], row["low"], row["close"], row["volume"])
                conn.commit()
                cursor.close()
                conn.close()

            except Exception as e:
                logging.warning(f"⚠️ Gagal update data {symbol}-{interval}: {e}")

def main_loop():
    while True:
        logging.info(f"🕒 Sinkronisasi waktu {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        wait_until_next_candle()
        logging.info("🚀 Mulai update data & analisa prediksi")
        run_all()

if __name__ == "__main__":
    main_loop()
