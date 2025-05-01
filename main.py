import os
import joblib
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import time
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
from scipy.signal import argrelextrema
import pyodbc  # <-- tambahkan ini

# Load environment variables
load_dotenv()

# Constants
MODEL_DIR = "models"
CACHE_DIR = "cache"

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt"),
        logging.StreamHandler()
    ]
)

# Koneksi SQL Server
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

def fetch_binance_klines(symbol, interval='1h', limit=2):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume",
                                       "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df

def fetch_binance_last_price(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logging.warning(f"Gagal ambil harga terakhir {symbol}: {e}")
        return None

def wait_until_next_candle(interval='1h'):
    now = datetime.utcnow()
    next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    wait_seconds = int((next_candle - now).total_seconds())

    print(f"\n⏳ Menuju candle baru ({interval}):", end=" ")
    while wait_seconds > 0:
        m, s = divmod(wait_seconds, 60)
        time_str = f"{m:02d}:{s:02d}"
        print(f"\r⏳ Menuju candle baru ({interval}) dalam {time_str} (MM:SS)", end="", flush=True)
        time.sleep(1)
        wait_seconds -= 1

    print(f"\r🕒 Candle baru dimulai.{' '*30}")

def wait_for_all_new_candles(symbols, interval='1h'):
    logging.info("🕒 Menunggu semua candle baru terbentuk...")
    last_times = {}

    for symbol in symbols:
        df = fetch_binance_klines(symbol, interval, 2)
        last_times[symbol] = df.iloc[-1]['timestamp']

    while True:
        all_new = True
        current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        for symbol in symbols:
            try:
                df = fetch_binance_klines(symbol, interval, 2)
                latest_time = df.iloc[-1]['timestamp']

                if latest_time < current_time:
                    logging.info(f"⏳ {symbol} belum ada candle baru. Last: {latest_time}, Now: {current_time}")
                    all_new = False
                else:
                    logging.info(f"✅ {symbol} candle baru siap: {latest_time}")
            except Exception as e:
                logging.warning(f"⚠️ Gagal cek candle {symbol}: {e}")
                all_new = False

        if all_new:
            logging.info("✅ Semua candle sudah update.")
            break

        time.sleep(5)

def calculate_technical_indicators(df):
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["signal_line"] = macd.macd_signal()

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    if len(df) > 20:
        lows, highs = df["low"].values, df["high"].values
        swing_lows = argrelextrema(lows, np.less_equal, order=3)[0]
        swing_highs = argrelextrema(highs, np.greater_equal, order=3)[0]
        support_levels, resistance_levels = [], []

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

    df["trend"] = df.apply(lambda row: 'UPTREND' if row["close"] > row["ema_200"] else 'DOWNTREND', axis=1)
    df["trend_encoded"] = df["trend"].map({"DOWNTREND": 0, "UPTREND": 1})

    df.dropna(subset=["rsi", "macd", "signal_line", "ema_200", "bb_high", "bb_low"], inplace=True)
    return df

def analyze_symbol(symbol):
    try:
        logging.info(f"🔍 Analisis {symbol}...")
        model_path = os.path.join(CACHE_DIR, f"{symbol}_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, f"{symbol}_sql_model.pkl")
            if not os.path.exists(model_path):
                logging.error(f"❌ Model tidak ditemukan: {symbol}")
                return

        model_data = joblib.load(model_path)
        model = model_data["model"]
        features = model_data["features"]

        df = fetch_binance_klines(symbol, '1h', 100)
        if len(df) < 50:
            logging.warning(f"⚠️ Data kurang dari 50 row untuk {symbol}")
            return

        df = calculate_technical_indicators(df)
        if df.empty:
            logging.warning(f"⚠️ Tidak ada data valid setelah indikator untuk {symbol}")
            return

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
        last_price = fetch_binance_last_price(symbol)

        # Tanggal dan jam hasil candle +1 jam
        tgl = (latest['timestamp'] + pd.Timedelta(hours=1)).dt.strftime('%Y-%m-%d').values[0]
        jam = int((latest['timestamp'] + pd.Timedelta(hours=1)).dt.strftime('%H').values[0])
        volume = float(latest['volume'].values[0])

        # Simpan ke database
        conn = get_sql_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predict_log (tgl, jam, symbol, label, interval, current_price, confidence, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (tgl, jam, symbol, pred_label, "1h", price, confidence, volume))
        conn.commit()

        cursor.close()
        conn.close()

        log_msg = f"✅ Prediksi {symbol}: {pred_label} | Price: {price:.4f} | Confidence: {confidence:.2f}%"
        if last_price:
            log_msg += f" | Last Price: {last_price:.4f}"
        logging.info(log_msg)

    except Exception as e:
        logging.error(f"❌ Error analisis {symbol}: {e}\n{traceback.format_exc()}")

def run_all():
    symbols = read_symbols_from_file()
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(analyze_symbol, symbols)

def main_loop():
    symbols = read_symbols_from_file()
    while True:
        logging.info(f"🕒 Sinkronisasi waktu {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        wait_until_next_candle()
        wait_for_all_new_candles(symbols)
        logging.info("🚀 Mulai analisis prediksi")
        run_all()

if __name__ == "__main__":
    main_loop()
