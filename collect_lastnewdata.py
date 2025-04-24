import pandas as pd
import requests
import pyodbc
import time
from datetime import datetime, timedelta

# Konfigurasi koneksi SQL Server
SQL_SERVER = "localhost"
SQL_DATABASE = "gotbai"
SQL_USERNAME = "sa"
SQL_PASSWORD = "LEtoy_89"
SQL_DRIVER = "{ODBC Driver 17 for SQL Server}"

def get_sql_connection():
    conn_str = (
        f"DRIVER={SQL_DRIVER};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USERNAME};"
        f"PWD={SQL_PASSWORD}"
    )
    return pyodbc.connect(conn_str)

def fetch_symbols_usdt():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url, timeout=10)
    data = response.json()
    symbols = [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        and not s["symbol"].endswith("DOWNUSDT")
        and not s["symbol"].endswith("UPUSDT")
        and not s["symbol"].startswith("USD")
    ]
    return symbols

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

    response = requests.get(base_url, params=params, timeout=10)
    data = response.json()

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
    df.rename(columns={"open": "opened", "close": "closet"}, inplace=True)
    return df

def save_to_sql(df, symbol, interval, conn):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                IF NOT EXISTS (
                    SELECT 1 FROM historical_klines
                    WHERE symbol = ? AND interval = ? AND timestamp = ?
                )
                INSERT INTO historical_klines (
                    symbol, interval, timestamp, opened, high, low, closet, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, symbol, interval, row["timestamp"],
                 symbol, interval, row["timestamp"],
                 row["opened"], row["high"], row["low"], row["closet"], row["volume"])
        except Exception as e:
            print(f"❌ Gagal simpan data {symbol}-{interval}: {e}")
    conn.commit()
    cursor.close()

def get_last_timestamp_from_db(symbol, interval, conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(timestamp) FROM historical_klines WHERE symbol = ? AND interval = ?
    """, symbol, interval)
    row = cursor.fetchone()
    cursor.close()
    return row[0] if row[0] else None

def download_history_continue(symbol, interval):
    print(f"📥 Melanjutkan download data {symbol} ({interval})...")
    conn = get_sql_connection()

    last_ts = get_last_timestamp_from_db(symbol, interval, conn)
    if last_ts is None:
        # Jika tidak ada data di database, mulai dari 1 tahun lalu
        current_start = datetime.utcnow() - timedelta(days=365)
    else:
        current_start = last_ts + timedelta(minutes={
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "1d": 1440
        }[interval])

    end_time = datetime.utcnow()
    delta_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440
    }[interval]
    batch_minutes = timedelta(minutes=delta_minutes * 1000)

    # Cek data awal apakah ada data untuk simbol ini
    test_df = fetch_binance_klines_batch(symbol, interval, start_time=current_start, end_time=current_start + batch_minutes)
    if test_df.empty:
        print(f"🚫 {symbol} tidak memiliki data candle aktif untuk interval {interval}. Melewati...")
        conn.close()
        return

    while current_start < end_time:
        current_end = min(current_start + batch_minutes, end_time)
        try:
            df = fetch_binance_klines_batch(symbol, interval, start_time=current_start, end_time=current_end)
            if not df.empty:
                save_to_sql(df, symbol, interval, conn)
                print(f"✅ {symbol} {interval} | {df.iloc[0]['timestamp']} s/d {df.iloc[-1]['timestamp']} ({len(df)} bar)")
            else:
                print(f"⚠️ Tidak ada data pada {current_start} s/d {current_end}")
        except Exception as e:
            print(f"❌ Gagal ambil {symbol}-{interval} {current_start}: {e}")
        current_start = current_end
        time.sleep(0.2)

    conn.close()

def main():
    symbols = fetch_symbols_usdt()
    intervals = ["1h", "4h"]

    for symbol in symbols:
        for interval in intervals:
            download_history_continue(symbol, interval)

if __name__ == "__main__":
    main()
