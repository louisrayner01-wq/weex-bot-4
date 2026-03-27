"""
indicators.py
Calculates all technical indicators used as ML features.
Uses the 'ta' library (pip install ta) for reliability.
"""

import numpy as np
import pandas as pd
import ta
import logging
from typing import List

logger = logging.getLogger(__name__)


def candles_to_df(raw_candles: List[List]) -> pd.DataFrame:
    """
    Convert raw Weex candle list to a clean DataFrame.
    Weex returns 11 fields per candle — we only need the first 6:
    [timestamp_ms, open, high, low, close, volume, ...]
    """
    if not raw_candles:
        return pd.DataFrame()

    # Trim to first 6 fields — Weex sends extra fields we don't need
    trimmed = [row[:6] for row in raw_candles]
    df = pd.DataFrame(trimmed, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to the OHLCV DataFrame.
    Returns the df with extra feature columns (NaN rows at start are normal).
    """
    if len(df) < 60:
        logger.warning("Need at least 60 candles for reliable indicators (got %d)", len(df))
        return df

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"]  = ta.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta.trend.ema_indicator(close, window=21)
    df["ema_50"] = ta.trend.ema_indicator(close, window=50)

    # EMA cross signals (positive = bullish)
    df["ema_9_21_cross"]  = df["ema_9"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]

    # MACD
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()   # histogram

    # ADX (trend strength)
    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_obj.adx()

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Rate of change
    df["roc_5"]  = ta.momentum.ROCIndicator(close, window=5).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(close, window=20).roc()

    # ── Volatility ─────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()   # 0=lower, 1=upper (position within band)
    df["bb_width"] = bb.bollinger_wband()   # band width (volatility proxy)

    # ATR (used by risk manager too)
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["vol_sma_20"]   = vol.rolling(20).mean()
    df["vol_ratio"]    = vol / df["vol_sma_20"].replace(0, np.nan)   # relative volume

    # On-balance volume slope
    df["obv"]       = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_slope"] = df["obv"].diff(5)

    # ── Price momentum (raw %) ─────────────────────────────────────────────────
    df["ret_1"]  = close.pct_change(1)
    df["ret_5"]  = close.pct_change(5)
    df["ret_20"] = close.pct_change(20)

    # Normalise close vs EMA
    df["close_vs_ema21"] = (close - df["ema_21"]) / df["ema_21"].replace(0, np.nan)
    df["close_vs_bb"]    = df["bb_pct"] - 0.5   # centred at 0

    return df


# ── Feature list used by the ML model ─────────────────────────────────────────
FEATURE_COLS = [
    "ema_9_21_cross", "ema_21_50_cross",
    "macd", "macd_signal", "macd_diff",
    "adx",
    "rsi_14", "rsi_7",
    "stoch_k", "stoch_d",
    "roc_5", "roc_20",
    "bb_pct", "bb_width",
    "atr_14",
    "vol_ratio", "obv_slope",
    "ret_1", "ret_5", "ret_20",
    "close_vs_ema21", "close_vs_bb",
]
