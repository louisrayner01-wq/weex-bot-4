"""
data_collector.py
═════════════════
Fetches and ACCUMULATES OHLCV data for multiple timeframes from Weex.
Saves to /data/{SYMBOL}_{TIMEFRAME}.csv  (Railway persistent volume).

Timeframes collected : 5m, 15m, 1h, 4h, 1d
Symbols              : BTCUSDT, ETHUSDT, SOLUSDT

Append strategy
───────────────
Instead of overwriting the CSV on each run, the collector loads the
existing file, finds the latest saved timestamp, fetches the newest
batch from the API, and appends only the candles that are genuinely
new.  This means the CSVs grow indefinitely — giving the analysis
and ML model ever-increasing history to learn from.

On first run (no CSV exists): saves the initial 301-candle batch.
On subsequent runs: appends only candles newer than the last row.

Refresh rate
────────────
Each timeframe refreshes at its own natural cadence:
  5m  → skip if last candle < 5 min old
  15m → skip if last candle < 15 min old
  1h  → skip if last candle < 60 min old
  4h  → skip if last candle < 4 h old
  1d  → skip if last candle < 24 h old
This avoids redundant API calls while keeping data up to date.
"""

import os
import time
import logging
import pandas as pd
from typing import Optional, List

from weex_client import WeexClient

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TIMEFRAMES: List[str] = ["5", "15", "60", "240", "1440"]   # minutes (string keys)

TF_LABELS = {
    "5":    "5m",
    "15":   "15m",
    "60":   "1h",
    "240":  "4h",
    "1440": "1d",
}

SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

REQUEST_DELAY = 0.4   # seconds between requests (rate-limit safety)

# Minimum age of the most-recent candle before we bother fetching again.
# Set to 1× the timeframe period so we only call the API when at least
# one new candle has had a chance to close.
TF_REFRESH_MINUTES = {
    "5":    5,
    "15":   15,
    "60":   60,
    "240":  240,
    "1440": 1440,
}


# ── Core collector ────────────────────────────────────────────────────────────

class DataCollector:
    """Fetches and caches multi-timeframe historical OHLCV data from Weex."""

    def __init__(self, client: WeexClient, data_dir: str = "/data"):
        self.client   = client
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def collect_all(self) -> None:
        """
        Collect (or refresh) historical data for every symbol × timeframe
        combination.  Skips any file that is still fresh.
        """
        total = len(SYMBOLS) * len(TIMEFRAMES)
        done  = 0
        logger.info("═══ Data collection starting — %d symbol/timeframe combos ═══", total)

        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    self.collect_one(symbol, tf)
                except Exception as exc:
                    logger.error("Failed collecting %s %s: %s", symbol, TF_LABELS[tf], exc)
                done += 1
                logger.info("Progress: %d / %d", done, total)

        logger.info("═══ Data collection complete ═══")

    def collect_one(self, symbol: str, timeframe_min: str) -> Optional[pd.DataFrame]:
        """
        Fetch and APPEND new candles for one symbol/timeframe.

        Logic:
          1. Load the existing CSV (if any).
          2. Check whether at least one new candle has had time to close —
             if the most-recent saved candle is still within the current
             candle period, skip (no new data to collect).
          3. Fetch the latest batch from the API.
          4. Append only the candles newer than the last saved timestamp,
             deduplicate, sort, and save.

        Over time the CSV grows continuously, giving the ML model an
        ever-richer history to learn from.
        """
        label    = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        filepath = self._filepath(symbol, label)
        refresh_min = TF_REFRESH_MINUTES.get(timeframe_min, 60)

        # ── Load existing data ────────────────────────────────────────────────
        existing: Optional[pd.DataFrame] = None
        if os.path.exists(filepath):
            try:
                existing = pd.read_csv(filepath, parse_dates=["timestamp"])
                if existing.empty:
                    existing = None
            except Exception as exc:
                logger.warning("Could not load %s: %s — will re-fetch.", filepath, exc)
                existing = None

        # ── Freshness check ───────────────────────────────────────────────────
        if existing is not None and not existing.empty:
            last_ts    = existing["timestamp"].max()
            age_min    = (pd.Timestamp.utcnow().tz_localize(None) - last_ts).total_seconds() / 60
            if age_min < refresh_min:
                logger.info("  FRESH   %s %s  — last candle %.0f min ago (refresh at %d min)",
                            symbol, label, age_min, refresh_min)
                return existing

        # ── Fetch latest batch ────────────────────────────────────────────────
        logger.info("  FETCH   %s %s  …", symbol, label)
        raw = self._fetch_history(symbol, timeframe_min)

        if not raw:
            logger.warning("  No data returned for %s %s", symbol, label)
            return existing

        new_df = self._to_dataframe(raw)

        # ── Append or initialise ──────────────────────────────────────────────
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = (combined
                        .drop_duplicates("timestamp")
                        .sort_values("timestamp")
                        .reset_index(drop=True))
            n_new = len(combined) - len(existing)
            if n_new > 0:
                combined.to_csv(filepath, index=False)
                logger.info("  APPEND  %s %s  +%d new candles → %d total",
                            symbol, label, n_new, len(combined))
            else:
                logger.info("  UP-TO-DATE  %s %s  (%d candles, nothing new)",
                            symbol, label, len(existing))
            return combined
        else:
            new_df.to_csv(filepath, index=False)
            logger.info("  SAVED   %s %s  → %d candles  (initial fetch)",
                        symbol, label, len(new_df))
            return new_df

    def load(self, symbol: str, timeframe_min: str) -> Optional[pd.DataFrame]:
        """Load a saved CSV; returns None if it doesn't exist yet."""
        label    = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        filepath = self._filepath(symbol, label)
        if not os.path.exists(filepath):
            return None
        return pd.read_csv(filepath, parse_dates=["timestamp"])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _filepath(self, symbol: str, label: str) -> str:
        return os.path.join(self.data_dir, f"{symbol}_{label}.csv")

    def _fetch_history(self, symbol: str, timeframe_min: str) -> List[List]:
        """
        Fetch the maximum available candle history from Weex in a single request.

        Weex's klines endpoint does not support endTime/startTime pagination —
        passing those parameters returns a 400 error.  Instead we request the
        largest batch the API will accept, trying progressively smaller limits
        until one succeeds.  This gives us the most recent N candles available.

        Returns a list of candle rows in chronological order.
        """
        api_symbol = symbol.replace("_SPBL", "")

        # Try largest limits first — stop at the first successful response
        for limit in [1000, 500, 300]:
            batch = self.client.get_candles(
                symbol      = api_symbol,
                granularity = timeframe_min,
                limit       = limit,
                # No end_time — Weex rejects it with 400
            )
            if batch:
                logger.info("    … fetched %d candles for %s %s (limit=%d)",
                            len(batch), symbol, timeframe_min, limit)
                return batch
            time.sleep(REQUEST_DELAY)

        return []

    def _to_dataframe(self, raw: List[List]) -> pd.DataFrame:
        """Convert raw candle list to a clean, deduped, sorted DataFrame."""
        trimmed = [row[:6] for row in raw]
        df = pd.DataFrame(trimmed,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df.dropna()
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
        return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    exc = cfg["exchange"]
    client = WeexClient(
        api_key    = exc.get("api_key", ""),
        api_secret = exc.get("api_secret", ""),
        passphrase = exc.get("passphrase", ""),
        base_url   = exc.get("base_url", "https://api-spot.weex.com"),
    )

    data_dir = cfg.get("data", {}).get("data_dir", "/data")
    collector = DataCollector(client, data_dir=data_dir)
    collector.collect_all()


