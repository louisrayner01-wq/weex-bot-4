"""
data_collector.py
═════════════════
Fetches historical OHLCV data for multiple timeframes from Weex.
Saves to /data/{SYMBOL}_{TIMEFRAME}.csv  (Railway persistent volume).

Timeframes collected : 5m, 15m, 1h, 4h, 1d
Symbols              : BTCUSDT, ETHUSDT, SOLUSDT
History target       : 2 years per symbol/timeframe

Pagination strategy
───────────────────
Weex's klines endpoint returns at most 500 candles per request.
We step backwards in time using the endTime parameter, fetching
500-candle batches until we reach the lookback target or the API
returns nothing.

Cache logic
───────────
If a CSV exists and was written less than `refresh_hours` ago,
the file is treated as fresh and no API calls are made.
This keeps Railway redeploys fast (only the first run does the
full backfill; subsequent redeploys just top-up with new candles).
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

BATCH_SIZE    = 500          # candles per API request (Weex max)
LOOKBACK_DAYS = 730          # 2 years of history
REQUEST_DELAY = 0.4          # seconds between requests (rate-limit safety)
REFRESH_HOURS = 12           # re-fetch if CSV is older than this


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
        Fetch and save one symbol/timeframe CSV.
        Returns the DataFrame (from cache or newly collected).
        """
        label    = TF_LABELS.get(timeframe_min, timeframe_min + "m")
        filepath = self._filepath(symbol, label)

        # ── Cache check ───────────────────────────────────────────────────────
        if os.path.exists(filepath):
            age_h = (time.time() - os.path.getmtime(filepath)) / 3600
            if age_h < REFRESH_HOURS:
                logger.info("  CACHED  %s %s  (%.1fh old, refresh > %.0fh)",
                            symbol, label, age_h, REFRESH_HOURS)
                return pd.read_csv(filepath, parse_dates=["timestamp"])

        # ── Fetch ─────────────────────────────────────────────────────────────
        logger.info("  FETCH   %s %s  (target: %d days) …", symbol, label, LOOKBACK_DAYS)
        raw = self._fetch_history(symbol, timeframe_min)

        if not raw:
            logger.warning("  No data returned for %s %s", symbol, label)
            return None

        df = self._to_dataframe(raw)
        df.to_csv(filepath, index=False)
        logger.info("  SAVED   %s %s  → %d candles  (%s)",
                    symbol, label, len(df), filepath)
        return df

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
        Paginate backwards through Weex klines to collect LOOKBACK_DAYS of data.
        Returns a flat list of candle rows in chronological order.
        """
        now_ms         = int(time.time() * 1000)
        target_start   = now_ms - int(LOOKBACK_DAYS * 24 * 3600 * 1000)
        current_end_ms = now_ms
        all_candles    = []
        batch_num      = 0

        # Use symbol without _SPBL suffix for market data
        api_symbol = symbol.replace("_SPBL", "")

        while current_end_ms > target_start:
            batch = self.client.get_candles(
                symbol      = api_symbol,
                granularity = timeframe_min,
                limit       = BATCH_SIZE,
                end_time    = current_end_ms,
            )

            if not batch:
                logger.debug("Empty batch — stopping pagination for %s %s", symbol, timeframe_min)
                break

            # batch is chronological (oldest first) after reversal in WeexClient
            all_candles = batch + all_candles   # prepend older data
            batch_num  += 1

            oldest_ts = int(float(batch[0][0]))   # first row = oldest candle

            if batch_num % 20 == 0:
                logger.info("    … batch %d  |  %d candles so far  |  oldest: %s",
                            batch_num, len(all_candles),
                            pd.Timestamp(oldest_ts, unit="ms").strftime("%Y-%m-%d"))

            if oldest_ts <= target_start:
                break

            # Step back to just before the oldest candle we have
            current_end_ms = oldest_ts - 1
            time.sleep(REQUEST_DELAY)

        return all_candles

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
