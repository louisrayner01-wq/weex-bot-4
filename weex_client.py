"""
weex_client.py
Handles all communication with the Weex Spot REST API.
Includes HMAC-SHA256 authentication, rate-limit awareness,
and a clean interface for market data + order management.
"""

import hashlib
import hmac
import base64
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ── Weex Spot API endpoints ────────────────────────────────────────────────────
ENDPOINTS = {
    "ticker":        "/api/v3/market/ticker",
    "candles":       "/api/v3/market/klines",       # confirmed correct endpoint
    "depth":         "/api/v3/market/depth",
    "balance":       "/api/v2/account/assets",
    "place_order":   "/api/v2/trade/orders",
    "cancel_order":  "/api/v2/trade/cancel-order",
    "get_order":     "/api/v2/trade/order",
    "open_orders":   "/api/v2/trade/unfilled-orders",
}

# Map our minute-based timeframe config values → Weex interval strings
# e.g. config "60" → "1h", "240" → "4h"
INTERVAL_MAP = {
    "1":    "1m",
    "3":    "3m",
    "5":    "5m",
    "15":   "15m",
    "30":   "30m",
    "60":   "1h",
    "120":  "2h",
    "240":  "4h",
    "360":  "6h",
    "720":  "12h",
    "1440": "1d",
}

def _market_symbol(symbol: str) -> str:
    """
    Weex market data endpoints use plain symbols (e.g. BTCUSDT),
    while order endpoints use the suffixed form (e.g. BTCUSDT_SPBL).
    Strip the suffix for market data calls.
    """
    return symbol.replace("_SPBL", "").replace("_UMCBL", "").replace("_DMCBL", "")


class WeexClient:
    """Authenticated Weex Spot API client."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                 base_url: str = "https://api-spot.weex.com"):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url   = base_url.rstrip("/")
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Signature ──────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """HMAC-SHA256 → Base64 per Weex auth spec."""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(message, "utf-8"),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode()

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, body),
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP":  ts,
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None, auth: bool = True) -> Dict:
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        headers = self._auth_headers("GET", path + qs) if auth else {}
        try:
            resp = self.session.get(self.base_url + path + qs, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") not in (None, "0", 0, "00000"):
                logger.warning("API warning [GET %s]: %s", path, data)
            return data
        except Exception as exc:
            logger.error("GET %s failed: %s", path, exc)
            return {}

    def _post(self, path: str, payload: Dict, auth: bool = True) -> Dict:
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._auth_headers("POST", path, body) if auth else {}
        try:
            resp = self.session.post(self.base_url + path, headers=headers, data=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") not in (None, "0", 0, "00000"):
                logger.warning("API warning [POST %s]: %s", path, data)
            return data
        except Exception as exc:
            logger.error("POST %s failed: %s", path, exc)
            return {}

    # ── Public market data (no auth needed) ───────────────────────────────────

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Return current price data for a symbol."""
        data = self._get(ENDPOINTS["ticker"],
                         {"symbol": _market_symbol(symbol)}, auth=False)
        return data.get("data")

    def get_candles(self, symbol: str, granularity: str = "60", limit: int = 300) -> List[List]:
        """
        Fetch OHLCV candles from Weex.
        granularity: minutes as string from config ("60" = 1 hour)
        Returns list of [timestamp_ms, open, high, low, close, volume] in chronological order.
        """
        interval = INTERVAL_MAP.get(str(granularity), "1h")
        params = {
            "symbol":   _market_symbol(symbol),
            "interval": interval,
            "limit":    limit,
        }
        data = self._get(ENDPOINTS["candles"], params, auth=False)
        raw = data.get("data", [])
        if not raw:
            logger.debug("Empty candle response for %s: %s", symbol, data)
            return []
        # Weex returns newest-first — reverse to chronological order
        return list(reversed(raw)) if raw else []

    # ── Private account / trading endpoints ───────────────────────────────────

    def get_balance(self) -> Dict[str, float]:
        """Return {coin: available_balance} dict."""
        data = self._get(ENDPOINTS["balance"])
        assets = data.get("data", {}).get("list", [])
        return {a["coinName"]: float(a.get("available", 0)) for a in assets}

    def place_order(self, symbol: str, side: str, quantity: float,
                    price: Optional[float] = None,
                    order_type: str = "market",
                    client_oid: Optional[str] = None) -> Dict:
        """
        Place a spot order.
        symbol: full form e.g. BTCUSDT_SPBL
        side: 'buy' | 'sell'
        order_type: 'market' | 'limit'
        """
        payload: Dict[str, Any] = {
            "symbol":    symbol,
            "side":      side,
            "orderType": order_type,
            "quantity":  str(round(quantity, 6)),
        }
        if order_type == "limit" and price is not None:
            payload["price"] = str(round(price, 2))
        if client_oid:
            payload["clientOrderId"] = client_oid

        logger.info("Placing %s %s order: %s qty=%s price=%s",
                    order_type, side, symbol, quantity, price)
        return self._post(ENDPOINTS["place_order"], payload)

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        payload = {"symbol": symbol, "orderId": order_id}
        return self._post(ENDPOINTS["cancel_order"], payload)

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        data = self._get(ENDPOINTS["get_order"],
                         {"symbol": symbol, "orderId": order_id})
        return data.get("data")

    def get_open_orders(self, symbol: str) -> List[Dict]:
        data = self._get(ENDPOINTS["open_orders"], {"symbol": symbol})
        return data.get("data", {}).get("orderList", [])

    # ── Utility ───────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Quick connectivity check using public ticker endpoint."""
        result = self.get_ticker("BTCUSDT")
        return result is not None
