"""
bot.py  —  Main trading loop

Before every trade the bot prints a full trade quality card:

  ┌─ TRADE QUALITY  BTC ─────────────────────────────┐
  │  Signal     : BUY  (confidence 71.3 %)            │
  │  Win rate   : 58.2 %  (from 34 past trades)       │
  │  Avg R/R    : 2.14                                 │
  │  Expect.Val : +0.041 %  per trade  ✅             │
  │  Actual R/R : 2.10  (SL=£1.5  TP=£3.15)          │
  │  Position   : 0.00021 BTC  (risking £5.00)        │
  │  VERDICT    : TAKE THE TRADE                       │
  └──────────────────────────────────────────────────┘

Run with: python bot.py
"""

import time
import logging
import os
import yaml
from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)
from typing import Dict, Optional

import pandas as pd

from weex_client  import WeexClient
from indicators   import candles_to_df
from strategy     import TradingStrategy, BUY, SELL, HOLD
from risk_manager import RiskManager, Position
from trade_logger import TradeLogger


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Railway / cloud deployment: override with environment variables ────────
    # Set these in Railway's Variables tab — never commit real keys to git.
    if os.getenv("WEEX_API_KEY"):
        cfg["exchange"]["api_key"]    = os.environ["WEEX_API_KEY"]
    if os.getenv("WEEX_API_SECRET"):
        cfg["exchange"]["api_secret"] = os.environ["WEEX_API_SECRET"]
    if os.getenv("WEEX_PASSPHRASE"):
        cfg["exchange"]["passphrase"] = os.environ["WEEX_PASSPHRASE"]

    # PAPER_TRADING env var overrides config.yaml (set to "false" on Railway for live)
    if os.getenv("PAPER_TRADING") is not None:
        cfg["trading"]["paper_trading"] = os.getenv("PAPER_TRADING", "true").lower() == "true"

    # Optional overrides
    if os.getenv("LEVERAGE"):
        cfg["trading"]["leverage"] = int(os.getenv("LEVERAGE"))
    if os.getenv("RISK_PER_TRADE"):
        cfg["risk"]["risk_per_trade_abs"] = float(os.getenv("RISK_PER_TRADE"))

    return cfg


def setup_logging(cfg: dict):
    log_file = cfg["logging"]["log_file"]
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    level = getattr(logging, cfg["logging"].get("level", "INFO"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trade quality card
# ─────────────────────────────────────────────────────────────────────────────

def print_trade_card(pair: str, signal: int, confidence: float,
                     ev: Optional[float], win_rate: Optional[float],
                     avg_rr: Optional[float], actual_rr: float,
                     qty: float, risk_amount: float,
                     ev_reason: str, verdict: bool):
    sig_str = {BUY: "BUY", SELL: "SELL", HOLD: "HOLD"}.get(signal, "?")

    # Build EV string
    if ev is not None:
        ev_str = f"{ev*100:+.3f}%  {'✅' if ev > 0 else '❌'}"
    else:
        ev_str = "N/A (still gathering data)"

    wr_str  = f"{win_rate*100:.1f}%" if win_rate is not None else "N/A"
    rr_str  = f"{avg_rr:.2f}"        if avg_rr   is not None else "N/A"
    verdict_str = "✅ TAKE THE TRADE" if verdict else "❌ SKIP (EV/Win-rate gate)"

    border = "─" * 52
    logging.getLogger("TradeCard").info(
        "\n┌%s┐\n"
        "│  %-50s│\n"
        "│  Signal      : %-4s  (confidence %5.1f %%)       │\n"
        "│  Win rate    : %-44s│\n"
        "│  Avg R/R     : %-44s│\n"
        "│  Expect.Val  : %-44s│\n"
        "│  Actual R/R  : %-5.2f  (ATR-derived SL/TP)         │\n"
        "│  Position    : %.5f %s  (risking £%.2f)  │\n"
        "│  EV detail   : %-44s│\n"
        "│  VERDICT     : %-44s│\n"
        "└%s┘",
        border,
        f"TRADE QUALITY  {pair}",
        sig_str, confidence * 100,
        wr_str,
        rr_str,
        ev_str,
        actual_rr,
        qty, pair.replace("USDT_SPBL", ""), risk_amount,
        ev_reason[:44],
        verdict_str,
        border,
    )


# ─────────────────────────────────────────────────────────────────────────────
class TradingBot:

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg  = load_config(config_path)
        setup_logging(self.cfg)
        self.log  = logging.getLogger("Bot")

        ec = self.cfg["exchange"]
        self.paper    = self.cfg["trading"]["paper_trading"]
        self.pairs    = [p for p in self.cfg["trading"]["pairs"] if p["enabled"]]
        self.tf       = str(self.cfg["trading"]["timeframe"])
        self.lookback = self.cfg["trading"]["lookback_candles"]

        mode = "🟡 PAPER" if self.paper else "🔴 LIVE"
        self.log.info("%s MODE  |  Account: £%.2f  |  Risk/trade: £%.2f  |  Lev: dynamic (max %dx)",
                      mode,
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"],
                      self.cfg["risk"].get("max_leverage", 20))

        self.client   = WeexClient(
            api_key=ec["api_key"], api_secret=ec["api_secret"],
            passphrase=ec["passphrase"], base_url=ec["base_url"],
        )
        self.strategy = TradingStrategy(self.cfg)
        self.risk     = RiskManager(self.cfg)
        self.logger   = TradeLogger(self.cfg["logging"]["trades_file"])
        self._tick_count = 0

    # ── Data ──────────────────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        raw = self.client.get_candles(symbol, granularity=self.tf, limit=self.lookback)
        if not raw:
            self.log.error("No candle data for %s", symbol)
            return None
        return candles_to_df(raw)

    def live_price(self, symbol: str, df: pd.DataFrame) -> float:
        # Use the most-recent candle close as our entry price reference.
        # The Weex v3 ticker endpoint is currently unavailable; candle close
        # price is accurate enough for hourly signal generation and paper trading.
        # For live execution the order fills at the exchange's market price anyway.
        return float(df["close"].iloc[-1])

    def get_equity(self) -> float:
        if self.paper:
            return self.risk.equity
        try:
            return self.client.get_balance().get("USDT", self.risk.equity)
        except Exception:
            return self.risk.equity

    # ── Order execution ───────────────────────────────────────────────────────

    def _buy(self, symbol: str, qty: float, price: float) -> Optional[str]:
        if self.paper:
            self.log.info("[PAPER] BUY  %s  qty=%.5f @ £%.4f", symbol, qty, price)
            return f"paper-buy-{symbol}-{int(time.time())}"
        resp = self.client.place_order(symbol, "buy", qty)
        oid  = resp.get("data", {}).get("orderId")
        if not oid:
            self.log.error("Buy order failed: %s", resp)
        return oid

    def _sell(self, symbol: str, qty: float, price: float) -> Optional[str]:
        if self.paper:
            self.log.info("[PAPER] SELL %s  qty=%.5f @ £%.4f", symbol, qty, price)
            return f"paper-sell-{symbol}-{int(time.time())}"
        resp = self.client.place_order(symbol, "sell", qty)
        oid  = resp.get("data", {}).get("orderId")
        if not oid:
            self.log.error("Sell order failed: %s", resp)
        return oid

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def warmup(self):
        self.log.info("⏳ Fetching historical data and training initial model…")
        primary = self.pairs[0]["symbol"]
        df = self.fetch_candles(primary)
        if df is not None and len(df) >= self.strategy.min_samples:
            self.strategy.train(df)
            self.log.info("✅ Initial model ready.")
        else:
            n = len(df) if df is not None else 0
            self.log.info("⚠️  Only %d candles available — model will train after %d trades.",
                          n, self.strategy.min_ev_trades)

    # ── Main tick ─────────────────────────────────────────────────────────────

    def tick(self):
        equity = self.get_equity()
        self.risk.update_equity(equity)
        self.log.info("══ Tick #%d @ %s  │  Equity: £%.2f ══",
                      self._tick_count, utcnow().strftime("%H:%M UTC"), equity)

        if self.risk.trading_halted():
            return

        for pair_cfg in self.pairs:
            symbol = pair_cfg["symbol"]

            # 1. Fetch candles
            df = self.fetch_candles(symbol)
            if df is None or len(df) < 60:
                continue

            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            # ── 2. Check open positions (SL / TP) ────────────────────────────
            exit_reason = self.risk.should_exit(symbol, price)
            if exit_reason:
                pos = self.risk.open_positions.get(symbol)
                if pos:
                    self._sell(symbol, pos.quantity, price)
                    trade = self.risk.close_position(symbol, price)
                    if trade:
                        self.logger.log_trade(trade, self.risk.equity, exit_reason)
                        self.strategy.record_outcome(trade, df)
                continue

            # ── 3. Get model signal ───────────────────────────────────────────
            signal, buy_p, sell_p = self.strategy.predict(df)
            confidence = buy_p if signal == BUY else sell_p if signal == SELL else max(buy_p, sell_p)

            if signal == HOLD:
                self.log.info("  %s  → HOLD  (buy_p=%.2f  sell_p=%.2f)",
                              symbol, buy_p, sell_p)
                continue

            # ── 4. Calculate R/R and position size ───────────────────────────
            side = "long" if signal == BUY else "short"
            sl   = self.risk.stop_loss_price(price, atr, side)
            tp   = self.risk.take_profit_price(price, atr, side)
            rr_ok, actual_rr = self.risk.rr_acceptable(price, sl, tp, side)

            # Historical stats for EV gate
            ev, win_rate, avg_rr = self.strategy.stats.ev_and_winrate(symbol)
            ev_ok, ev_reason     = self.strategy.trade_is_worth_it(symbol)

            # Dynamic position size + auto-leverage based on stop distance
            qty, leverage = self.risk.calc_position(price, atr, win_rate)

            # ── 5. Print trade quality card ───────────────────────────────────
            verdict = rr_ok and ev_ok and signal != HOLD
            print_trade_card(
                pair=symbol, signal=signal, confidence=confidence,
                ev=ev, win_rate=win_rate, avg_rr=avg_rr,
                actual_rr=actual_rr, qty=qty,
                risk_amount=self.risk.risk_per_trade_abs,
                ev_reason=ev_reason, verdict=verdict,
            )

            # ── 6. Gate checks ────────────────────────────────────────────────
            if not rr_ok:
                self.log.info("  ⛔ %s  R/R %.2f < min %.2f — skipped",
                              symbol, actual_rr, self.risk.min_rr)
                continue

            if not ev_ok:
                self.log.info("  ⛔ %s  EV gate — %s", symbol, ev_reason)
                continue

            can_open, reason = self.risk.can_open(symbol)
            if not can_open:
                self.log.info("  ⛔ %s  %s", symbol, reason)
                continue

            if qty <= 0:
                self.log.warning("  ⚠️  %s  Position size is 0 — check ATR/equity", symbol)
                continue

            # ── 7. Open the position ──────────────────────────────────────────
            if signal == BUY:
                order_id = self._buy(symbol, qty, price)
                if order_id:
                    pos = Position(
                        pair=symbol, side="long",
                        entry_price=price, quantity=qty,
                        stop_loss=sl, take_profit=tp,
                        leverage=leverage,
                        entry_time=utcnow().isoformat(),
                        order_id=order_id,
                    )
                    self.risk.open_position(pos)

            elif signal == SELL and symbol in self.risk.open_positions:
                # Close existing long on sell signal
                pos = self.risk.open_positions[symbol]
                self._sell(symbol, pos.quantity, price)
                trade = self.risk.close_position(symbol, price)
                if trade:
                    self.logger.log_trade(trade, self.risk.equity, "signal_exit")
                    self.strategy.record_outcome(trade, df)

        # Print performance summary every 5 ticks
        if self._tick_count % 5 == 0:
            self.logger.print_summary()
            top_pairs = self.strategy.best_pairs()
            if top_pairs:
                self.log.info("📊 Pair ranking by EV: %s",
                              {k: f"{v*100:+.3f}%" for k, v in top_pairs.items()})

        self._tick_count += 1

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        self.log.info("🚀 Weex Trading Bot  v2  starting…")
        self.log.info("   Pairs     : %s", [p["name"] for p in self.pairs])
        self.log.info("   Timeframe : %s min", self.tf)
        self.log.info("   Leverage  : dynamic — auto-set per trade (max %dx)",
                      self.cfg["risk"].get("max_leverage", 20))
        self.log.info("   Account   : £%.2f  |  Risk/trade: £%.2f",
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"])
        self.log.info("   Retraining: starts every 5 trades → slows as data grows")

        self.warmup()

        interval = self.cfg["trading"]["loop_interval_s"]
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                self.log.info("Shutdown requested by user.")
                self.logger.print_summary()
                break
            except Exception as exc:
                self.log.exception("Unexpected error in tick: %s", exc)

            self.log.info("💤 Next tick in %ds…", interval)
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = TradingBot("config.yaml")
    bot.run()

