"""
bot.py  —  Main trading loop

Before every trade the bot prints a full trade quality card:

  ┌─ TRADE QUALITY  BTC ─────────────────────────────┐
  │  Signal     : BUY  (confidence 71.3 %)            │
  │  Win rate   : 58.2 %  (from 34 past trades)       │
  │  Avg R/R    : 2.14                                 │
  │  Expect.Val : +0.041 %  per trade  ✅             │
  │  Actual R/R : 2.10  (SL=£1.5  TP=£3.15)          │
  │  HTF trend  : 1h BULLISH ✅ (confluence confirmed) │
  │  Position   : 0.00021 BTC  (risking £5.00)        │
  │  VERDICT    : TAKE THE TRADE                       │
  └──────────────────────────────────────────────────┘

Startup sequence
────────────────
1. Data collection  — pulls up to 2 years of multi-TF candles from Weex
                      (skipped if CSVs are fresh, ~2 min on first run)
2. Analysis         — backtests every timeframe + confluence combination
                      and saves recommendations to /data/analysis_results.json
                      (skipped if results are < 7 days old)
3. Strategy init    — loads analysis recommendations, pre-trains model on
                      the full historical dataset
4. Trading loop     — evaluates signals every N minutes, applies HTF confluence
                      filter before executing any trade

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

from weex_client   import WeexClient
from indicators    import candles_to_df
from strategy      import TradingStrategy, BUY, SELL, HOLD
from risk_manager  import RiskManager, Position
from trade_logger  import TradeLogger
from data_collector import DataCollector, TF_LABELS
from analysis      import Analyzer


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # ── Railway / cloud deployment: override with environment variables ────────
    if os.getenv("WEEX_API_KEY"):
        cfg["exchange"]["api_key"]    = os.environ["WEEX_API_KEY"]
    if os.getenv("WEEX_API_SECRET"):
        cfg["exchange"]["api_secret"] = os.environ["WEEX_API_SECRET"]
    if os.getenv("WEEX_PASSPHRASE"):
        cfg["exchange"]["passphrase"] = os.environ["WEEX_PASSPHRASE"]

    if os.getenv("PAPER_TRADING") is not None:
        cfg["trading"]["paper_trading"] = os.getenv("PAPER_TRADING", "true").lower() == "true"

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
                     ev_reason: str, verdict: bool,
                     htf_label: str = "", htf_direction: int = 0):
    sig_str = {BUY: "BUY", SELL: "SELL", HOLD: "HOLD"}.get(signal, "?")

    if ev is not None:
        ev_str = f"{ev*100:+.3f}%  {'✅' if ev > 0 else '❌'}"
    else:
        ev_str = "N/A (still gathering data)"

    wr_str  = f"{win_rate*100:.1f}%" if win_rate is not None else "N/A"
    rr_str  = f"{avg_rr:.2f}"        if avg_rr   is not None else "N/A"
    verdict_str = "✅ TAKE THE TRADE" if verdict else "❌ SKIP (EV/Win-rate gate)"

    htf_str = ""
    if htf_label:
        trend_word = {1: "BULLISH ✅", -1: "BEARISH ⚠️", 0: "NEUTRAL"}.get(htf_direction, "N/A")
        htf_str = f"{htf_label} {trend_word}"

    border = "─" * 52
    logging.getLogger("TradeCard").info(
        "\n┌%s┐\n"
        "│  %-50s│\n"
        "│  Signal      : %-4s  (confidence %5.1f %%)       │\n"
        "│  Win rate    : %-44s│\n"
        "│  Avg R/R     : %-44s│\n"
        "│  Expect.Val  : %-44s│\n"
        "│  Actual R/R  : %-5.2f  (ATR-derived SL/TP)         │\n"
        "│  HTF trend   : %-44s│\n"
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
        htf_str or "N/A (no HTF data)",
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
        self.data_dir = self.cfg.get("data", {}).get("data_dir", "/data")

        # Higher-timeframe for confluence filter
        # Analysis recommendations override this after the analysis runs
        data_cfg      = self.cfg.get("data", {})
        self.htf_tf   = str(data_cfg.get("confluence_timeframe", "60"))  # default 1h
        self.htf_label = TF_LABELS.get(self.htf_tf, "1h")

        mode = "🟡 PAPER" if self.paper else "🔴 LIVE"
        self.log.info("%s MODE  |  Account: £%.2f  |  Risk/trade: £%.2f  |  Lev: dynamic (max %dx)",
                      mode,
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"],
                      self.cfg["risk"].get("max_leverage", 20))

        self.client    = WeexClient(
            api_key=ec["api_key"], api_secret=ec["api_secret"],
            passphrase=ec["passphrase"], base_url=ec["base_url"],
        )
        self.collector = DataCollector(self.client, data_dir=self.data_dir)
        self.analyzer  = Analyzer(data_dir=self.data_dir, results_dir=self.data_dir)
        self.strategy  = TradingStrategy(self.cfg)
        self.risk      = RiskManager(self.cfg)
        self.logger    = TradeLogger(self.cfg["logging"]["trades_file"])
        self._tick_count = 0

    # ── Data ──────────────────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str,
                      tf: Optional[str] = None,
                      limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        raw = self.client.get_candles(
            symbol,
            granularity = tf or self.tf,
            limit       = limit or self.lookback,
        )
        if not raw:
            self.log.error("No candle data for %s", symbol)
            return None
        return candles_to_df(raw)

    def live_price(self, symbol: str, df: pd.DataFrame) -> float:
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

    # ── Startup sequence ──────────────────────────────────────────────────────

    def startup(self):
        """
        Full startup pipeline:
          1. Collect historical data (skipped if CSVs are fresh)
          2. Run analysis (skipped if results are < 7 days old)
          3. Apply analysis recommendations
          4. Train initial model on historical data
        """
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.info("  STARTUP PIPELINE")
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # ── Step 1: Data collection ───────────────────────────────────────────
        self.log.info("STEP 1/3  Data collection")
        try:
            self.collector.collect_all()
        except Exception as exc:
            self.log.error("Data collection error (continuing anyway): %s", exc)

        # ── Step 2: Analysis ──────────────────────────────────────────────────
        self.log.info("STEP 2/3  Strategy analysis")
        try:
            if not self.analyzer.results_are_fresh(max_age_days=7):
                self.analyzer.run()
            else:
                self.log.info("  Analysis results are fresh — skipping re-run.")
        except Exception as exc:
            self.log.error("Analysis error (continuing anyway): %s", exc)

        # ── Step 3: Apply recommendations + train ─────────────────────────────
        self.log.info("STEP 3/3  Initial model training")
        try:
            # Reload analysis so strategy picks up fresh recommendations
            self.strategy.reload_analysis()
            self._apply_analysis_recommendations()
            self._initial_train()
        except Exception as exc:
            self.log.error("Training error (continuing anyway): %s", exc)

        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.log.info("  STARTUP COMPLETE — entering trading loop")
        self.log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _apply_analysis_recommendations(self):
        """
        Read analysis results and update bot settings accordingly:
          • Use the recommended signal timeframe
          • Use the recommended higher-TF filter
        """
        if not self.strategy.analysis:
            return
        recs = self.strategy.analysis.get("recommendations", {})

        # Update HTF confluence timeframe
        best_filter = recs.get("best_filter_timeframe")  # e.g. "1h"
        if best_filter:
            # Map label back to minutes key
            rev_map = {v: k for k, v in TF_LABELS.items()}
            htf_min = rev_map.get(best_filter)
            if htf_min:
                self.htf_tf    = htf_min
                self.htf_label = best_filter
                self.log.info("📊 Analysis recommends HTF filter: %s", best_filter)

        # Log top features so they're visible in logs
        top_feats = recs.get("top_features", [])[:5]
        if top_feats:
            self.log.info("📊 Top features from analysis: %s", top_feats)

    def _initial_train(self):
        """
        Train the model on the full historical dataset for the primary pair.
        Falls back to live candles if no CSV exists.
        """
        primary = self.pairs[0]["symbol"]
        self.log.info("⏳ Training model on historical data for %s…", primary)

        # Try loading historical CSV first (this feeds the train() call inside strategy)
        hist = self.strategy.load_historical_candles(primary, self.htf_label)
        if hist is not None and len(hist) >= self.strategy.min_samples:
            self.strategy.train(hist, symbol=primary, timeframe_label=self.htf_label)
            self.log.info("✅ Model trained on %d historical candles.", len(hist))
        else:
            # Fallback: use live candles
            df = self.fetch_candles(primary)
            if df is not None and len(df) >= self.strategy.min_samples:
                self.strategy.train(df, symbol=primary, timeframe_label=self.htf_label)
                self.log.info("✅ Model trained on %d live candles (historical CSV not ready).",
                              len(df))
            else:
                n = len(df) if df is not None else 0
                self.log.info("⚠️  Only %d candles available — model will train once data builds up.", n)

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

            # ── 1. Fetch primary-TF candles ───────────────────────────────────
            df = self.fetch_candles(symbol)
            if df is None or len(df) < 60:
                continue

            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            # ── 2. Fetch higher-TF candles for confluence ─────────────────────
            htf_df = None
            if self.htf_tf != self.tf:
                htf_df = self.fetch_candles(symbol, tf=self.htf_tf, limit=100)
            htf_direction = self.strategy.htf_trend(htf_df) if htf_df is not None else 0

            # ── 3. Check open positions (SL / TP) ────────────────────────────
            exit_reason = self.risk.should_exit(symbol, price)
            if exit_reason:
                pos = self.risk.open_positions.get(symbol)
                if pos:
                    self._sell(symbol, pos.quantity, price)
                    trade = self.risk.close_position(symbol, price)
                    if trade:
                        self.logger.log_trade(trade, self.risk.equity, exit_reason)
                        self.strategy.record_outcome(
                            trade, df,
                            symbol=symbol,
                            timeframe_label=TF_LABELS.get(self.tf, "15m"),
                        )
                continue

            # ── 4. Get model signal ───────────────────────────────────────────
            signal, buy_p, sell_p = self.strategy.predict(df)

            # ── 5. Apply HTF confluence filter ────────────────────────────────
            signal, buy_p, sell_p = self.strategy.apply_confluence(
                signal, buy_p, sell_p, htf_direction
            )

            confidence = buy_p if signal == BUY else sell_p if signal == SELL else max(buy_p, sell_p)

            if signal == HOLD:
                self.log.info("  %s  → HOLD  (buy_p=%.2f  sell_p=%.2f  htf=%+d)",
                              symbol, buy_p, sell_p, htf_direction)
                continue

            # ── 6. Calculate R/R and position size ───────────────────────────
            side = "long" if signal == BUY else "short"
            sl   = self.risk.stop_loss_price(price, atr, side)
            tp   = self.risk.take_profit_price(price, atr, side)
            rr_ok, actual_rr = self.risk.rr_acceptable(price, sl, tp, side)

            ev, win_rate, avg_rr = self.strategy.stats.ev_and_winrate(symbol)
            ev_ok, ev_reason     = self.strategy.trade_is_worth_it(symbol)

            qty, leverage = self.risk.calc_position(price, atr, win_rate)

            # ── 7. Print trade quality card ───────────────────────────────────
            verdict = rr_ok and ev_ok and signal != HOLD
            print_trade_card(
                pair=symbol, signal=signal, confidence=confidence,
                ev=ev, win_rate=win_rate, avg_rr=avg_rr,
                actual_rr=actual_rr, qty=qty,
                risk_amount=self.risk.risk_per_trade_abs,
                ev_reason=ev_reason, verdict=verdict,
                htf_label=self.htf_label, htf_direction=htf_direction,
            )

            # ── 8. Gate checks ────────────────────────────────────────────────
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

            # ── 9. Open the position ──────────────────────────────────────────
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
                pos = self.risk.open_positions[symbol]
                self._sell(symbol, pos.quantity, price)
                trade = self.risk.close_position(symbol, price)
                if trade:
                    self.logger.log_trade(trade, self.risk.equity, "signal_exit")
                    self.strategy.record_outcome(
                        trade, df,
                        symbol=symbol,
                        timeframe_label=TF_LABELS.get(self.tf, "15m"),
                    )

        # Performance summary every 5 ticks
        if self._tick_count % 5 == 0:
            self.logger.print_summary()
            top_pairs = self.strategy.best_pairs()
            if top_pairs:
                self.log.info("📊 Pair ranking by EV: %s",
                              {k: f"{v*100:+.3f}%" for k, v in top_pairs.items()})

        self._tick_count += 1

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        self.log.info("🚀 Weex Trading Bot  v3  starting…")
        self.log.info("   Pairs      : %s", [p["name"] for p in self.pairs])
        self.log.info("   Signal TF  : %s min", self.tf)
        self.log.info("   Filter TF  : %s (HTF confluence)", self.htf_label)
        self.log.info("   Leverage   : dynamic — auto-set per trade (max %dx)",
                      self.cfg["risk"].get("max_leverage", 20))
        self.log.info("   Account    : £%.2f  |  Risk/trade: £%.2f",
                      self.cfg["risk"]["initial_capital"],
                      self.cfg["risk"]["risk_per_trade_abs"])

        self.startup()

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

