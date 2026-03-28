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
        data_cfg       = self.cfg.get("data", {})
        self.htf_tf    = str(data_cfg.get("confluence_timeframe", "60"))  # default 1h
        self.htf_label = TF_LABELS.get(self.htf_tf, "1h")

        # Lower-timeframe for reversal exit signal (1h when trading 4h)
        tc_cfg          = self.cfg.get("trading", {})
        self.ltf_tf     = str(tc_cfg.get("ltf_reversal_tf", "60"))
        self.ltf_label  = TF_LABELS.get(self.ltf_tf, "1h")

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
          • Switch the signal timeframe to the best one found by analysis
          • Use the recommended higher-TF filter (or auto-pick the next TF up)
          • Update the loop interval to match the new signal TF
        """
        if not self.strategy.analysis:
            return
        recs = self.strategy.analysis.get("recommendations", {})
        rev_map = {v: k for k, v in TF_LABELS.items()}

        # ── Auto-select HTF based on signal TF (next timeframe up) ────────────
        HTF_UP       = {"5": "60", "15": "60", "60": "240", "240": "1440", "1440": "1440"}
        TF_INTERVALS = {"5": 300, "15": 900, "60": 3600, "240": 14400, "1440": 86400}

        # ── Switch signal timeframe if analysis found a better one ─────────────
        best_signal = recs.get("best_signal_timeframe")  # e.g. "4h"
        if best_signal:
            sig_min = rev_map.get(best_signal)
            if sig_min and sig_min != self.tf:
                old_tf   = self.tf
                self.tf  = sig_min
                # Update loop interval so the run() loop uses the right cadence
                self.cfg["trading"]["loop_interval_s"] = TF_INTERVALS.get(sig_min,
                                                         self.cfg["trading"]["loop_interval_s"])
                # Auto-set HTF to one step up from new signal TF
                htf_min        = HTF_UP.get(sig_min, "1440")
                self.htf_tf    = htf_min
                self.htf_label = TF_LABELS.get(htf_min, "1d")
                self.log.info("📊 Analysis switched signal TF: %s → %s  (HTF=%s  loop=%ds)",
                              TF_LABELS.get(old_tf, old_tf), best_signal,
                              self.htf_label,
                              self.cfg["trading"]["loop_interval_s"])

        # ── Override HTF if analysis explicitly recommends a filter ───────────
        best_filter = recs.get("best_filter_timeframe")  # e.g. "1h"
        if best_filter and best_filter != "None":
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
        Uses the SIGNAL timeframe CSV (not the HTF CSV) so the model learns
        the same candle patterns it will predict on during ticking.
        Falls back to live candles if no CSV exists.
        """
        primary   = self.pairs[0]["symbol"]
        sig_label = TF_LABELS.get(self.tf, "4h")   # e.g. "240" → "4h"
        self.log.info("⏳ Training model on historical data for %s [%s]…", primary, sig_label)

        # Try loading the signal-TF historical CSV first
        hist = self.strategy.load_historical_candles(primary, sig_label)
        if hist is not None and len(hist) >= self.strategy.min_samples:
            self.strategy.train(hist, symbol=primary, timeframe_label=sig_label)
            self.log.info("✅ Model trained on %d historical candles [%s].", len(hist), sig_label)
        else:
            # Fallback: fetch live candles at the signal TF
            df = self.fetch_candles(primary)
            if df is not None and len(df) >= self.strategy.min_samples:
                self.strategy.train(df, symbol=primary, timeframe_label=sig_label)
                self.log.info("✅ Model trained on %d live candles (historical CSV not ready).",
                              len(df))
            else:
                n = len(df) if df is not None else 0
                self.log.info("⚠️  Only %d candles available — model will train once data builds up.", n)

    # ── LTF reversal detector ─────────────────────────────────────────────────

    def _ltf_reversal(self, df_ltf: pd.DataFrame, side: str) -> bool:
        """
        Returns True if the lower-timeframe (1h) shows a reversal signal
        that suggests the current trend is running out of steam.

        For a LONG position (looking for bearish reversal):
          • WaveTrend WT1 crossed BELOW WT2 while coming from positive territory, OR
          • MACD histogram flipped from positive to negative this candle

        For a SHORT position (looking for bullish reversal):
          • WaveTrend WT1 crossed ABOVE WT2 while coming from negative territory, OR
          • MACD histogram flipped from negative to positive
        """
        if df_ltf is None or len(df_ltf) < 30:
            return False
        try:
            from indicators import compute_features
            df = compute_features(df_ltf.copy())
            if len(df) < 2:
                return False
            last = df.iloc[-1]
            prev = df.iloc[-2]

            if side == "long":
                wt_cross_bear  = float(last.get("vmcb_wt_cross_bear", 0)) == 1.0
                wt_was_pos     = float(last.get("vmcb_wt1", 0))           >  0
                macd_flip_bear = (float(last.get("macd_diff", 0)) < 0 and
                                  float(prev.get("macd_diff", 0)) >= 0)
                return (wt_cross_bear and wt_was_pos) or macd_flip_bear
            else:
                wt_cross_bull  = float(last.get("vmcb_wt_cross_bull", 0)) == 1.0
                wt_was_neg     = float(last.get("vmcb_wt1", 0))           <  0
                macd_flip_bull = (float(last.get("macd_diff", 0)) > 0 and
                                  float(prev.get("macd_diff", 0)) <= 0)
                return (wt_cross_bull and wt_was_neg) or macd_flip_bull
        except Exception as exc:
            self.log.debug("LTF reversal check error: %s", exc)
            return False

    # ── Shared entry logic ────────────────────────────────────────────────────

    def _try_enter(self, symbol: str, df: pd.DataFrame,
                   price: float, atr: float, htf_direction: int) -> bool:
        """
        Evaluate a potential entry for `symbol` given current candle data.
        Called from both tick() (4h) and scan_entries() (15 min).
        Returns True if a position was opened.
        """
        signal, buy_p, sell_p = self.strategy.predict(df)
        signal, buy_p, sell_p = self.strategy.apply_confluence(
            signal, buy_p, sell_p, htf_direction
        )

        confidence = buy_p if signal == BUY else sell_p if signal == SELL else max(buy_p, sell_p)

        if signal == HOLD:
            self.log.info("  %s  → HOLD  (buy_p=%.2f  sell_p=%.2f  htf=%+d)",
                          symbol, buy_p, sell_p, htf_direction)
            return False

        side = "long" if signal == BUY else "short"
        sl   = self.risk.stop_loss_price(price, atr, side)
        tp   = self.risk.take_profit_price(price, atr, side)
        tp1  = self.risk.tp1_price_for(price, atr, side)
        rr_ok, actual_rr = self.risk.rr_acceptable(price, sl, tp, side)

        ev, win_rate, avg_rr = self.strategy.stats.ev_and_winrate(symbol)
        ev_ok, ev_reason     = self.strategy.trade_is_worth_it(symbol)

        qty, leverage = self.risk.calc_position(price, atr, win_rate)

        verdict = rr_ok and ev_ok and signal != HOLD
        print_trade_card(
            pair=symbol, signal=signal, confidence=confidence,
            ev=ev, win_rate=win_rate, avg_rr=avg_rr,
            actual_rr=actual_rr, qty=qty,
            risk_amount=self.risk.risk_per_trade_abs,
            ev_reason=ev_reason, verdict=verdict,
            htf_label=self.htf_label, htf_direction=htf_direction,
        )

        if not rr_ok:
            self.log.info("  ⛔ %s  R/R %.2f < min %.2f — skipped", symbol, actual_rr, self.risk.min_rr)
            return False
        if not ev_ok:
            self.log.info("  ⛔ %s  EV gate — %s", symbol, ev_reason)
            return False

        can_open, reason = self.risk.can_open(symbol)
        if not can_open:
            self.log.info("  ⛔ %s  %s", symbol, reason)
            return False
        if qty <= 0:
            self.log.warning("  ⚠️  %s  Position size is 0 — check ATR/equity", symbol)
            return False

        if signal == BUY:
            order_id = self._buy(symbol, qty, price)
            if order_id:
                pos = Position(
                    pair=symbol, side="long",
                    entry_price=price, quantity=qty,
                    stop_loss=sl, take_profit=tp,
                    tp1_price=tp1, quantity_original=qty,
                    leverage=leverage,
                    entry_time=utcnow().isoformat(),
                    order_id=order_id,
                )
                self.risk.open_position(pos)
                return True

        elif signal == SELL and symbol in self.risk.open_positions:
            pos = self.risk.open_positions[symbol]
            self._sell(symbol, pos.quantity, price)
            trade = self.risk.close_position(symbol, price)
            if trade:
                self.logger.log_trade(trade, self.risk.equity, "signal_exit")
                self.strategy.record_outcome(
                    trade, df, symbol=symbol,
                    timeframe_label=TF_LABELS.get(self.tf, "4h"),
                )

        return False

    # ── Exit monitor (runs every 5 min regardless of signal TF) ──────────────

    def monitor_exits(self):
        """
        Runs every 5 min. Handles three exit scenarios for open positions:

        1. Stop-loss / final TP  → full close (same as before)
        2. TP1 hit               → partial close (50%), move SL to breakeven
        3. LTF reversal (1h)     → full close of remaining qty if TP1 already hit
           (momentum fading signal — lets winners run further than a fixed TP)
        """
        if not self.risk.open_positions:
            return

        for pair_cfg in self.pairs:
            symbol = pair_cfg["symbol"]
            pos    = self.risk.open_positions.get(symbol)
            if not pos:
                continue

            # Quick candle fetch for current price
            df = self.fetch_candles(symbol, limit=10)
            if df is None or df.empty:
                continue
            price = self.live_price(symbol, df)

            exit_reason = self.risk.should_exit(symbol, price)

            # ── TP1: partial close — position stays open ───────────────────
            if exit_reason == "tp1":
                self._sell(symbol,
                           round(pos.quantity_original * self.risk.tp1_close_pct, 6),
                           price)
                trade = self.risk.partial_close(symbol, price)
                if trade:
                    self.logger.log_trade(trade, self.risk.equity, "tp1_partial")
                # Don't continue — position still open, check LTF next cycle
                continue

            # ── SL or final TP: full close ─────────────────────────────────
            if exit_reason in ("stop_loss", "take_profit"):
                self.log.info("🚨 %s  %s  @ £%.4f", exit_reason.upper(), symbol, price)
                self._sell(symbol, pos.quantity, price)
                trade = self.risk.close_position(symbol, price)
                if trade:
                    self.logger.log_trade(trade, self.risk.equity, exit_reason)
                    full_df = self.fetch_candles(symbol)
                    if full_df is not None:
                        self.strategy.record_outcome(
                            trade, full_df, symbol=symbol,
                            timeframe_label=TF_LABELS.get(self.tf, "4h"),
                        )
                continue

            # ── LTF reversal exit (only after TP1 has been hit) ───────────
            if pos.tp1_hit:
                ltf_df = self.fetch_candles(symbol, tf=self.ltf_tf, limit=50)
                if self._ltf_reversal(ltf_df, pos.side):
                    self.log.info("📉 LTF reversal exit  %s  @ £%.4f  (1h %s signal)",
                                  symbol, price, self.ltf_label)
                    self._sell(symbol, pos.quantity, price)
                    trade = self.risk.close_position(symbol, price)
                    if trade:
                        self.logger.log_trade(trade, self.risk.equity, "ltf_reversal")
                        full_df = self.fetch_candles(symbol)
                        if full_df is not None:
                            self.strategy.record_outcome(
                                trade, full_df, symbol=symbol,
                                timeframe_label=TF_LABELS.get(self.tf, "4h"),
                            )

    # ── Background data accumulation ─────────────────────────────────────────

    def accumulate_data(self):
        """
        Called on every 5-min monitor cycle.
        Appends any newly closed candles to all 15 CSV files.
        Each timeframe's collector skips the call if the data is already
        fresh (i.e. no new candle has closed since the last append).
        Over time this builds up unlimited history — critical for 5m/15m
        analysis which only has ~25 h / ~3 days from the initial API batch.
        """
        try:
            self.collector.collect_all(quiet=True)
        except Exception as exc:
            self.log.debug("Background data accumulation error: %s", exc)

    # ── 15-min entry scan (runs when flat — no open positions) ───────────────

    def scan_entries(self):
        """
        Lightweight entry scan that runs every entry_scan_interval_s (15 min)
        when the bot has no open positions.

        Fetches the latest candles and evaluates the ML signal + confluence
        filter for each pair. If a signal passes all gates, opens the trade.
        No retraining or performance summaries — just entry hunting.
        """
        if self.risk.open_positions:
            return   # already in a trade; monitor_exits handles everything

        if self.risk.trading_halted():
            return

        results = []   # collect per-pair results into one summary log line

        for pair_cfg in self.pairs:
            symbol = pair_cfg["symbol"]
            name   = pair_cfg["name"]
            if symbol in self.risk.open_positions:
                continue

            df = self.fetch_candles(symbol)
            if df is None or len(df) < 60:
                results.append(f"{name}→NO_DATA")
                continue

            price = self.live_price(symbol, df)
            atr   = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else price * 0.01

            htf_df = None
            if self.htf_tf != self.tf:
                htf_df = self.fetch_candles(symbol, tf=self.htf_tf, limit=100)
            htf_direction = self.strategy.htf_trend(htf_df) if htf_df is not None else 0

            # Peek at probabilities for the summary line
            signal, buy_p, sell_p = self.strategy.predict(df)
            signal, buy_p, sell_p = self.strategy.apply_confluence(
                signal, buy_p, sell_p, htf_direction
            )

            if signal == 0:   # HOLD
                results.append(f"{name}→HOLD(b{buy_p:.2f}/s{sell_p:.2f})")
                continue

            # Non-HOLD signal — run full gate checks via _try_enter
            entered = self._try_enter(symbol, df, price, atr, htf_direction)
            results.append(f"{name}→{'ENTERED' if entered else 'BLOCKED'}")

        self.log.info("🔍 Scan @ %s  |  %s",
                      utcnow().strftime("%H:%M UTC"), "  ".join(results))

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

            # ── 3. Skip entry evaluation if already in this position ─────────
            # (monitor_exits handles all exit logic on its 5-min cycle)
            if symbol in self.risk.open_positions:
                continue

            # ── 4–9. Evaluate entry signal and open position if warranted ─────
            self._try_enter(symbol, df, price, atr, htf_direction)

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

        # Three-speed loop:
        #   Every 5 min  → monitor_exits()   — SL/TP/TP1/LTF reversal check
        #   Every 15 min → scan_entries()    — ML entry hunt when flat
        #   Every 4 h    → tick()            — full tick: retrain, summaries, entries
        tc = self.cfg["trading"]
        MONITOR_INTERVAL = tc.get("monitor_interval_s",    300)    # 5 min
        SCAN_INTERVAL    = tc.get("entry_scan_interval_s", 900)    # 15 min
        SIGNAL_INTERVAL  = tc["loop_interval_s"]                   # 4 h

        last_scan_time   = 0.0   # force an entry scan almost immediately
        last_signal_time = 0.0   # force a full tick on startup

        while True:
            try:
                now = time.time()

                # 1. Always check SL / TP / TP1 / LTF reversal + accumulate data
                self.monitor_exits()
                self.accumulate_data()

                # 2. Full 4h tick — retraining, performance summary, entries
                if now - last_signal_time >= SIGNAL_INTERVAL:
                    self.tick()
                    last_signal_time = time.time()
                    last_scan_time   = time.time()   # reset so scan doesn't fire immediately after

                # 3. 15-min entry scan — only runs when no position is open
                elif now - last_scan_time >= SCAN_INTERVAL:
                    self.scan_entries()
                    last_scan_time = time.time()

            except KeyboardInterrupt:
                self.log.info("Shutdown requested by user.")
                self.logger.print_summary()
                break
            except Exception as exc:
                self.log.exception("Unexpected error in main loop: %s", exc)

            time.sleep(MONITOR_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = TradingBot("config.yaml")
    bot.run()



