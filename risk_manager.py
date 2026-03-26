"""
risk_manager.py
Position sizing, stop-loss / take-profit, R/R validation,
and daily drawdown protection.

Key changes vs v1:
  - Absolute £5 risk per trade (not a percentage), so position size
    is always calculated to lose exactly £5 if stop-loss is hit
  - R/R gate: trade is blocked if actual R/R < min_rr_ratio
  - Kelly-aware sizing: position never exceeds what the Kelly formula
    would recommend given historical win rate
  - Daily loss halt uses absolute £ amount, not percentage
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class Position:
    pair:          str
    side:          str        # 'long' | 'short'
    entry_price:   float
    quantity:      float
    stop_loss:     float
    take_profit:   float
    leverage:      int
    entry_time:    str
    candles_held:  int = 0
    order_id:      Optional[str] = None

    @property
    def rr_ratio(self) -> float:
        if self.side == "long":
            risk    = self.entry_price - self.stop_loss
            reward  = self.take_profit - self.entry_price
        else:
            risk    = self.stop_loss - self.entry_price
            reward  = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0.0


class RiskManager:

    def __init__(self, cfg: dict):
        rc = cfg["risk"]
        sc = cfg["strategy"]

        self.initial_capital    = rc["initial_capital"]      # £100
        self.risk_per_trade_abs = rc["risk_per_trade_abs"]   # £5
        self.sl_atr_mult        = rc["stop_loss_atr_mult"]   # 1.5
        self.tp_atr_mult        = rc["take_profit_atr_mult"] # 3.0
        self.min_rr             = rc["min_rr_ratio"]         # 1.5
        self.max_open           = rc["max_open_positions"]   # 2
        self.max_daily_loss     = rc["max_daily_loss_abs"]   # £10
        self.min_holding        = sc.get("min_holding_candles", 2)

        self.equity             = float(self.initial_capital)
        self.day_start_equity   = float(self.initial_capital)
        self.today              = date.today()
        self.open_positions: Dict[str, Position] = {}

    # ── Equity ────────────────────────────────────────────────────────────────

    def update_equity(self, new_equity: float):
        today = date.today()
        if today != self.today:
            self.day_start_equity = new_equity
            self.today = today
        self.equity = new_equity

    def daily_loss(self) -> float:
        return self.day_start_equity - self.equity   # positive = loss

    def trading_halted(self) -> bool:
        halted = self.daily_loss() >= self.max_daily_loss
        if halted:
            logger.warning(
                "🛑 Daily loss limit hit (£%.2f / £%.2f). Trading halted for today.",
                self.daily_loss(), self.max_daily_loss
            )
        return halted

    # ── Stop / take-profit prices ─────────────────────────────────────────────

    def stop_loss_price(self, entry: float, atr: float, side: str) -> float:
        dist = atr * self.sl_atr_mult
        return entry - dist if side == "long" else entry + dist

    def take_profit_price(self, entry: float, atr: float, side: str) -> float:
        dist = atr * self.tp_atr_mult
        return entry + dist if side == "long" else entry - dist

    # ── R/R validation ────────────────────────────────────────────────────────

    def rr_acceptable(self, entry: float, sl: float, tp: float, side: str) -> Tuple[bool, float]:
        """
        Returns (is_acceptable, actual_rr).
        Checks the real R/R from the ATR-derived stop and take-profit levels.
        """
        if side == "long":
            risk   = entry - sl
            reward = tp - entry
        else:
            risk   = sl - entry
            reward = entry - tp

        if risk <= 0:
            return False, 0.0

        rr = reward / risk
        ok = rr >= self.min_rr
        return ok, round(rr, 2)

    # ── Position sizing ───────────────────────────────────────────────────────

    def position_size(self, entry: float, atr: float, leverage: int = 1,
                      win_rate: Optional[float] = None) -> float:
        """
        Size the position so that if stop-loss is hit, we lose exactly £risk_per_trade_abs.

        stop_distance = atr × sl_multiplier  (in price terms)
        qty = risk_amount / stop_distance

        With leverage, the notional exposure is qty × entry; margin used is
        (qty × entry) / leverage. We also cap at a Kelly-adjusted fraction.
        """
        if entry <= 0 or atr <= 0:
            return 0.0

        stop_distance = atr * self.sl_atr_mult          # e.g. £500 per BTC
        risk_amount   = self.risk_per_trade_abs          # £5

        # Base quantity (how many units to lose exactly £5 if SL hit)
        qty = (risk_amount * leverage) / stop_distance

        # Kelly fraction cap: if we have a win rate, don't bet more than Kelly suggests
        if win_rate is not None and 0 < win_rate < 1:
            avg_rr    = self.tp_atr_mult / self.sl_atr_mult   # theoretical R/R = 2
            kelly_f   = win_rate - (1 - win_rate) / avg_rr
            kelly_f   = max(kelly_f, 0.0) * 0.5              # half-Kelly for safety
            kelly_qty = (self.equity * kelly_f * leverage) / entry
            qty = min(qty, kelly_qty)

        return round(max(qty, 0.0), 6)

    # ── Gate checks ───────────────────────────────────────────────────────────

    def can_open(self, pair: str) -> Tuple[bool, str]:
        if self.trading_halted():
            return False, "daily loss limit reached"
        if pair in self.open_positions:
            return False, "already have an open position on this pair"
        if len(self.open_positions) >= self.max_open:
            return False, f"max open positions ({self.max_open}) reached"
        if self.equity < self.risk_per_trade_abs:
            return False, f"equity (£{self.equity:.2f}) below minimum trade risk (£{self.risk_per_trade_abs})"
        return True, "ok"

    def should_exit(self, pair: str, current_price: float) -> Optional[str]:
        pos = self.open_positions.get(pair)
        if not pos:
            return None
        pos.candles_held += 1
        if pos.candles_held < self.min_holding:
            return None
        if pos.side == "long":
            if current_price <= pos.stop_loss:   return "stop_loss"
            if current_price >= pos.take_profit: return "take_profit"
        else:
            if current_price >= pos.stop_loss:   return "stop_loss"
            if current_price <= pos.take_profit: return "take_profit"
        return None

    # ── Position registry ─────────────────────────────────────────────────────

    def open_position(self, pos: Position):
        self.open_positions[pos.pair] = pos
        logger.info(
            "📈 OPEN  %s %s @ £%.4f | SL=£%.4f | TP=£%.4f | R/R=%.2f | Qty=%.5f | Lev=%dx",
            pos.side.upper(), pos.pair, pos.entry_price,
            pos.stop_loss, pos.take_profit, pos.rr_ratio, pos.quantity, pos.leverage,
        )

    def close_position(self, pair: str, exit_price: float) -> Optional[dict]:
        pos = self.open_positions.pop(pair, None)
        if not pos:
            return None

        # PnL in price terms
        if pos.side == "long":
            price_move = exit_price - pos.entry_price
        else:
            price_move = pos.entry_price - exit_price

        # Actual £ PnL: qty × price_move × leverage
        pnl_usdt = pos.quantity * price_move * pos.leverage
        pnl_pct  = price_move / pos.entry_price * pos.leverage

        self.equity = max(self.equity + pnl_usdt, 0.0)

        emoji = "🟢" if pnl_usdt >= 0 else "🔴"
        logger.info(
            "%s CLOSE %s @ £%.4f | PnL=%+.4f%% (%+.2f £) | Equity=£%.2f",
            emoji, pair, exit_price, pnl_pct * 100, pnl_usdt, self.equity,
        )

        return {
            "pair":         pair,
            "side":         pos.side,
            "entry_price":  pos.entry_price,
            "exit_price":   exit_price,
            "quantity":     pos.quantity,
            "leverage":     pos.leverage,
            "pnl_pct":      pnl_pct,
            "pnl_usdt":     pnl_usdt,
            "candles_held": pos.candles_held,
        }
