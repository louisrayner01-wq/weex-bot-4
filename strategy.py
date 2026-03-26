"""
strategy.py  —  Self-improving ensemble ML strategy

Architecture
────────────
1. THREE models vote together (soft-vote majority):
     • Random Forest       — robust, handles non-linear patterns well
     • Gradient Boosting   — corrects errors iteratively, strong on tabular data
     • Extra-Trees         — adds randomness, reduces overfitting
   Wrapped in CalibratedClassifierCV so the probabilities it emits are
   ACTUALLY reliable percentages (not just relative scores).

2. Expected Value gate
   Before every trade we calculate:
     EV = win_rate × avg_win_pct − loss_rate × avg_loss_pct
   We only enter if EV > min_ev AND historical win_rate > min_win_rate.
   This is the mathematical edge that separates systematic trading from gambling.

3. Dynamic retraining schedule
   The bot retrains MORE often early on so it learns fast:
     Trades  0-14  →  retrain every  5 completed trades
     Trades 15-39  →  retrain every 10
     Trades 40-99  →  retrain every 20
     Trades 100+   →  retrain every 40
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional, List, Dict

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from indicators import FEATURE_COLS, compute_features

logger = logging.getLogger(__name__)

BUY  =  1
HOLD =  0
SELL = -1

# ─────────────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────────────

def label_candles(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.005) -> pd.Series:
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(HOLD, index=df.index)
    labels[future_return >  threshold] = BUY
    labels[future_return < -threshold] = SELL
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Trade outcome statistics
# ─────────────────────────────────────────────────────────────────────────────

class TradeStats:
    """Rolling statistics used for the Expected Value gate."""

    def __init__(self, min_trades: int = 10):
        self.min_trades = min_trades
        self._history: List[Dict] = []

    def record(self, outcome: dict):
        self._history.append(outcome)

    def for_pair(self, symbol: Optional[str] = None) -> List[Dict]:
        if symbol is None:
            return self._history
        return [t for t in self._history if t.get("pair") == symbol]

    def ev_and_winrate(self, symbol: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (expected_value, win_rate, avg_rr) for the given pair (or overall).
        Returns (None, None, None) if there is not enough data.
        EV = win_rate × avg_win − loss_rate × avg_loss  (in fractional terms)
        """
        history = self.for_pair(symbol)
        if len(history) < self.min_trades:
            return None, None, None

        pnls = [t["pnl_pct"] for t in history]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate  = len(wins)  / len(pnls)
        avg_win   = float(np.mean(wins))   if wins   else 0.0
        avg_loss  = float(np.mean(losses)) if losses else 0.0  # negative number

        ev = win_rate * avg_win + (1 - win_rate) * avg_loss   # avg_loss already negative

        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        return ev, win_rate, avg_rr

    def rank_pairs(self) -> Dict[str, float]:
        pairs = {t["pair"] for t in self._history}
        ranks = {}
        for pair in pairs:
            ev, wr, _ = self.ev_and_winrate(pair)
            if ev is not None:
                ranks[pair] = ev
        return dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))

    def summary_str(self, symbol: Optional[str] = None) -> str:
        ev, wr, rr = self.ev_and_winrate(symbol)
        history = self.for_pair(symbol)
        if ev is None:
            return f"({len(history)}/{self.min_trades} trades needed for EV calc)"
        return (f"trades={len(history)}  win_rate={wr*100:.1f}%  "
                f"avg_rr={rr:.2f}  EV={ev*100:+.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic retraining schedule
# ─────────────────────────────────────────────────────────────────────────────

def retrain_frequency(total_trades: int, stages: List[Dict]) -> int:
    """
    Look up how many completed trades between retrains,
    based on the `retrain_stages` list in config.
    """
    freq = stages[-1]["retrain_every"]
    for stage in reversed(stages):
        if total_trades >= stage["after_trades"]:
            freq = stage["retrain_every"]
            break
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble() -> CalibratedClassifierCV:
    """
    Three diverse models combined with soft voting, then wrapped in
    isotonic calibration so predicted probabilities match real frequencies.
    """
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft",           # use averaged probabilities, not majority vote
        weights=[1.5, 2.0, 1.0], # GB gets more weight (typically more accurate)
    )

    # Isotonic calibration corrects the probability scale so
    # "65% confident" actually means we're right ~65% of the time.
    return CalibratedClassifierCV(ensemble, method="isotonic", cv=3)


# ─────────────────────────────────────────────────────────────────────────────
# Main strategy class
# ─────────────────────────────────────────────────────────────────────────────

class TradingStrategy:

    def __init__(self, cfg: dict):
        sc = cfg["strategy"]
        self.buy_threshold   = sc["buy_threshold"]
        self.sell_threshold  = sc["sell_threshold"]
        self.retrain_stages  = sc["retrain_stages"]
        self.min_samples     = sc["min_training_samples"]
        self.label_horizon   = sc["label_horizon_candles"]
        self.label_threshold = sc["label_move_threshold"]
        self.min_ev          = sc["min_ev"]
        self.min_win_rate    = sc["min_win_rate"]
        self.min_ev_trades   = sc["min_ev_trades"]
        self.models_dir      = cfg["logging"]["models_dir"]

        os.makedirs(self.models_dir, exist_ok=True)

        self.model: Optional[CalibratedClassifierCV] = None
        self.scaler = StandardScaler()
        self.selected_features: List[str] = list(FEATURE_COLS)

        self.total_trades    = 0
        self.trades_since_retrain = 0
        self.stats = TradeStats(min_trades=self.min_ev_trades)
        self.feature_importance: Dict[str, float] = {}

        self._try_load_model()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _model_path(self):  return os.path.join(self.models_dir, "ensemble_model.joblib")
    def _meta_path(self):   return os.path.join(self.models_dir, "meta.joblib")

    def _try_load_model(self):
        if os.path.exists(self._model_path()):
            try:
                self.model             = joblib.load(self._model_path())
                meta                   = joblib.load(self._meta_path())
                self.scaler            = meta["scaler"]
                self.selected_features = meta["features"]
                self.total_trades      = meta.get("total_trades", 0)
                self.stats._history    = meta.get("trade_history", [])
                logger.info("✅ Loaded saved model (trained on %d trades)", self.total_trades)
            except Exception as exc:
                logger.warning("Could not load model (%s) — will train fresh.", exc)

    def _save_model(self):
        joblib.dump(self.model, self._model_path())
        joblib.dump({
            "scaler":        self.scaler,
            "features":      self.selected_features,
            "total_trades":  self.total_trades,
            "trade_history": self.stats._history,
        }, self._meta_path())
        logger.info("💾 Model saved (%d total trades)", self.total_trades)

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_features(df)
        available = [c for c in self.selected_features if c in df.columns]
        X = df[available].replace([np.inf, -np.inf], np.nan)
        return X.dropna()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame):
        df = compute_features(df)
        labels = label_candles(df, self.label_horizon, self.label_threshold)

        available = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = df[available].replace([np.inf, -np.inf], np.nan)
        y = labels

        valid = X_raw.notna().all(axis=1) & y.notna()
        X_raw, y = X_raw[valid].iloc[:-self.label_horizon], y[valid].iloc[:-self.label_horizon]

        if len(X_raw) < self.min_samples:
            logger.info("⏳ Need %d samples to train (have %d). Skipping.",
                        self.min_samples, len(X_raw))
            return

        # ── Scale ─────────────────────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X_raw)
        X_df     = pd.DataFrame(X_scaled, columns=available)

        # ── Feature selection: drop bottom-10% importance features ────────────
        prelim_rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                           class_weight="balanced", random_state=42, n_jobs=-1)
        prelim_rf.fit(X_df, y)
        importances = dict(zip(available, prelim_rf.feature_importances_))
        threshold   = np.percentile(list(importances.values()), 10)
        self.selected_features = [f for f, imp in importances.items() if imp >= threshold]
        self.feature_importance = importances
        logger.info("Feature selection: %d → %d features kept",
                    len(available), len(self.selected_features))

        X_sel = X_df[self.selected_features]

        # ── Build + fit calibrated ensemble ──────────────────────────────────
        self.model = build_ensemble()
        self.model.fit(X_sel, y)

        # ── Cross-val accuracy ────────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_sel, y, cv=cv, scoring="accuracy")
        logger.info("🎯 Model trained | samples=%d | CV accuracy=%.2f±%.2f | features=%d",
                    len(X_sel), scores.mean(), scores.std(), len(self.selected_features))

        # ── Log top features ──────────────────────────────────────────────────
        top5 = sorted(self.selected_features,
                       key=lambda f: importances.get(f, 0), reverse=True)[:5]
        logger.info("Top features: %s",
                    {f: round(importances[f], 3) for f in top5})

        self._save_model()
        self.trades_since_retrain = 0

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[int, float, float]:
        """
        Returns (signal, buy_probability, sell_probability).
        signal: BUY (1), HOLD (0), SELL (-1)
        Probabilities are calibrated — 0.65 really does mean ~65% likely.
        """
        if self.model is None:
            return HOLD, 0.0, 0.0

        X = self._prepare_X(df)
        available = [c for c in self.selected_features if c in X.columns]
        if not available or X.empty:
            return HOLD, 0.0, 0.0

        last = X[available].iloc[[-1]]
        try:
            X_scaled = self.scaler.transform(
                pd.DataFrame(last.values, columns=available,
                             index=last.index).reindex(columns=available)
            )
            X_sel = pd.DataFrame(X_scaled, columns=available)[self.selected_features
                                                               if all(f in available for f in self.selected_features)
                                                               else available]
        except Exception as exc:
            logger.debug("Prediction transform error: %s", exc)
            return HOLD, 0.0, 0.0

        proba   = self.model.predict_proba(X_sel)[0]
        classes = list(self.model.classes_)

        buy_p  = float(proba[classes.index(BUY)])  if BUY  in classes else 0.0
        sell_p = float(proba[classes.index(SELL)]) if SELL in classes else 0.0

        if buy_p >= self.buy_threshold:
            return BUY,  buy_p, sell_p
        if sell_p >= self.buy_threshold:
            return SELL, buy_p, sell_p
        return HOLD, buy_p, sell_p

    # ── Expected Value gate ───────────────────────────────────────────────────

    def trade_is_worth_it(self, symbol: str) -> Tuple[bool, str]:
        """
        Mathematical go/no-go for each trade.
        Returns (should_trade, reason_string).

        Checks:
          1. Enough historical trades to evaluate (if not, we trade freely to gather data)
          2. Win rate >= min_win_rate
          3. Expected Value > min_ev
        """
        ev, win_rate, avg_rr = self.stats.ev_and_winrate(symbol)

        # Not enough data yet — trade to gather it
        if ev is None:
            n = len(self.stats.for_pair(symbol))
            return True, f"Gathering data ({n}/{self.min_ev_trades} trades)"

        # Win rate check
        if win_rate < self.min_win_rate:
            return False, (f"Win rate {win_rate*100:.1f}% < minimum {self.min_win_rate*100:.0f}% "
                           f"— skipping until performance improves")

        # EV check
        if ev < self.min_ev:
            return False, (f"EV {ev*100:+.3f}% < minimum {self.min_ev*100:.2f}% "
                           f"— not enough mathematical edge")

        return True, (f"✅ EV={ev*100:+.3f}%  win_rate={win_rate*100:.1f}%  avg_rr={avg_rr:.2f}")

    # ── Outcome recording (triggers retraining) ───────────────────────────────

    def record_outcome(self, outcome: dict, df: pd.DataFrame):
        """Call whenever a trade closes. Triggers retraining when due."""
        self.stats.record(outcome)
        self.total_trades        += 1
        self.trades_since_retrain += 1

        ev, wr, rr = self.stats.ev_and_winrate(outcome.get("pair"))
        pnl = outcome.get("pnl_pct", 0)
        logger.info("📝 Trade #%d  %s  PnL=%+.2f%%  │  Overall: %s",
                    self.total_trades,
                    outcome.get("pair", "?"),
                    pnl * 100,
                    self.stats.summary_str(outcome.get("pair")))

        freq = retrain_frequency(self.total_trades, self.retrain_stages)
        if self.trades_since_retrain >= freq:
            logger.info("🔄 Retraining triggered (every %d trades at this stage)…", freq)
            self.train(df)

    # ── Pair ranking for dynamic allocation ───────────────────────────────────

    def best_pairs(self) -> Dict[str, float]:
        return self.stats.rank_pairs()
