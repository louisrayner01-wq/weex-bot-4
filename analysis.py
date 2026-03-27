"""
analysis.py
═══════════
Reads historical OHLCV CSVs and answers three questions:

  1. Which single timeframe produces the most accurate signals?
  2. Which higher-timeframe filter (confluence) improves win rate most?
  3. Which features matter most — and should the bot focus on?

Outputs
───────
  /data/analysis_results.json   — machine-readable results loaded by strategy.py
  Console report                — human-readable summary printed at startup

How it works
────────────
For each timeframe we:
  • Compute all 22 technical indicators (same as live trading)
  • Label each candle BUY/HOLD/SELL using the same forward-return method
  • Train a Random Forest with 5-fold cross-validation
  • Record accuracy, per-class precision/recall, and feature importances

For each (signal_tf × filter_tf) confluence pair we:
  • Only count BUY/SELL signals where the higher-TF trend agrees
    (price above EMA21 = bullish; below = bearish)
  • Compare precision and win-rate with vs. without the filter
  • Record the improvement

Results are ranked and saved so strategy.py can auto-configure itself.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score

from indicators import compute_features, FEATURE_COLS
from data_collector import TF_LABELS, TIMEFRAMES, SYMBOLS

logger = logging.getLogger(__name__)

RESULTS_FILE = "analysis_results.json"
MIN_SAMPLES  = 200    # minimum candles needed to analyse a timeframe
LABEL_HORIZON = 4     # candles forward for labelling
LABEL_THRESH  = 0.005 # 0.5 % move = meaningful signal
BUY, HOLD, SELL = 1, 0, -1


# ── Labelling ─────────────────────────────────────────────────────────────────

def label_candles(df: pd.DataFrame,
                  horizon: int   = LABEL_HORIZON,
                  threshold: float = LABEL_THRESH) -> pd.Series:
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(HOLD, index=df.index)
    labels[future_ret >  threshold] = BUY
    labels[future_ret < -threshold] = SELL
    return labels


# ── Single-timeframe analysis ─────────────────────────────────────────────────

def analyse_timeframe(df: pd.DataFrame,
                      tf_label: str,
                      symbol: str) -> Optional[Dict[str, Any]]:
    """
    Train and cross-validate a Random Forest on one timeframe.
    Returns a dict of metrics, or None if data is insufficient.
    """
    df = compute_features(df.copy())
    labels = label_candles(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = df[available].copy()
    df_feat["label"] = labels

    # Drop rows with NaN features or the unlabellable tail
    df_feat = df_feat.dropna()
    df_feat = df_feat[df_feat["label"] != HOLD]   # binary: BUY vs SELL only

    if len(df_feat) < MIN_SAMPLES:
        logger.warning("  Skipping %s %s — only %d labelled samples",
                       symbol, tf_label, len(df_feat))
        return None

    X = df_feat[available].values
    y = df_feat["label"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model — lean RF for speed
    rf = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 8,
        min_samples_leaf = 10,
        random_state = 42,
        n_jobs       = -1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated accuracy
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
    accuracy  = float(np.mean(cv_scores))

    # Fit on full set for feature importances
    rf.fit(X_scaled, y)

    importances = dict(zip(available, rf.feature_importances_.tolist()))
    top_features = sorted(importances, key=importances.get, reverse=True)[:12]

    # Class balance
    buys  = int((y == BUY).sum())
    sells = int((y == SELL).sum())

    result = {
        "symbol":       symbol,
        "timeframe":    tf_label,
        "samples":      len(df_feat),
        "buy_count":    buys,
        "sell_count":   sells,
        "cv_accuracy":  round(accuracy, 4),
        "cv_std":       round(float(np.std(cv_scores)), 4),
        "feature_importance": {k: round(v, 6) for k, v in
                               sorted(importances.items(),
                                      key=lambda x: x[1], reverse=True)},
        "top_features": top_features,
    }
    logger.info("  %s %s  |  acc=%.3f±%.3f  |  samples=%d  |  top=%s",
                symbol, tf_label, accuracy, np.std(cv_scores),
                len(df_feat), top_features[:3])
    return result


# ── Confluence analysis ───────────────────────────────────────────────────────

def _higher_tf_trend(high_df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of trend labels (+1 bullish / -1 bearish / 0 neutral)
    for a higher-timeframe DataFrame, based on EMA21 and MACD direction.
    """
    high_df = compute_features(high_df.copy())

    trend = pd.Series(0, index=high_df.index)
    if "ema_21" in high_df.columns:
        bullish_ema = high_df["close"] > high_df["ema_21"]
        bearish_ema = high_df["close"] < high_df["ema_21"]
        trend[bullish_ema] =  1
        trend[bearish_ema] = -1

    # MACD confirmation (secondary vote — both must agree)
    if "macd_diff" in high_df.columns:
        macd_bull = high_df["macd_diff"] > 0
        macd_bear = high_df["macd_diff"] < 0
        # Require both EMA and MACD to agree for a strong trend
        strong_bull = bullish_ema & macd_bull
        strong_bear = bearish_ema & macd_bear
        trend[~strong_bull & ~strong_bear] = 0   # neutral if they disagree

    high_df["_trend"] = trend.values
    return high_df.set_index("timestamp")["_trend"]


def analyse_confluence(signal_df: pd.DataFrame,
                       filter_df: pd.DataFrame,
                       signal_tf: str,
                       filter_tf: str,
                       symbol: str) -> Optional[Dict[str, Any]]:
    """
    Compare win rate on signal_tf with and without a filter_tf trend gate.
    Returns metrics dict.
    """
    signal_df = compute_features(signal_df.copy())
    signal_labels = label_candles(signal_df)
    signal_df["label"] = signal_labels

    # Get higher-TF trend direction indexed by timestamp
    htrend = _higher_tf_trend(filter_df.copy())

    # Merge: for each signal candle, find the most recent higher-TF trend
    signal_df = signal_df.set_index("timestamp")
    # Forward-fill the higher-TF trend onto the signal-TF timestamps
    combined = signal_df.copy()
    combined["htf_trend"] = np.nan
    combined["htf_trend"] = (
        combined["htf_trend"]
        .combine_first(htrend.reindex(combined.index))
    )
    combined["htf_trend"] = combined["htf_trend"].ffill()
    combined = combined.reset_index().dropna(subset=["label", "htf_trend"])
    combined = combined[combined["label"] != HOLD]

    if len(combined) < MIN_SAMPLES:
        return None

    # Baseline: all signals regardless of HTF trend
    all_buys  = combined[combined["label"] == BUY]
    all_sells = combined[combined["label"] == SELL]

    # Confluence-filtered: only signals where HTF trend agrees
    conf_buys  = all_buys[all_buys["htf_trend"] == 1]
    conf_sells = all_sells[all_sells["htf_trend"] == -1]

    def win_rate(df_signals: pd.DataFrame, direction: int) -> float:
        if len(df_signals) == 0:
            return 0.0
        # "Win" = the label itself is BUY/SELL (already filtered)
        # We measure whether the labelled direction actually happened
        # (by definition it did — we use forward precision instead)
        return len(df_signals) / max(len(combined[combined["label"] == direction]), 1)

    baseline_coverage = (len(all_buys) + len(all_sells)) / max(len(combined), 1)
    filtered_coverage = (len(conf_buys) + len(conf_sells)) / max(len(combined), 1)

    # Precision: of BUY signals, how many were correct?
    # We approximate via the signal-TF model accuracy on filtered vs unfiltered
    def label_precision(subset: pd.DataFrame) -> float:
        """Fraction of signals that have a non-trivial feature set."""
        if len(subset) < 20:
            return 0.0
        feats = [c for c in FEATURE_COLS if c in subset.columns]
        X = subset[feats].dropna()
        if len(X) < 20:
            return 0.0
        y = subset.loc[X.index, "label"]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=50, max_depth=6,
                                    random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    baseline_acc = label_precision(combined)

    conf_subset = pd.concat([conf_buys, conf_sells])
    filtered_acc = label_precision(conf_subset) if len(conf_subset) >= 20 else baseline_acc

    improvement = round(filtered_acc - baseline_acc, 4)

    result = {
        "symbol":            symbol,
        "signal_tf":         signal_tf,
        "filter_tf":         filter_tf,
        "baseline_samples":  len(combined),
        "filtered_samples":  len(conf_subset),
        "baseline_accuracy": round(baseline_acc, 4),
        "filtered_accuracy": round(filtered_acc, 4),
        "accuracy_gain":     improvement,
        "signal_coverage":   round(filtered_coverage, 4),
    }
    logger.info("  Confluence %s→%s %s  |  base=%.3f → filtered=%.3f  (+%.3f)  coverage=%.1f%%",
                signal_tf, filter_tf, symbol,
                baseline_acc, filtered_acc, improvement,
                filtered_coverage * 100)
    return result


# ── Master analyser ───────────────────────────────────────────────────────────

class Analyzer:

    def __init__(self, data_dir: str = "/data", results_dir: str = "/data"):
        self.data_dir    = data_dir
        self.results_dir = results_dir
        self.results_path = os.path.join(results_dir, RESULTS_FILE)

    def results_are_fresh(self, max_age_days: int = 7) -> bool:
        if not os.path.exists(self.results_path):
            return False
        age = (time.time() - os.path.getmtime(self.results_path)) / 86400
        return age < max_age_days

    def load_results(self) -> Optional[Dict]:
        if not os.path.exists(self.results_path):
            return None
        with open(self.results_path) as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline and save results.
        Returns the results dict.
        """
        import time as _time
        t0 = _time.time()
        logger.info("═══ Analysis pipeline starting ═══")

        tf_results:    List[Dict] = []
        conf_results:  List[Dict] = []

        for symbol in SYMBOLS:
            logger.info("── Analysing %s ──", symbol)

            # Load all available timeframe data for this symbol
            tf_data: Dict[str, pd.DataFrame] = {}
            for tf in TIMEFRAMES:
                label = TF_LABELS[tf]
                fp = os.path.join(self.data_dir, f"{symbol}_{label}.csv")
                if os.path.exists(fp):
                    df = pd.read_csv(fp, parse_dates=["timestamp"])
                    if len(df) >= MIN_SAMPLES:
                        tf_data[tf] = df
                        logger.info("  Loaded %s %s — %d candles", symbol, label, len(df))
                    else:
                        logger.warning("  Skipping %s %s — insufficient data (%d rows)",
                                       symbol, label, len(df))
                else:
                    logger.warning("  No CSV found for %s %s", symbol, label)

            if not tf_data:
                logger.warning("No data for %s — skipping", symbol)
                continue

            # ── Single-timeframe analysis ─────────────────────────────────────
            for tf, df in tf_data.items():
                res = analyse_timeframe(df, TF_LABELS[tf], symbol)
                if res:
                    tf_results.append(res)

            # ── Confluence analysis ───────────────────────────────────────────
            # Signal timeframes: 5m, 15m, 1h
            # Filter timeframes: always higher than signal
            signal_tfs = [t for t in ["5", "15", "60"] if t in tf_data]
            filter_tfs = [t for t in ["60", "240", "1440"] if t in tf_data]

            for sig_tf in signal_tfs:
                for flt_tf in filter_tfs:
                    # Filter must be a strictly higher timeframe
                    if int(flt_tf) <= int(sig_tf):
                        continue
                    res = analyse_confluence(
                        tf_data[sig_tf], tf_data[flt_tf],
                        TF_LABELS[sig_tf], TF_LABELS[flt_tf],
                        symbol,
                    )
                    if res:
                        conf_results.append(res)

        # ── Aggregate recommendations ─────────────────────────────────────────
        recommendations = self._build_recommendations(tf_results, conf_results)

        results = {
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "duration_seconds":  round(_time.time() - t0, 1),
            "timeframe_results": tf_results,
            "confluence_results": conf_results,
            "recommendations":   recommendations,
        }

        os.makedirs(self.results_dir, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved analysis → %s", self.results_path)

        self._print_report(recommendations, tf_results, conf_results)
        return results

    # ── Recommendations ───────────────────────────────────────────────────────

    def _build_recommendations(self,
                                tf_results: List[Dict],
                                conf_results: List[Dict]) -> Dict[str, Any]:
        if not tf_results:
            return {}

        # Best signal timeframe (highest average CV accuracy)
        tf_acc: Dict[str, List[float]] = {}
        for r in tf_results:
            tf_acc.setdefault(r["timeframe"], []).append(r["cv_accuracy"])
        tf_avg = {tf: float(np.mean(accs)) for tf, accs in tf_acc.items()}
        best_signal_tf = max(tf_avg, key=tf_avg.get)

        # Best confluence pair (highest average accuracy gain)
        conf_by_pair: Dict[str, List[float]] = {}
        for r in conf_results:
            key = f"{r['signal_tf']}+{r['filter_tf']}"
            conf_by_pair.setdefault(key, []).append(r["accuracy_gain"])
        conf_avg = {k: float(np.mean(v)) for k, v in conf_by_pair.items()}
        best_confluence = max(conf_avg, key=conf_avg.get) if conf_avg else None

        # Global top features (average importance across all TF models)
        feat_scores: Dict[str, List[float]] = {}
        for r in tf_results:
            for feat, imp in r.get("feature_importance", {}).items():
                feat_scores.setdefault(feat, []).append(imp)
        global_importance = {
            feat: float(np.mean(scores))
            for feat, scores in feat_scores.items()
        }
        top_features = sorted(global_importance, key=global_importance.get,
                               reverse=True)[:15]

        # Per-symbol best timeframe
        per_symbol: Dict[str, str] = {}
        for symbol in SYMBOLS:
            sym_results = [r for r in tf_results if r["symbol"] == symbol]
            if sym_results:
                best = max(sym_results, key=lambda x: x["cv_accuracy"])
                per_symbol[symbol] = best["timeframe"]

        # Confluence filter recommendation
        best_filter_tf = None
        if best_confluence:
            signal_part, filter_part = best_confluence.split("+")
            if signal_part == best_signal_tf:
                best_filter_tf = filter_part

        return {
            "best_signal_timeframe":  best_signal_tf,
            "best_filter_timeframe":  best_filter_tf,
            "best_confluence_pair":   best_confluence,
            "avg_confluence_gain":    conf_avg.get(best_confluence, 0.0) if best_confluence else 0.0,
            "top_features":           top_features,
            "global_feature_importance": {
                k: round(v, 6) for k, v in
                sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
            },
            "timeframe_accuracy":     {tf: round(v, 4) for tf, v in
                                       sorted(tf_avg.items(), key=lambda x: x[1], reverse=True)},
            "per_symbol_best_tf":     per_symbol,
        }

    # ── Report ────────────────────────────────────────────────────────────────

    def _print_report(self,
                      recs: Dict,
                      tf_results: List[Dict],
                      conf_results: List[Dict]) -> None:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════╗",
            "║              STRATEGY ANALYSIS REPORT                       ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "── Timeframe accuracy (avg across symbols) ─────────────────────",
        ]
        for tf, acc in recs.get("timeframe_accuracy", {}).items():
            bar = "█" * int(acc * 40)
            lines.append(f"  {tf:>5}   {acc:.3f}  {bar}")

        lines += [
            "",
            f"  ✅  Best signal timeframe : {recs.get('best_signal_timeframe', 'N/A')}",
            f"  ✅  Best filter timeframe : {recs.get('best_filter_timeframe', 'N/A')}",
            f"  ✅  Best confluence pair  : {recs.get('best_confluence_pair', 'N/A')}",
            f"  ✅  Avg confluence gain   : +{recs.get('avg_confluence_gain', 0):.3f}",
            "",
            "── Top features (ranked by importance) ─────────────────────────",
        ]
        for i, feat in enumerate(recs.get("top_features", [])[:10], 1):
            imp = recs.get("global_feature_importance", {}).get(feat, 0)
            lines.append(f"  {i:>2}.  {feat:<22}  {imp:.4f}")

        lines += [
            "",
            "── Per-symbol best timeframe ────────────────────────────────────",
        ]
        for sym, tf in recs.get("per_symbol_best_tf", {}).items():
            lines.append(f"  {sym:<12}  →  {tf}")

        lines += [
            "",
            "── Confluence improvement ───────────────────────────────────────",
        ]
        sorted_conf = sorted(conf_results,
                              key=lambda x: x.get("accuracy_gain", 0), reverse=True)[:6]
        for r in sorted_conf:
            gain = r.get("accuracy_gain", 0)
            sign = "+" if gain >= 0 else ""
            lines.append(f"  {r['signal_tf']:>4} + {r['filter_tf']:<4}  {r['symbol']:<12}"
                         f"  {sign}{gain:.3f}  "
                         f"({'▲ better' if gain > 0 else '▼ worse'})")

        lines.append("")
        for line in lines:
            logger.info(line)


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("data", {}).get("data_dir", "/data")
    analyzer = Analyzer(data_dir=data_dir, results_dir=data_dir)
    analyzer.run()
