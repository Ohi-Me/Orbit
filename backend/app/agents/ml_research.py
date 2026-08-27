"""
ML Research Agent
=================
Trains and compares models on the factor panel under PURGED, EMBARGOED
walk-forward validation, and -- critically -- returns the out-of-sample
predictions themselves so the Backtesting Agent can trade the model that was
actually validated.

THE TWO THINGS THIS FIXES FROM THE ORIGINAL BUILD
-------------------------------------------------
1. FOLD-BOUNDARY LABEL LEAKAGE. The original split train and test at adjacent
   dates. But the label is a 21-day forward return, so a training sample on
   the last day of the training window has a label that resolves 21 days INTO
   the test window. The model was therefore trained on the outcome it was
   about to be tested on. `purged_walk_forward_folds` drops training samples
   whose label window overlaps the test period (purging) and additionally
   skips the first days of the test window (embargo), following the standard
   remedy for overlapping-label cross-validation.

2. THE MODEL AND THE STRATEGY WERE DISCONNECTED. The original pipeline
   validated logistic regression and XGBoost, then backtested a completely
   separate hand-weighted factor blend that no model had any part in. The ML
   results and the backtest results were about different objects, so the
   platform could not answer its own research question. This agent now emits
   `oos_predictions` -- every test-fold prediction, keyed by (date, ticker,
   model) -- and the Backtesting Agent trades those directly.

Models compared, in increasing complexity, because the question is whether
complexity earns its keep:
    1. Logistic regression   -- linear baseline
    2. XGBoost               -- non-linear tabular
    3. LSTM                  -- sequence, recurrent
    4. Transformer encoder   -- sequence, attention

Every model sees identical folds, identical features, and identical
standardization discipline, so differences are attributable to the model.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.core.stats import (
    effective_sample_size,
    information_coefficient,
    newey_west_tstat,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

from app.models_dl.sequence_models import (
    HAS_TORCH,
    SEQ_LEN,
    build_sequences,
    train_sequence_model,
)

LABEL = "fwd_return_21d"
LABEL_HORIZON = 21


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def purged_walk_forward_folds(
    dates: pd.Series,
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    horizon: int = LABEL_HORIZON,
    embargo: int = 5,
) -> list[dict]:
    """Expanding-window folds with purging and an embargo.

    For each fold:
        train  = [first_date, train_end - horizon]   <- purged
        gap    = (train_end - horizon, test_start)   <- discarded entirely
        test   = [test_start + embargo, test_end]

    The purge width is the label horizon because that is exactly how far a
    training label reaches forward. The embargo is a smaller additional
    buffer against serial correlation in features that straddle the boundary.
    """
    uniq = pd.DatetimeIndex(sorted(pd.Series(dates).dropna().unique()))
    n = len(uniq)
    if n < 60:
        return []

    min_train = int(n * min_train_frac)
    remaining = n - min_train
    fold_size = max(remaining // max(n_folds, 1), 1)

    folds = []
    train_end_idx = min_train
    for k in range(n_folds):
        test_start_idx = min(train_end_idx + embargo, n - 1)
        test_end_idx = min(train_end_idx + fold_size, n)
        if test_start_idx >= test_end_idx:
            break

        purged_train_end_idx = train_end_idx - horizon
        if purged_train_end_idx <= 20:  # nothing meaningful left to train on
            train_end_idx = test_end_idx
            continue

        folds.append(
            {
                "fold": k,
                "train_start": uniq[0],
                "train_end": uniq[purged_train_end_idx - 1],
                "test_start": uniq[test_start_idx],
                "test_end": uniq[test_end_idx - 1],
                "purged_days": horizon,
                "embargo_days": embargo,
                "raw_train_end": uniq[train_end_idx - 1],
            }
        )
        train_end_idx = test_end_idx

    return folds


def _prep(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    df = panel.dropna(subset=factor_cols + [LABEL]).copy()
    df["label"] = (df[LABEL] > 0).astype(int)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------
def _evaluate_fold(
    test_df: pd.DataFrame, proba: np.ndarray, top_quantile: float = 0.67
) -> dict:
    """Score one fold's predictions.

    Reports both a naive and a HAC t-statistic on the implied long/short
    signal return. The naive one is kept deliberately and shown next to the
    corrected one, because the gap between them is the single clearest
    illustration of why overlapping-window significance testing misleads.
    """
    y_true = test_df["label"].to_numpy()
    pred = (proba > 0.5).astype(int)

    acc = float(accuracy_score(y_true, pred))
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")

    # BASE RATE -- reported because accuracy alone is close to meaningless on
    # an unbalanced directional label. If 67% of 21-day forward returns in a
    # test fold are positive, a model that predicts "up" every time scores
    # 0.67 and looks excellent while carrying no information at all. The lift
    # over the base rate, and the fraction of predictions that are positive,
    # are what expose that.
    base_rate = float(max(y_true.mean(), 1 - y_true.mean())) if len(y_true) else float("nan")
    pred_positive_rate = float(pred.mean()) if len(pred) else float("nan")

    # Cross-sectional IC: rank correlation of the score against the realized
    # forward return, computed per date and averaged.
    tmp = test_df.copy()
    tmp["proba"] = proba
    ics = []
    for _, grp in tmp.groupby("date"):
        if len(grp) < 3:
            continue
        ic = information_coefficient(grp["proba"].to_numpy(), grp[LABEL].to_numpy())
        if ic is not None:
            ics.append(ic)
    mean_ic = float(np.mean(ics)) if ics else None

    # Implied long/short signal return, formed cross-sectionally per date so
    # it reflects a tradeable ranking rather than a pooled threshold.
    per_date_returns = []
    for _, grp in tmp.groupby("date"):
        if len(grp) < 3:
            continue
        hi = grp["proba"].quantile(top_quantile)
        lo = grp["proba"].quantile(1 - top_quantile)
        longs = grp.loc[grp["proba"] >= hi, LABEL]
        shorts = grp.loc[grp["proba"] <= lo, LABEL]
        if len(longs) == 0 or len(shorts) == 0:
            continue
        per_date_returns.append(float(longs.mean() - shorts.mean()))

    signal_ret = np.array(per_date_returns)
    nw = newey_west_tstat(signal_ret, lags=LABEL_HORIZON - 1) if len(signal_ret) >= 8 else None

    return {
        "n_test": int(len(test_df)),
        "accuracy": round(acc, 4),
        "base_rate": round(base_rate, 4) if np.isfinite(base_rate) else None,
        "accuracy_lift_over_base_rate": round(acc - base_rate, 4) if np.isfinite(base_rate) else None,
        "predicted_positive_rate": round(pred_positive_rate, 4) if np.isfinite(pred_positive_rate) else None,
        "degenerate_predictor": bool(
            np.isfinite(pred_positive_rate) and (pred_positive_rate > 0.95 or pred_positive_rate < 0.05)
        ),
        "auc": round(auc, 4) if np.isfinite(auc) else None,
        "information_coefficient": round(mean_ic, 4) if mean_ic is not None else None,
        "signal_mean_return": round(float(signal_ret.mean()), 5) if len(signal_ret) else None,
        "signal_t_stat": nw["t_stat"] if nw else None,
        "signal_p_value": nw["p_value"] if nw else None,
        "signal_naive_t_stat": nw["naive_t_stat"] if nw else None,
        "t_inflation_vs_naive": nw["inflation_factor"] if nw else None,
        "n_signal_periods": int(len(signal_ret)),
    }


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
def run_ml_research_agent(
    panel: pd.DataFrame,
    factor_cols: list[str],
    n_folds: int = 4,
    embargo: int = 5,
    include_deep_learning: bool = True,
    seed: int = 42,
    mlflow_run=None,
) -> dict:
    """Walk-forward model comparison. Returns metrics AND out-of-sample scores.

    `oos_predictions` is the contract with the Backtesting Agent: a long
    frame of (date, ticker, model, score) covering every test fold, which is
    what makes the backtest an evaluation of a validated model rather than of
    an unrelated hand-tuned formula.
    """
    if not factor_cols:
        return {"status": "no_factors", "note": "No factors passed the coverage gate."}

    df = _prep(panel, factor_cols)
    if len(df) < 250:
        return {
            "status": "insufficient_data",
            "n_usable_rows": int(len(df)),
            "note": f"Only {len(df)} complete rows after factor warm-up. Walk-forward "
            "evaluation needs at least 250. Lengthen the history or drop "
            "long-warm-up factors (momentum_12_1 costs ~273 rows per ticker).",
        }

    folds = purged_walk_forward_folds(
        df["date"], n_folds=n_folds, horizon=LABEL_HORIZON, embargo=embargo
    )
    if not folds:
        return {"status": "insufficient_data", "note": "Could not build purged folds from this history."}

    models: list[str] = ["logistic_regression"]
    if HAS_XGB:
        models.append("xgboost")
    if include_deep_learning and HAS_TORCH:
        models.extend(["lstm", "transformer"])

    per_model: dict[str, list[dict]] = {m: [] for m in models}
    oos_rows: list[pd.DataFrame] = []
    training_info: dict[str, list] = {m: [] for m in models}
    timings: dict[str, float] = {m: 0.0 for m in models}

    for fold in folds:
        train_mask = (df["date"] >= fold["train_start"]) & (df["date"] <= fold["train_end"])
        test_mask = (df["date"] >= fold["test_start"]) & (df["date"] <= fold["test_end"])
        train, test = df[train_mask], df[test_mask]

        if len(train) < 100 or len(test) < 20 or train["label"].nunique() < 2:
            continue

        # ---- tabular models ------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[factor_cols])
        X_test = scaler.transform(test[factor_cols])
        y_train = train["label"].to_numpy()

        for name in [m for m in models if m in ("logistic_regression", "xgboost")]:
            t0 = time.time()
            if name == "logistic_regression":
                model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            else:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=2,
                )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            timings[name] += time.time() - t0

            metrics = _evaluate_fold(test, proba)
            metrics.update(
                {
                    "fold": fold["fold"],
                    "train_period": [str(fold["train_start"].date()), str(fold["train_end"].date())],
                    "test_period": [str(fold["test_start"].date()), str(fold["test_end"].date())],
                    "n_train": int(len(train)),
                    "purged_days": fold["purged_days"],
                    "embargo_days": fold["embargo_days"],
                }
            )
            per_model[name].append(metrics)

            oos_rows.append(
                pd.DataFrame(
                    {
                        "date": test["date"].to_numpy(),
                        "ticker": test["ticker"].to_numpy(),
                        "model": name,
                        "score": proba,
                        "fold": fold["fold"],
                        LABEL: test[LABEL].to_numpy(),
                    }
                )
            )

            if name == "logistic_regression":
                training_info[name].append(
                    {"fold": fold["fold"], "n_features": len(factor_cols), "solver": "lbfgs"}
                )
            else:
                training_info[name].append(
                    {
                        "fold": fold["fold"],
                        "feature_importance": dict(
                            sorted(
                                zip(factor_cols, [float(v) for v in model.feature_importances_]),
                                key=lambda kv: -kv[1],
                            )[:8]
                        ),
                    }
                )

        # ---- sequence models -----------------------------------------------
        seq_models = [m for m in models if m in ("lstm", "transformer")]
        if seq_models:
            # FAIRNESS: build sequences over the CONTIGUOUS history up to the
            # end of this test fold, then slice train/test by the sequence's
            # END date. Building them from the test slice alone would silently
            # discard the first SEQ_LEN-1 days of every test window -- the
            # sequence models would then be scored on a smaller, later subset
            # than the tabular models and the comparison would be meaningless.
            # Using pre-test days *inside* a test sample's input window is not
            # leakage: those observations are strictly in the past relative to
            # the prediction date, exactly as they would be live.
            hist = df[df["date"] <= fold["test_end"]]
            Xall, yall, rall, idx_all = build_sequences(hist, factor_cols, seq_len=SEQ_LEN)

            if len(Xall):
                seq_dates = pd.DatetimeIndex([d for d, _ in idx_all])
                tr_sel = seq_dates <= fold["train_end"]
                te_sel = (seq_dates >= fold["test_start"]) & (seq_dates <= fold["test_end"])
            else:
                tr_sel = te_sel = np.zeros(0, dtype=bool)

            Xtr, ytr = (Xall[tr_sel], yall[tr_sel]) if len(Xall) else (Xall, yall)
            Xte = Xall[te_sel] if len(Xall) else Xall
            rte = rall[te_sel] if len(Xall) else rall
            idx_te = [idx_all[i] for i in np.where(te_sel)[0]] if len(Xall) else []

            if len(Xtr) >= 120 and len(Xte) >= 20:
                seq_test_df = pd.DataFrame(
                    {
                        "date": [d for d, _ in idx_te],
                        "ticker": [t for _, t in idx_te],
                        LABEL: rte,
                        "label": (rte > 0).astype(int),
                    }
                )
                for name in seq_models:
                    t0 = time.time()
                    try:
                        proba, info = train_sequence_model(
                            "lstm" if name == "lstm" else "transformer",
                            Xtr,
                            ytr,
                            Xte,
                            seed=seed + fold["fold"],
                        )
                    except Exception as e:
                        per_model[name].append(
                            {"fold": fold["fold"], "status": "failed", "error": f"{type(e).__name__}: {e}"}
                        )
                        continue
                    timings[name] += time.time() - t0

                    metrics = _evaluate_fold(seq_test_df, proba)
                    metrics.update(
                        {
                            "fold": fold["fold"],
                            "train_period": [str(fold["train_start"].date()), str(fold["train_end"].date())],
                            "test_period": [str(fold["test_start"].date()), str(fold["test_end"].date())],
                            "n_train": int(len(Xtr)),
                            "purged_days": fold["purged_days"],
                            "embargo_days": fold["embargo_days"],
                        }
                    )
                    per_model[name].append(metrics)
                    training_info[name].append({"fold": fold["fold"], **info})

                    oos_rows.append(
                        pd.DataFrame(
                            {
                                "date": seq_test_df["date"].to_numpy(),
                                "ticker": seq_test_df["ticker"].to_numpy(),
                                "model": name,
                                "score": proba,
                                "fold": fold["fold"],
                                LABEL: seq_test_df[LABEL].to_numpy(),
                            }
                        )
                    )
            else:
                for name in seq_models:
                    per_model[name].append(
                        {
                            "fold": fold["fold"],
                            "status": "skipped",
                            "detail": f"Not enough sequence samples (train={len(Xtr)}, test={len(Xte)}); "
                            f"a {SEQ_LEN}-day window needs a longer contiguous history per ticker.",
                        }
                    )

    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame(
        columns=["date", "ticker", "model", "score", "fold", LABEL]
    )

    # ---- summaries ---------------------------------------------------------
    summary = []
    for name in models:
        folds_ok = [f for f in per_model[name] if f.get("accuracy") is not None]
        if not folds_ok:
            summary.append({"model": name, "n_folds_evaluated": 0, "status": "no_successful_folds"})
            continue
        accs = [f["accuracy"] for f in folds_ok]
        aucs = [f["auc"] for f in folds_ok if f["auc"] is not None]
        ics = [f["information_coefficient"] for f in folds_ok if f["information_coefficient"] is not None]
        prets = [f["signal_mean_return"] for f in folds_ok if f["signal_mean_return"] is not None]
        lifts = [f["accuracy_lift_over_base_rate"] for f in folds_ok if f.get("accuracy_lift_over_base_rate") is not None]
        bases = [f["base_rate"] for f in folds_ok if f.get("base_rate") is not None]
        n_degenerate = sum(1 for f in folds_ok if f.get("degenerate_predictor"))
        summary.append(
            {
                "model": name,
                "mean_accuracy": round(float(np.mean(accs)), 4),
                "std_accuracy": round(float(np.std(accs)), 4),
                "mean_base_rate": round(float(np.mean(bases)), 4) if bases else None,
                "mean_accuracy_lift": round(float(np.mean(lifts)), 4) if lifts else None,
                "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
                "mean_information_coefficient": round(float(np.mean(ics)), 4) if ics else None,
                "mean_signal_return": round(float(np.mean(prets)), 5) if prets else None,
                "n_degenerate_folds": n_degenerate,
                "n_folds_evaluated": len(folds_ok),
                "train_seconds": round(timings[name], 2),
                "status": "ok",
            }
        )

    # Rank by IC where available -- it measures ranking skill, which is what a
    # cross-sectional long/short strategy actually monetizes. Accuracy on a
    # near-balanced binary label is a much weaker discriminator.
    ranked = [s for s in summary if s.get("status") == "ok"]
    ranked.sort(
        key=lambda s: (
            s.get("mean_information_coefficient") if s.get("mean_information_coefficient") is not None else -9,
            s.get("mean_auc") or 0,
        ),
        reverse=True,
    )
    best_model = ranked[0]["model"] if ranked else None
    baseline = next((s for s in summary if s["model"] == "logistic_regression"), None)

    # A model is only credited with beating the field if it clears THREE bars,
    # not one. Ranking best-of-N is trivially satisfiable when every entrant is
    # bad: the original comparison would have crowned a model with a negative
    # information coefficient simply because the linear baseline was worse.
    #   1. positive information coefficient (it ranks better than chance)
    #   2. positive accuracy lift over the base rate (it beats "always up")
    #   3. a better IC than the linear baseline
    beats_baseline = None
    model_verdict = "no_models_evaluated"
    if best_model and ranked:
        best = ranked[0]
        m_ic = best.get("mean_information_coefficient")
        m_lift = best.get("mean_accuracy_lift")
        b_ic = baseline.get("mean_information_coefficient") if baseline and baseline.get("status") == "ok" else None

        has_ic = m_ic is not None and m_ic > 0.02
        has_lift = m_lift is not None and m_lift > 0.01
        beats_linear = b_ic is None or (m_ic is not None and m_ic > b_ic)
        beats_baseline = bool(has_ic and has_lift and beats_linear)

        if beats_baseline:
            model_verdict = "candidate_signal"
        elif has_ic or has_lift:
            model_verdict = "marginal_no_usable_edge"
        else:
            model_verdict = "no_edge_detected"

    verdict_detail = {
        "candidate_signal": "The leading model clears a positive information "
        "coefficient, a positive accuracy lift over the base rate, and the linear "
        "baseline. Worth carrying into the backtest as a candidate -- not a "
        "confirmed edge until costs and the Critic's checks are applied.",
        "marginal_no_usable_edge": "The leading model clears one of the two "
        "thresholds but not both. Ranking best-of-N is not evidence of skill when "
        "the field is weak; treat this as no usable edge.",
        "no_edge_detected": "No model produced a positive information coefficient "
        "AND a positive lift over the base rate. The honest conclusion is that "
        "these factors do not predict direction on this universe and horizon. "
        "Reporting that is a result, not a failure.",
        "no_models_evaluated": "No model completed enough folds to evaluate.",
    }.get(model_verdict, "")

    n_names = int(panel["ticker"].nunique())
    avg_corr = _average_cross_correlation(panel)
    ess = effective_sample_size(len(df), n_names, avg_corr, overlap=LABEL_HORIZON)

    return {
        "status": "ok",
        "n_usable_rows": int(len(df)),
        "n_factors": len(factor_cols),
        "factors_used": factor_cols,
        "models_compared": models,
        "xgboost_available": HAS_XGB,
        "torch_available": HAS_TORCH,
        "folds": [
            {
                "fold": f["fold"],
                "train": [str(f["train_start"].date()), str(f["train_end"].date())],
                "test": [str(f["test_start"].date()), str(f["test_end"].date())],
                "purged_days": f["purged_days"],
                "embargo_days": f["embargo_days"],
            }
            for f in folds
        ],
        "per_model_folds": [{"model": m, "folds": per_model[m]} for m in models],
        "summary": summary,
        "best_model": best_model,
        "best_model_beats_baseline": beats_baseline,
        "model_verdict": model_verdict,
        "model_verdict_detail": verdict_detail,
        "training_info": training_info,
        "effective_sample_size": ess,
        "oos_predictions": oos,
        "validation_scheme": {
            "type": "purged_embargoed_expanding_walk_forward",
            "label_horizon_days": LABEL_HORIZON,
            "purge_days": LABEL_HORIZON,
            "embargo_days": embargo,
            "note": "Training samples whose 21-day label window overlaps the test "
            "period are removed (purging); the first days of each test window are "
            "skipped (embargo). Without purging, a model is trained on the outcome "
            "it is then tested on.",
        },
    }


def _average_cross_correlation(panel: pd.DataFrame) -> float:
    """Mean pairwise correlation of forward returns across the universe.

    Feeds the effective-sample-size calculation: six names in one market are
    not six independent observations, and the reported n should say so.
    """
    try:
        wide = panel.pivot_table(index="date", columns="ticker", values=LABEL)
        corr = wide.corr().to_numpy()
        n = corr.shape[0]
        if n < 2:
            return 0.0
        off = corr[~np.eye(n, dtype=bool)]
        off = off[np.isfinite(off)]
        return float(np.clip(off.mean(), 0.0, 0.99)) if len(off) else 0.0
    except Exception:
        return 0.0
