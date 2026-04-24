from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ethics_experiment.metrics import compute_utility, fairness_metrics


def train_logistic_model(
    train_df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
    seed: int,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    x_train = train_df[list(feature_columns)].to_numpy()
    y_train = train_df[target_column].to_numpy()
    model.fit(x_train, y_train)
    return model


def predict_probabilities(model: Pipeline, df: pd.DataFrame, feature_columns: Iterable[str]) -> np.ndarray:
    x = df[list(feature_columns)].to_numpy()
    probs = model.predict_proba(x)[:, 1]
    return probs


def choose_fair_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    group: np.ndarray,
    utility_weights: Dict[str, float],
    fairness_epsilon: float,
    threshold_grid: np.ndarray,
    spd_epsilon: float | None = None,
) -> Tuple[float, pd.DataFrame, bool]:
    frontier_rows = []

    for threshold in threshold_grid:
        y_pred = (probs >= threshold).astype(int)
        fair = fairness_metrics(y_true=y_true, y_pred=y_pred, group=group)
        utility = compute_utility(y_true=y_true, y_pred=y_pred, utility_weights=utility_weights)

        if np.isnan(fair["tpr_gap"]):
            abs_tpr_gap = float("inf")
        else:
            abs_tpr_gap = float(abs(fair["tpr_gap"]))

        if np.isnan(fair["spd"]):
            abs_spd = float("inf")
        else:
            abs_spd = float(abs(fair["spd"]))

        fairness_penalty = abs_tpr_gap + abs_spd

        frontier_rows.append(
            {
                "threshold": float(threshold),
                "utility": float(utility),
                "abs_tpr_gap": abs_tpr_gap,
                "abs_spd": abs_spd,
                "fairness_penalty": fairness_penalty,
                "spd": float(fair["spd"]),
                "tpr_gap": float(fair["tpr_gap"]),
            }
        )

    frontier_df = pd.DataFrame(frontier_rows)

    feasible = frontier_df[frontier_df["abs_tpr_gap"] <= fairness_epsilon].copy()
    if spd_epsilon is not None:
        feasible = feasible[feasible["abs_spd"] <= spd_epsilon].copy()

    if not feasible.empty:
        selected = feasible.sort_values(
            by=["utility", "fairness_penalty", "threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        fallback = False
    else:
        selected = frontier_df.sort_values(
            by=["fairness_penalty", "utility", "threshold"],
            ascending=[True, False, True],
        ).iloc[0]
        fallback = True

    selected_threshold = float(selected["threshold"])
    frontier_df["selected"] = np.isclose(frontier_df["threshold"], selected_threshold)

    return selected_threshold, frontier_df, fallback


def apply_groupwise_thresholds(
    probs: np.ndarray,
    group: np.ndarray,
    threshold_group0: float,
    threshold_group1: float,
) -> np.ndarray:
    pred = np.zeros_like(probs, dtype=int)
    mask_0 = group == 0
    mask_1 = ~mask_0

    pred[mask_0] = (probs[mask_0] >= threshold_group0).astype(int)
    pred[mask_1] = (probs[mask_1] >= threshold_group1).astype(int)
    return pred


def choose_groupwise_fair_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    group: np.ndarray,
    utility_weights: Dict[str, float],
    fairness_epsilon: float,
    spd_epsilon: float,
    threshold_grid: np.ndarray,
) -> Tuple[float, float, pd.DataFrame, bool]:
    rows = []

    for threshold_group0 in threshold_grid:
        for threshold_group1 in threshold_grid:
            y_pred = apply_groupwise_thresholds(
                probs=probs,
                group=group,
                threshold_group0=float(threshold_group0),
                threshold_group1=float(threshold_group1),
            )

            fair = fairness_metrics(y_true=y_true, y_pred=y_pred, group=group)
            utility = compute_utility(y_true=y_true, y_pred=y_pred, utility_weights=utility_weights)

            if np.isnan(fair["tpr_gap"]):
                abs_tpr_gap = float("inf")
            else:
                abs_tpr_gap = float(abs(fair["tpr_gap"]))

            if np.isnan(fair["spd"]):
                abs_spd = float("inf")
            else:
                abs_spd = float(abs(fair["spd"]))

            rows.append(
                {
                    "threshold_group0": float(threshold_group0),
                    "threshold_group1": float(threshold_group1),
                    "utility": float(utility),
                    "abs_tpr_gap": abs_tpr_gap,
                    "abs_spd": abs_spd,
                    "fairness_penalty": abs_tpr_gap + abs_spd,
                    "spd": float(fair["spd"]),
                    "tpr_gap": float(fair["tpr_gap"]),
                }
            )

    pair_df = pd.DataFrame(rows)

    feasible = pair_df[
        (pair_df["abs_tpr_gap"] <= fairness_epsilon) &
        (pair_df["abs_spd"] <= spd_epsilon)
    ].copy()

    if not feasible.empty:
        selected = feasible.sort_values(
            by=["utility", "fairness_penalty", "threshold_group0", "threshold_group1"],
            ascending=[False, True, True, True],
        ).iloc[0]
        fallback = False
    else:
        selected = pair_df.sort_values(
            by=["fairness_penalty", "utility", "threshold_group0", "threshold_group1"],
            ascending=[True, False, True, True],
        ).iloc[0]
        fallback = True

    t0 = float(selected["threshold_group0"])
    t1 = float(selected["threshold_group1"])
    pair_df["selected"] = np.isclose(pair_df["threshold_group0"], t0) & np.isclose(pair_df["threshold_group1"], t1)

    return t0, t1, pair_df, fallback


def apply_human_oversight(
    probs: np.ndarray,
    model_pred: np.ndarray,
    oversight_signal: np.ndarray,
    oversight_band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = oversight_band
    escalated_mask = (probs >= lower) & (probs <= upper)

    reviewer_pred = (oversight_signal >= 0.5).astype(int)
    final_pred = model_pred.copy()
    final_pred[escalated_mask] = reviewer_pred[escalated_mask]

    return final_pred, escalated_mask, reviewer_pred
