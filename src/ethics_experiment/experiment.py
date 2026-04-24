from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ethics_experiment.agents import (
    apply_groupwise_thresholds,
    apply_human_oversight,
    choose_fair_threshold,
    choose_groupwise_fair_thresholds,
    predict_probabilities,
    train_logistic_model,
)
from ethics_experiment.data import (
    generate_domain_dataset,
    get_feature_columns,
    split_domain_dataset,
)
from ethics_experiment.metrics import aggregate_core_metrics, oversight_metrics


VARIANTS = [
    "baseline",
    "no_sensitive_proxy",
    "label_correction_only",
    "ethical_no_oversight",
    "ethical_with_oversight",
]


def _domain_descriptives(domain: str, seed: int, df: pd.DataFrame) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for group_value in [0, 1]:
        subset = df[df["group_id"] == group_value]
        rows.append(
            {
                "domain": domain,
                "seed": seed,
                "group_id": group_value,
                "n": int(subset.shape[0]),
                "y_true_rate": float(subset["y_true"].mean()),
                "y_hist_rate": float(subset["y_hist"].mean()),
                "proxy_mean": float(subset["proxy_feature"].mean()),
            }
        )
    return rows


def _evaluate_variant(
    domain: str,
    seed: int,
    variant: str,
    threshold: float,
    threshold_group0: float,
    threshold_group1: float,
    groupwise_thresholding_used: bool,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    group: np.ndarray,
    utility_weights: Dict[str, float],
    fairness_threshold_fallback: bool | None = None,
    oversight_info: Dict[str, float] | None = None,
) -> Dict[str, float | int | str | bool]:
    metrics = aggregate_core_metrics(
        domain=domain,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        group=group,
        utility_weights=utility_weights,
    )

    row: Dict[str, float | int | str | bool] = {
        "domain": domain,
        "seed": seed,
        "variant": variant,
        "threshold": float(threshold),
        "threshold_group0": float(threshold_group0),
        "threshold_group1": float(threshold_group1),
        "groupwise_thresholding_used": bool(groupwise_thresholding_used),
        "fairness_threshold_fallback": fairness_threshold_fallback if fairness_threshold_fallback is not None else False,
        "escalation_rate": float("nan"),
        "override_rate": float("nan"),
        "corrected_error_rate": float("nan"),
        "oversight_gain": float("nan"),
        "residual_tpr_gap": float("nan"),
        "residual_spd": float("nan"),
    }
    row.update(metrics)

    if oversight_info is not None:
        row.update(oversight_info)

    return row


def _run_single_domain_seed(
    domain: str,
    seed: int,
    config: Dict,
    first_seed: int,
) -> Dict[str, object]:
    domain_cfg = config["domains"][domain]

    df = generate_domain_dataset(
        domain=domain,
        n_samples=int(config["n_samples_per_domain"]),
        seed=seed,
        bias_group_penalty=float(domain_cfg["bias_group_penalty"]),
        bias_proxy_penalty=float(domain_cfg["bias_proxy_penalty"]),
        historical_label_noise=float(domain_cfg.get("historical_label_noise", 0.08)),
    )

    split = split_domain_dataset(df=df, seed=seed, split_config=config["split"])
    train_df = split["train"]
    val_df = split["val"]
    test_df = split["test"]

    threshold_grid = np.linspace(
        float(config["threshold_grid"][0]),
        float(config["threshold_grid"][1]),
        int(config["threshold_grid"][2]),
    )

    default_threshold = float(config["default_threshold"])
    fairness_epsilon = float(config["fairness_epsilon"])
    spd_epsilon = float(config["spd_epsilon"]) if "spd_epsilon" in config else None
    oversight_band = (float(config["oversight_band"][0]), float(config["oversight_band"][1]))
    utility_weights = domain_cfg["utility_weights"]

    with_sensitive_proxy = get_feature_columns(domain=domain, include_sensitive_proxy=True)
    without_sensitive_proxy = get_feature_columns(domain=domain, include_sensitive_proxy=False)

    y_true_test = test_df["y_true"].to_numpy()
    group_test = test_df["group_id"].to_numpy()

    rows: List[Dict[str, object]] = []
    frontier_rows: List[Dict[str, object]] = []

    plot_payload: Dict[str, object] = {}

    model_baseline = train_logistic_model(train_df, with_sensitive_proxy, "y_hist", seed)
    probs_baseline = predict_probabilities(model_baseline, test_df, with_sensitive_proxy)
    pred_baseline = (probs_baseline >= default_threshold).astype(int)
    rows.append(
        _evaluate_variant(
            domain=domain,
            seed=seed,
            variant="baseline",
            threshold=default_threshold,
            threshold_group0=default_threshold,
            threshold_group1=default_threshold,
            groupwise_thresholding_used=False,
            y_true=y_true_test,
            y_pred=pred_baseline,
            probs=probs_baseline,
            group=group_test,
            utility_weights=utility_weights,
        )
    )

    model_no_sensitive = train_logistic_model(train_df, without_sensitive_proxy, "y_hist", seed)
    probs_no_sensitive = predict_probabilities(model_no_sensitive, test_df, without_sensitive_proxy)
    pred_no_sensitive = (probs_no_sensitive >= default_threshold).astype(int)
    rows.append(
        _evaluate_variant(
            domain=domain,
            seed=seed,
            variant="no_sensitive_proxy",
            threshold=default_threshold,
            threshold_group0=default_threshold,
            threshold_group1=default_threshold,
            groupwise_thresholding_used=False,
            y_true=y_true_test,
            y_pred=pred_no_sensitive,
            probs=probs_no_sensitive,
            group=group_test,
            utility_weights=utility_weights,
        )
    )

    model_label_correct = train_logistic_model(train_df, with_sensitive_proxy, "y_true", seed)
    probs_label_correct = predict_probabilities(model_label_correct, test_df, with_sensitive_proxy)
    pred_label_correct = (probs_label_correct >= default_threshold).astype(int)
    rows.append(
        _evaluate_variant(
            domain=domain,
            seed=seed,
            variant="label_correction_only",
            threshold=default_threshold,
            threshold_group0=default_threshold,
            threshold_group1=default_threshold,
            groupwise_thresholding_used=False,
            y_true=y_true_test,
            y_pred=pred_label_correct,
            probs=probs_label_correct,
            group=group_test,
            utility_weights=utility_weights,
        )
    )

    model_ethical = train_logistic_model(train_df, without_sensitive_proxy, "y_true", seed)
    probs_val_ethical = predict_probabilities(model_ethical, val_df, without_sensitive_proxy)
    y_true_val = val_df["y_true"].to_numpy()
    group_val = val_df["group_id"].to_numpy()

    selected_threshold, frontier_df, threshold_fallback = choose_fair_threshold(
        y_true=y_true_val,
        probs=probs_val_ethical,
        group=group_val,
        utility_weights=utility_weights,
        fairness_epsilon=fairness_epsilon,
        threshold_grid=threshold_grid,
        spd_epsilon=spd_epsilon,
    )

    frontier_df["domain"] = domain
    frontier_df["seed"] = seed
    frontier_rows.extend(frontier_df.to_dict(orient="records"))

    probs_ethical = predict_probabilities(model_ethical, test_df, without_sensitive_proxy)

    threshold_group0 = selected_threshold
    threshold_group1 = selected_threshold
    groupwise_thresholding_used = False

    if threshold_fallback and spd_epsilon is not None:
        pair_t0, pair_t1, _, pair_fallback = choose_groupwise_fair_thresholds(
            y_true=y_true_val,
            probs=probs_val_ethical,
            group=group_val,
            utility_weights=utility_weights,
            fairness_epsilon=fairness_epsilon,
            spd_epsilon=spd_epsilon,
            threshold_grid=threshold_grid,
        )
        threshold_group0 = pair_t0
        threshold_group1 = pair_t1
        groupwise_thresholding_used = True
        threshold_fallback = pair_fallback

    pred_ethical = apply_groupwise_thresholds(
        probs=probs_ethical,
        group=group_test,
        threshold_group0=threshold_group0,
        threshold_group1=threshold_group1,
    )

    rows.append(
        _evaluate_variant(
            domain=domain,
            seed=seed,
            variant="ethical_no_oversight",
            threshold=selected_threshold,
            threshold_group0=threshold_group0,
            threshold_group1=threshold_group1,
            groupwise_thresholding_used=groupwise_thresholding_used,
            y_true=y_true_test,
            y_pred=pred_ethical,
            probs=probs_ethical,
            group=group_test,
            utility_weights=utility_weights,
            fairness_threshold_fallback=threshold_fallback,
        )
    )

    final_pred, escalated_mask, _ = apply_human_oversight(
        probs=probs_ethical,
        model_pred=pred_ethical,
        oversight_signal=test_df["oversight_signal"].to_numpy(),
        oversight_band=oversight_band,
    )

    oversight_info = oversight_metrics(
        y_true=y_true_test,
        model_pred=pred_ethical,
        final_pred=final_pred,
        escalated_mask=escalated_mask,
        group=group_test,
        utility_weights=utility_weights,
    )

    rows.append(
        _evaluate_variant(
            domain=domain,
            seed=seed,
            variant="ethical_with_oversight",
            threshold=selected_threshold,
            threshold_group0=threshold_group0,
            threshold_group1=threshold_group1,
            groupwise_thresholding_used=groupwise_thresholding_used,
            y_true=y_true_test,
            y_pred=final_pred,
            probs=probs_ethical,
            group=group_test,
            utility_weights=utility_weights,
            fairness_threshold_fallback=threshold_fallback,
            oversight_info=oversight_info,
        )
    )

    if seed == first_seed:
        plot_payload = {
            "y_true": y_true_test,
            "group": group_test,
            "baseline_probs": probs_baseline,
            "ethical_probs": probs_ethical,
            "oversight_flow": {
                "auto_positive": int(np.sum((~escalated_mask) & (pred_ethical == 1))),
                "auto_negative": int(np.sum((~escalated_mask) & (pred_ethical == 0))),
                "escalated": int(np.sum(escalated_mask)),
                "overridden": int(np.sum((pred_ethical != final_pred) & escalated_mask)),
            },
        }

    return {
        "rows": rows,
        "descriptives": _domain_descriptives(domain=domain, seed=seed, df=df),
        "frontier_rows": frontier_rows,
        "plot_payload": plot_payload,
    }


def run_full_experiment(config: Dict) -> Dict[str, object]:
    random_seeds = [int(x) for x in config["random_seeds"]]
    domains = list(config["domains"].keys())

    all_rows: List[Dict[str, object]] = []
    all_descriptives: List[Dict[str, object]] = []
    all_frontier_rows: List[Dict[str, object]] = []
    plot_payload: Dict[str, Dict[str, object]] = {}

    first_seed = random_seeds[0]

    for seed in random_seeds:
        for domain in domains:
            result = _run_single_domain_seed(
                domain=domain,
                seed=seed,
                config=config,
                first_seed=first_seed,
            )
            all_rows.extend(result["rows"])
            all_descriptives.extend(result["descriptives"])
            all_frontier_rows.extend(result["frontier_rows"])
            if result["plot_payload"]:
                plot_payload[domain] = result["plot_payload"]

    results_df = pd.DataFrame(all_rows)
    descriptives_df = pd.DataFrame(all_descriptives)
    frontier_df = pd.DataFrame(all_frontier_rows)

    return {
        "results_df": results_df,
        "descriptives_df": descriptives_df,
        "frontier_df": frontier_df,
        "plot_payload": plot_payload,
    }
