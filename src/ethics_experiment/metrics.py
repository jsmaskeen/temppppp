from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if np.unique(y_true).size < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))

    return metrics


def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    mask_0 = group == 0
    mask_1 = group == 1

    approval_0 = float(np.mean(y_pred[mask_0])) if np.any(mask_0) else float("nan")
    approval_1 = float(np.mean(y_pred[mask_1])) if np.any(mask_1) else float("nan")

    tp_0 = np.sum((y_true == 1) & (y_pred == 1) & mask_0)
    fn_0 = np.sum((y_true == 1) & (y_pred == 0) & mask_0)
    fp_0 = np.sum((y_true == 0) & (y_pred == 1) & mask_0)
    tn_0 = np.sum((y_true == 0) & (y_pred == 0) & mask_0)

    tp_1 = np.sum((y_true == 1) & (y_pred == 1) & mask_1)
    fn_1 = np.sum((y_true == 1) & (y_pred == 0) & mask_1)
    fp_1 = np.sum((y_true == 0) & (y_pred == 1) & mask_1)
    tn_1 = np.sum((y_true == 0) & (y_pred == 0) & mask_1)

    tpr_0 = _safe_div(tp_0, tp_0 + fn_0)
    tpr_1 = _safe_div(tp_1, tp_1 + fn_1)
    fpr_0 = _safe_div(fp_0, fp_0 + tn_0)
    fpr_1 = _safe_div(fp_1, fp_1 + tn_1)
    fnr_0 = _safe_div(fn_0, tp_0 + fn_0)
    fnr_1 = _safe_div(fn_1, tp_1 + fn_1)

    spd = approval_1 - approval_0 if not np.isnan(approval_0) and not np.isnan(approval_1) else float("nan")
    di = _safe_div(approval_1, approval_0) if not np.isnan(approval_0) and not np.isnan(approval_1) else float("nan")

    tpr_gap = tpr_1 - tpr_0 if not np.isnan(tpr_0) and not np.isnan(tpr_1) else float("nan")
    fpr_gap = fpr_1 - fpr_0 if not np.isnan(fpr_0) and not np.isnan(fpr_1) else float("nan")
    fnr_gap = fnr_1 - fnr_0 if not np.isnan(fnr_0) and not np.isnan(fnr_1) else float("nan")

    if np.isnan(tpr_gap) or np.isnan(fpr_gap):
        aod = float("nan")
    else:
        aod = 0.5 * (tpr_gap + fpr_gap)

    return {
        "approval_rate_g0": approval_0,
        "approval_rate_g1": approval_1,
        "spd": spd,
        "disparate_impact": di,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "aod": aod,
    }


def compute_utility(y_true: np.ndarray, y_pred: np.ndarray, utility_weights: Dict[str, float]) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    raw = utility_weights["tp"] * tp + utility_weights["fp"] * fp + utility_weights["fn"] * fn
    return float(raw / y_true.size)


def domain_harm_metrics(domain: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | str]:
    positives = y_pred == 1

    if domain == "loan":
        if np.any(positives):
            harm_rate = float(np.mean(y_true[positives] == 0))
        else:
            harm_rate = float("nan")
        return {"harm_metric": "default_rate_among_approved", "harm_rate": harm_rate}

    if domain == "hiring":
        if np.any(positives):
            harm_rate = float(np.mean(y_true[positives] == 0))
        else:
            harm_rate = float("nan")
        return {"harm_metric": "bad_hire_rate_among_selected", "harm_rate": harm_rate}

    if domain == "medical_triage":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        harm_rate = _safe_div(fn, tp + fn)
        return {"harm_metric": "missed_urgent_rate", "harm_rate": harm_rate}

    raise ValueError(f"Unsupported domain: {domain}")


def oversight_metrics(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    final_pred: np.ndarray,
    escalated_mask: np.ndarray,
    group: np.ndarray,
    utility_weights: Dict[str, float],
) -> Dict[str, float]:
    escalation_rate = float(np.mean(escalated_mask))

    if np.any(escalated_mask):
        override_rate = float(np.mean((model_pred[escalated_mask] != final_pred[escalated_mask]).astype(float)))

        escalated_errors = (model_pred != y_true) & escalated_mask
        corrected_errors = escalated_errors & (final_pred == y_true)
        corrected_error_rate = _safe_div(np.sum(corrected_errors), np.sum(escalated_errors))
    else:
        override_rate = 0.0
        corrected_error_rate = float("nan")

    utility_before = compute_utility(y_true, model_pred, utility_weights)
    utility_after = compute_utility(y_true, final_pred, utility_weights)

    residual_fair = fairness_metrics(y_true=y_true, y_pred=final_pred, group=group)

    return {
        "escalation_rate": escalation_rate,
        "override_rate": override_rate,
        "corrected_error_rate": corrected_error_rate,
        "oversight_gain": utility_after - utility_before,
        "residual_tpr_gap": residual_fair["tpr_gap"],
        "residual_spd": residual_fair["spd"],
    }


def aggregate_core_metrics(
    domain: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    group: np.ndarray,
    utility_weights: Dict[str, float],
) -> Dict[str, float | str]:
    core = classification_metrics(y_true=y_true, y_pred=y_pred, probs=probs)
    fair = fairness_metrics(y_true=y_true, y_pred=y_pred, group=group)
    utility = compute_utility(y_true=y_true, y_pred=y_pred, utility_weights=utility_weights)
    harm = domain_harm_metrics(domain=domain, y_true=y_true, y_pred=y_pred)

    return {
        **core,
        **fair,
        **harm,
        "utility": utility,
    }
