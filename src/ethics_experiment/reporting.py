from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from ethics_experiment.data import schema_rows


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    if clean.size == 1:
        value = float(clean[0])
        return value, value, value

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(clean, size=clean.size, replace=True)
        boot[i] = float(np.mean(sample))

    mean = float(np.mean(clean))
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return mean, low, high


def _summary_with_ci(results_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []

    grouped = results_df.groupby(["domain", "variant"], as_index=False)
    for (domain, variant), group in grouped:
        for metric in metrics:
            mean, low, high = _bootstrap_ci(group[metric].to_numpy(dtype=float))
            rows.append(
                {
                    "domain": domain,
                    "variant": variant,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    return pd.DataFrame(rows)


def _write_schema_table(tables_dir: Path) -> None:
    schema_df = pd.DataFrame(schema_rows())
    schema_df.to_csv(tables_dir / "dataset_schema.csv", index=False)


def _write_descriptive_tables(descriptives_df: pd.DataFrame, tables_dir: Path) -> None:
    descriptives_df.to_csv(tables_dir / "group_descriptives_by_seed.csv", index=False)

    agg = (
        descriptives_df
        .groupby(["domain", "group_id"], as_index=False)
        .agg(
            n_mean=("n", "mean"),
            y_true_rate_mean=("y_true_rate", "mean"),
            y_hist_rate_mean=("y_hist_rate", "mean"),
            proxy_mean=("proxy_mean", "mean"),
        )
    )
    agg.to_csv(tables_dir / "group_descriptives_summary.csv", index=False)


def _write_metric_tables(results_df: pd.DataFrame, tables_dir: Path) -> None:
    results_df.to_csv(tables_dir / "results_by_seed.csv", index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision",
        "recall",
        "utility",
        "approval_rate_g0",
        "approval_rate_g1",
        "spd",
        "disparate_impact",
        "tpr_gap",
        "fpr_gap",
        "fnr_gap",
        "aod",
        "harm_rate",
        "escalation_rate",
        "override_rate",
        "corrected_error_rate",
        "oversight_gain",
    ]

    summary_df = _summary_with_ci(results_df, metric_cols)
    summary_df.to_csv(tables_dir / "metrics_summary_with_ci.csv", index=False)

    key_metrics = ["utility", "accuracy", "roc_auc", "spd", "tpr_gap", "harm_rate", "escalation_rate", "oversight_gain"]
    ablation = (
        results_df
        .groupby(["domain", "variant"], as_index=False)[key_metrics]
        .mean(numeric_only=True)
    )
    ablation.to_csv(tables_dir / "ablation_table.csv", index=False)


def _write_frontier_table(frontier_df: pd.DataFrame, tables_dir: Path) -> None:
    frontier_df.to_csv(tables_dir / "fairness_utility_frontier.csv", index=False)


def write_all_tables(
    results_df: pd.DataFrame,
    descriptives_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    config: Dict,
    tables_dir: Path,
) -> None:
    del config
    _write_schema_table(tables_dir=tables_dir)
    _write_descriptive_tables(descriptives_df=descriptives_df, tables_dir=tables_dir)
    _write_metric_tables(results_df=results_df, tables_dir=tables_dir)
    _write_frontier_table(frontier_df=frontier_df, tables_dir=tables_dir)


def _plot_group_decision_rates(results_df: pd.DataFrame, plots_dir: Path) -> None:
    variants = ["baseline", "ethical_with_oversight"]
    domains = sorted(results_df["domain"].unique())
    domain_summaries: Dict[str, pd.DataFrame] = {}

    for domain in domains:
        domain_summaries[domain] = (
            results_df[(results_df["domain"] == domain) & (results_df["variant"].isin(variants))]
            .groupby("variant", as_index=False)[["approval_rate_g0", "approval_rate_g1"]]
            .mean(numeric_only=True)
            .set_index("variant")
        )

    fig, axes = plt.subplots(1, len(domains), figsize=(16, 4.5), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        subset = domain_summaries[domain]

        x = np.arange(2)
        width = 0.36

        baseline_rates = [subset.loc["baseline", "approval_rate_g0"], subset.loc["baseline", "approval_rate_g1"]]
        ethical_rates = [subset.loc["ethical_with_oversight", "approval_rate_g0"], subset.loc["ethical_with_oversight", "approval_rate_g1"]]

        ax.bar(x - width / 2, baseline_rates, width=width, label="baseline", alpha=0.85)
        ax.bar(x + width / 2, ethical_rates, width=width, label="ethical+oversight", alpha=0.85)

        ax.set_title(domain.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(["Group 0", "Group 1"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Positive decision rate")

        if idx == 0:
            ax.legend()

    fig.suptitle("Group-wise Decision Rates: Baseline vs Ethical Agent", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "group_decision_rates.png", dpi=180)
    plt.close(fig)

    for domain in domains:
        subset = domain_summaries[domain]
        x = np.arange(2)
        width = 0.36

        baseline_rates = [subset.loc["baseline", "approval_rate_g0"], subset.loc["baseline", "approval_rate_g1"]]
        ethical_rates = [subset.loc["ethical_with_oversight", "approval_rate_g0"], subset.loc["ethical_with_oversight", "approval_rate_g1"]]

        single_fig, single_ax = plt.subplots(figsize=(6, 4.5))
        single_ax.bar(x - width / 2, baseline_rates, width=width, label="baseline", alpha=0.85)
        single_ax.bar(x + width / 2, ethical_rates, width=width, label="ethical+oversight", alpha=0.85)

        single_ax.set_title(domain.replace("_", " ").title())
        single_ax.set_xticks(x)
        single_ax.set_xticklabels(["Group 0", "Group 1"])
        single_ax.set_ylim(0.0, 1.0)
        single_ax.set_ylabel("Positive decision rate")
        single_ax.legend()

        single_fig.tight_layout()
        single_fig.savefig(plots_dir / f"group_decision_rates_{domain}.png", dpi=180)
        plt.close(single_fig)


def _plot_fairness_utility_frontier(frontier_df: pd.DataFrame, plots_dir: Path) -> None:
    agg = (
        frontier_df
        .groupby(["domain", "threshold"], as_index=False)
        .agg(
            utility=("utility", "mean"),
            abs_tpr_gap=("abs_tpr_gap", "mean"),
        )
    )

    domains = sorted(agg["domain"].unique())
    fig, axes = plt.subplots(1, len(domains), figsize=(16, 4.5), sharey=False)
    if len(domains) == 1:
        axes = [axes]

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        sub = agg[agg["domain"] == domain].sort_values("threshold")

        ax.scatter(sub["abs_tpr_gap"], sub["utility"], c=sub["threshold"], cmap="viridis", s=35)
        ax.axvline(0.05, color="red", linestyle="--", linewidth=1)
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("Absolute TPR gap")
        ax.set_ylabel("Utility")

    fig.suptitle("Fairness-Utility Frontier (Threshold Sweep)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "fairness_utility_frontier.png", dpi=180)
    plt.close(fig)

    for domain in domains:
        sub = agg[agg["domain"] == domain].sort_values("threshold")

        single_fig, single_ax = plt.subplots(figsize=(6, 4.5))
        scatter = single_ax.scatter(sub["abs_tpr_gap"], sub["utility"], c=sub["threshold"], cmap="viridis", s=35)
        single_ax.axvline(0.05, color="red", linestyle="--", linewidth=1)
        single_ax.set_title(domain.replace("_", " ").title())
        single_ax.set_xlabel("Absolute TPR gap")
        single_ax.set_ylabel("Utility")
        single_fig.colorbar(scatter, ax=single_ax, label="Threshold")

        single_fig.tight_layout()
        single_fig.savefig(plots_dir / f"fairness_utility_frontier_{domain}.png", dpi=180)
        plt.close(single_fig)


def _plot_roc_by_group(plot_payload: Dict[str, Dict[str, object]], plots_dir: Path) -> None:
    domains = sorted(plot_payload.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(17, 5), sharex=True, sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        payload = plot_payload[domain]

        y_true = np.asarray(payload["y_true"])
        group = np.asarray(payload["group"])
        baseline_probs = np.asarray(payload["baseline_probs"])
        ethical_probs = np.asarray(payload["ethical_probs"])

        for variant_name, probs, style in [
            ("baseline", baseline_probs, "-"),
            ("ethical", ethical_probs, "--"),
        ]:
            for g in [0, 1]:
                mask = group == g
                if np.unique(y_true[mask]).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true[mask], probs[mask])
                ax.plot(fpr, tpr, style, linewidth=1.7, label=f"{variant_name} g{g}")

        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("ROC Curves by Group and Agent", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_by_group.png", dpi=180)
    plt.close(fig)


def _plot_calibration_by_group(plot_payload: Dict[str, Dict[str, object]], plots_dir: Path) -> None:
    domains = sorted(plot_payload.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(17, 5), sharex=True, sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        payload = plot_payload[domain]

        y_true = np.asarray(payload["y_true"])
        group = np.asarray(payload["group"])
        baseline_probs = np.asarray(payload["baseline_probs"])
        ethical_probs = np.asarray(payload["ethical_probs"])

        for variant_name, probs, style in [
            ("baseline", baseline_probs, "-"),
            ("ethical", ethical_probs, "--"),
        ]:
            for g in [0, 1]:
                mask = group == g
                if np.unique(y_true[mask]).size < 2:
                    continue
                frac_pos, mean_pred = calibration_curve(y_true[mask], probs[mask], n_bins=10, strategy="uniform")
                ax.plot(mean_pred, frac_pos, style, linewidth=1.6, label=f"{variant_name} g{g}")

        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")

        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Calibration Curves by Group and Agent", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "calibration_by_group.png", dpi=180)
    plt.close(fig)


def _plot_oversight_flow(plot_payload: Dict[str, Dict[str, object]], plots_dir: Path) -> None:
    domains = sorted(plot_payload.keys())

    labels = [d.replace("_", " ").title() for d in domains]
    auto_pos = np.array([plot_payload[d]["oversight_flow"]["auto_positive"] for d in domains], dtype=float)
    auto_neg = np.array([plot_payload[d]["oversight_flow"]["auto_negative"] for d in domains], dtype=float)
    escalated = np.array([plot_payload[d]["oversight_flow"]["escalated"] for d in domains], dtype=float)
    overridden = np.array([plot_payload[d]["oversight_flow"]["overridden"] for d in domains], dtype=float)

    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(9.5, 5))

    ax.bar(x, auto_pos, label="Auto positive", alpha=0.9)
    ax.bar(x, auto_neg, bottom=auto_pos, label="Auto negative", alpha=0.9)
    ax.bar(x, escalated, bottom=auto_pos + auto_neg, label="Escalated", alpha=0.9)

    for i, val in enumerate(overridden):
        ax.text(i, auto_pos[i] + auto_neg[i] + escalated[i] + 10, f"overrides={int(val)}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of test cases")
    ax.set_title("Human Oversight Flow (First Seed Snapshot)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "oversight_flow.png", dpi=180)
    plt.close(fig)


def _plot_cross_domain_meta(results_df: pd.DataFrame, plots_dir: Path) -> None:
    mean_df = results_df.groupby(["domain", "variant"], as_index=False).mean(numeric_only=True)

    rows = []
    for domain in sorted(mean_df["domain"].unique()):
        baseline = mean_df[(mean_df["domain"] == domain) & (mean_df["variant"] == "baseline")].iloc[0]
        ethical = mean_df[(mean_df["domain"] == domain) & (mean_df["variant"] == "ethical_with_oversight")].iloc[0]

        baseline_gap = abs(float(baseline["tpr_gap"]))
        ethical_gap = abs(float(ethical["tpr_gap"]))
        if baseline_gap > 0:
            nfi = (baseline_gap - ethical_gap) / baseline_gap
        else:
            nfi = float("nan")

        baseline_utility = float(baseline["utility"])
        ethical_utility = float(ethical["utility"])
        utility_delta = ethical_utility - baseline_utility
        utility_change_relative = utility_delta / max(abs(baseline_utility), 1e-9)

        rows.append(
            {
                "domain": domain,
                "nfi": nfi,
                "utility_change_relative": utility_change_relative,
            }
        )

    meta_df = pd.DataFrame(rows)

    x = np.arange(meta_df.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - width / 2, meta_df["nfi"], width=width, label="Normalized Fairness Improvement")
    ax.bar(
        x + width / 2,
        meta_df["utility_change_relative"],
        width=width,
        label="Utility change / |baseline|",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in meta_df["domain"]])
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Domain Fairness Improvement and Relative Utility Change")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "cross_domain_meta.png", dpi=180)
    plt.close(fig)


def generate_all_plots(
    results_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    plot_payload: Dict[str, Dict[str, object]],
    plots_dir: Path,
) -> None:
    _plot_group_decision_rates(results_df=results_df, plots_dir=plots_dir)
    _plot_fairness_utility_frontier(frontier_df=frontier_df, plots_dir=plots_dir)
    _plot_roc_by_group(plot_payload=plot_payload, plots_dir=plots_dir)
    _plot_calibration_by_group(plot_payload=plot_payload, plots_dir=plots_dir)
    _plot_oversight_flow(plot_payload=plot_payload, plots_dir=plots_dir)
    _plot_cross_domain_meta(results_df=results_df, plots_dir=plots_dir)


def _domain_result_snapshot(results_df: pd.DataFrame, domain: str) -> Dict[str, float]:
    means = (
        results_df[results_df["domain"] == domain]
        .groupby("variant", as_index=False)
        .mean(numeric_only=True)
        .set_index("variant")
    )

    baseline = means.loc["baseline"]
    ethical = means.loc["ethical_with_oversight"]

    baseline_gap = abs(float(baseline["tpr_gap"]))
    ethical_gap = abs(float(ethical["tpr_gap"]))
    fairness_improvement = (baseline_gap - ethical_gap) / baseline_gap if baseline_gap > 0 else float("nan")

    baseline_utility = float(baseline["utility"])
    ethical_utility = float(ethical["utility"])
    utility_delta = ethical_utility - baseline_utility
    utility_change_relative = utility_delta / max(abs(baseline_utility), 1e-9)

    return {
        "baseline_spd": float(baseline["spd"]),
        "ethical_spd": float(ethical["spd"]),
        "baseline_tpr_gap": float(baseline["tpr_gap"]),
        "ethical_tpr_gap": float(ethical["tpr_gap"]),
        "baseline_utility": baseline_utility,
        "ethical_utility": ethical_utility,
        "utility_delta": utility_delta,
        "utility_change_relative": utility_change_relative,
        "fairness_improvement": fairness_improvement,
        "escalation_rate": float(ethical["escalation_rate"]),
    }
