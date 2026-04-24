from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DOMAIN_LEGITIMATE_FEATURES: Dict[str, List[str]] = {
    "loan": [
        "income",
        "debt_to_income",
        "employment_years",
        "prior_defaults",
        "credit_history_years",
        "requested_loan",
    ],
    "hiring": [
        "years_experience",
        "skill_test_score",
        "education_quality",
        "employment_gap_months",
        "portfolio_score",
    ],
    "medical_triage": [
        "severity_score",
        "comorbidity_count",
        "age",
        "oxygen_saturation",
        "heart_rate_risk",
    ],
}

DOMAIN_PROXY_COLUMNS: Dict[str, str] = {
    "loan": "zip_risk_proxy",
    "hiring": "referral_proxy",
    "medical_triage": "arrival_mode_proxy",
}

SENSITIVE_COLUMN = "group_id"
PROXY_COLUMN = "proxy_feature"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _clip(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(x, low, high)


def get_legitimate_features(domain: str) -> List[str]:
    if domain not in DOMAIN_LEGITIMATE_FEATURES:
        raise ValueError(f"Unsupported domain: {domain}")
    return DOMAIN_LEGITIMATE_FEATURES[domain]


def get_feature_columns(domain: str, include_sensitive_proxy: bool) -> List[str]:
    cols = list(get_legitimate_features(domain))
    if include_sensitive_proxy:
        cols.extend([SENSITIVE_COLUMN, PROXY_COLUMN])
    return cols


def split_domain_dataset(
    df: pd.DataFrame,
    seed: int,
    split_config: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    train_frac = split_config["train"]
    val_frac = split_config["val"]

    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_frac,
        random_state=seed,
        stratify=df["y_true"],
    )

    val_ratio_within_temp = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_ratio_within_temp,
        random_state=seed + 1,
        stratify=temp_df["y_true"],
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def _generate_loan_domain(
    n_samples: int,
    rng: np.random.Generator,
    group_id: np.ndarray,
    proxy_feature: np.ndarray,
) -> pd.DataFrame:
    latent_merit = rng.normal(0.0, 1.0, n_samples)

    income = _clip(56000 + 16000 * latent_merit + rng.normal(0, 9000, n_samples), 20000, 150000)
    debt_to_income = _clip(0.43 - 0.1 * latent_merit + 0.08 * group_id + rng.normal(0, 0.07, n_samples), 0.05, 0.9)
    employment_years = _clip(7.5 + 2.0 * latent_merit + rng.normal(0, 3.0, n_samples), 0.0, 35.0)

    default_lambda = _clip(1.1 - 0.45 * latent_merit + 0.4 * proxy_feature, 0.1, 3.5)
    prior_defaults = rng.poisson(default_lambda).astype(float)

    credit_history_years = _clip(9.0 + 2.2 * latent_merit + rng.normal(0, 4.0, n_samples), 0.0, 30.0)
    requested_loan = _clip(12000 + 3500 * (1 - latent_merit) + rng.normal(0, 5000, n_samples), 1000, 50000)

    repay_score = (
        -1.6
        + 0.00003 * income
        - 2.2 * debt_to_income
        - 0.6 * prior_defaults
        + 0.05 * employment_years
        + 0.04 * credit_history_years
        - 0.00002 * requested_loan
        + rng.normal(0, 0.35, n_samples)
    )
    p_true = _clip(sigmoid(repay_score), 0.001, 0.999)
    y_true = rng.binomial(1, p_true).astype(int)

    oversight_signal = _clip(
        sigmoid(
            0.00002 * income
            - 1.6 * debt_to_income
            - 0.45 * prior_defaults
            + 0.03 * credit_history_years
            + rng.normal(0, 0.8, n_samples)
        ),
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "income": income,
            "debt_to_income": debt_to_income,
            "employment_years": employment_years,
            "prior_defaults": prior_defaults,
            "credit_history_years": credit_history_years,
            "requested_loan": requested_loan,
            "zip_risk_proxy": proxy_feature,
            "p_true": p_true,
            "y_true": y_true,
            "oversight_signal": oversight_signal,
        }
    )


def _generate_hiring_domain(
    n_samples: int,
    rng: np.random.Generator,
    group_id: np.ndarray,
    proxy_feature: np.ndarray,
) -> pd.DataFrame:
    latent_merit = rng.normal(0.0, 1.0, n_samples)

    years_experience = _clip(5.5 + 2.3 * latent_merit + rng.normal(0, 2.0, n_samples), 0.0, 25.0)
    skill_test_score = _clip(sigmoid(1.1 * latent_merit + rng.normal(0, 0.8, n_samples)), 0.0, 1.0)
    education_quality = _clip(np.round(3 + 0.9 * latent_merit + rng.normal(0, 0.9, n_samples)), 1, 5)
    employment_gap_months = _clip(8.0 - 2.0 * latent_merit + 4.0 * proxy_feature + rng.normal(0, 4.0, n_samples), 0.0, 48.0)
    portfolio_score = _clip(sigmoid(1.2 * latent_merit + rng.normal(0, 0.8, n_samples)), 0.0, 1.0)

    referral_proxy = _clip(0.7 * proxy_feature + 0.3 * sigmoid(-0.8 * group_id + rng.normal(0, 1.0, n_samples)), 0.0, 1.0)

    performance_score = (
        -0.8
        + 0.09 * years_experience
        + 1.7 * skill_test_score
        + 0.25 * education_quality
        - 0.05 * employment_gap_months
        + 1.1 * portfolio_score
        + rng.normal(0, 0.35, n_samples)
    )
    p_true = _clip(sigmoid(performance_score), 0.001, 0.999)
    y_true = rng.binomial(1, p_true).astype(int)

    oversight_signal = _clip(
        sigmoid(
            -0.7
            + 1.2 * skill_test_score
            + 0.9 * portfolio_score
            + 0.03 * years_experience
            - 0.03 * employment_gap_months
            + rng.normal(0, 0.65, n_samples)
        ),
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "years_experience": years_experience,
            "skill_test_score": skill_test_score,
            "education_quality": education_quality,
            "employment_gap_months": employment_gap_months,
            "portfolio_score": portfolio_score,
            "referral_proxy": referral_proxy,
            "p_true": p_true,
            "y_true": y_true,
            "oversight_signal": oversight_signal,
        }
    )


def _generate_medical_domain(
    n_samples: int,
    rng: np.random.Generator,
    group_id: np.ndarray,
    proxy_feature: np.ndarray,
) -> pd.DataFrame:
    latent_merit = rng.normal(0.0, 1.0, n_samples)

    severity_score = _clip(sigmoid(1.2 * latent_merit + rng.normal(0, 0.9, n_samples)), 0.0, 1.0)
    comorbidity_lambda = _clip(1.6 - 0.3 * latent_merit + 0.2 * group_id, 0.2, 6.0)
    comorbidity_count = rng.poisson(comorbidity_lambda).astype(float)
    age = _clip(47 + 12 * rng.normal(0, 1, n_samples) + 4 * group_id, 18.0, 90.0)
    oxygen_saturation = _clip(98 - 9 * severity_score - 0.7 * comorbidity_count + rng.normal(0, 2.0, n_samples), 68.0, 100.0)
    heart_rate_risk = _clip(0.2 + 0.55 * severity_score + 0.05 * comorbidity_count + rng.normal(0, 0.12, n_samples), 0.0, 1.0)

    arrival_mode_proxy = _clip(proxy_feature, 0.0, 1.0)

    urgency_score = (
        -2.3
        + 4.0 * severity_score
        + 0.35 * comorbidity_count
        + 0.06 * (90 - oxygen_saturation)
        + 1.6 * heart_rate_risk
        + rng.normal(0, 0.35, n_samples)
    )
    p_true = _clip(sigmoid(urgency_score), 0.001, 0.999)
    y_true = rng.binomial(1, p_true).astype(int)

    oversight_signal = _clip(
        sigmoid(
            -1.6
            + 3.2 * severity_score
            + 0.22 * comorbidity_count
            + 0.05 * (92 - oxygen_saturation)
            + rng.normal(0, 0.7, n_samples)
        ),
        0.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "severity_score": severity_score,
            "comorbidity_count": comorbidity_count,
            "age": age,
            "oxygen_saturation": oxygen_saturation,
            "heart_rate_risk": heart_rate_risk,
            "arrival_mode_proxy": arrival_mode_proxy,
            "p_true": p_true,
            "y_true": y_true,
            "oversight_signal": oversight_signal,
        }
    )


def generate_domain_dataset(
    domain: str,
    n_samples: int,
    seed: int,
    bias_group_penalty: float,
    bias_proxy_penalty: float,
    historical_label_noise: float = 0.08,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group_id = rng.binomial(1, 0.5, n_samples).astype(int)
    proxy_feature = _clip(sigmoid(1.25 * group_id + rng.normal(0, 1.0, n_samples)), 0.0, 1.0)

    if domain == "loan":
        df = _generate_loan_domain(n_samples=n_samples, rng=rng, group_id=group_id, proxy_feature=proxy_feature)
    elif domain == "hiring":
        df = _generate_hiring_domain(n_samples=n_samples, rng=rng, group_id=group_id, proxy_feature=proxy_feature)
    elif domain == "medical_triage":
        df = _generate_medical_domain(n_samples=n_samples, rng=rng, group_id=group_id, proxy_feature=proxy_feature)
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    p_true = df["p_true"].to_numpy()
    hist_score = (
        p_true
        - bias_group_penalty * group_id
        - bias_proxy_penalty * proxy_feature
        + rng.normal(0, historical_label_noise, n_samples)
    )
    y_hist = (hist_score >= 0.5).astype(int)

    df.insert(0, "applicant_id", np.arange(1, n_samples + 1, dtype=int))
    df.insert(1, "domain", domain)
    df[SENSITIVE_COLUMN] = group_id
    df[PROXY_COLUMN] = proxy_feature
    df["y_hist"] = y_hist

    return df


def schema_rows() -> List[dict]:
    rows: List[dict] = []

    common = [
        ("applicant_id", "int", "Unique row id"),
        ("domain", "str", "Experiment domain"),
        ("group_id", "binary", "Protected group attribute for fairness auditing"),
        ("proxy_feature", "float", "Proxy correlated with group membership"),
        ("oversight_signal", "float", "Human reviewer confidence signal in [0,1]"),
        ("p_true", "float", "Latent true probability of positive outcome"),
        ("y_true", "binary", "True outcome label"),
        ("y_hist", "binary", "Biased historical decision label"),
    ]

    for domain, features in DOMAIN_LEGITIMATE_FEATURES.items():
        proxy_col = DOMAIN_PROXY_COLUMNS[domain]
        for name, dtype, desc in common:
            rows.append({"domain": domain, "column": name, "dtype": dtype, "description": desc})
        for feat in features:
            rows.append({
                "domain": domain,
                "column": feat,
                "dtype": "float",
                "description": "Domain-specific predictor",
            })
        rows.append({
            "domain": domain,
            "column": proxy_col,
            "dtype": "float",
            "description": "Domain-specific alias for proxy feature",
        })

    return rows
