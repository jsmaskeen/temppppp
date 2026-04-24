# Ethical Autonomous Agents: Cross-Domain Experiment

This project implements a reproducible study of ethical failure and ethical mitigation in autonomous decision agents across three domains:

- Loan approval
- Hiring shortlisting
- Medical triage

We generates synthetic data, trains a biased baseline agent, applies mitigations, and reports fairness, utility, and human-oversight outcomes.

## Variants implemented

1. `baseline`: trains on biased historical labels with sensitive and proxy features.
2. `no_sensitive_proxy`: removes sensitive/proxy features but still trains on biased historical labels.
3. `label_correction_only`: trains on true outcomes but keeps sensitive/proxy features.
4. `ethical_no_oversight`: trains on true outcomes, removes sensitive/proxy features, uses fairness-constrained thresholding.
5. `ethical_with_oversight`: adds human oversight for uncertain predictions.

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full experiment:

```bash
python run_experiment.py
```

3. Check outputs:

- `outputs/tables/`
- `outputs/plots/`
- `report/methodology.md`
- `report/discussion.md`

## Reproducibility

All experiment settings are in `config/experiment_config.json`.
The default setup runs five seeds: `11, 22, 33, 44, 55`.
