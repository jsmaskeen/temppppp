from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ethics_experiment.experiment import run_full_experiment
from ethics_experiment.reporting import (
    generate_all_plots,
    write_all_tables,
)


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "experiment_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    outputs_dir = PROJECT_ROOT / "outputs"
    tables_dir = outputs_dir / "tables"
    plots_dir = outputs_dir / "plots"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiment_result = run_full_experiment(config)
    results_df: pd.DataFrame = experiment_result["results_df"]
    descriptives_df: pd.DataFrame = experiment_result["descriptives_df"]
    frontier_df: pd.DataFrame = experiment_result["frontier_df"]
    plot_payload = experiment_result["plot_payload"]

    write_all_tables(
        results_df=results_df,
        descriptives_df=descriptives_df,
        frontier_df=frontier_df,
        config=config,
        tables_dir=tables_dir,
    )

    generate_all_plots(
        results_df=results_df,
        frontier_df=frontier_df,
        plot_payload=plot_payload,
        plots_dir=plots_dir,
    )

    print("Experiment completed.")
    print(f"Tables: {tables_dir}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
