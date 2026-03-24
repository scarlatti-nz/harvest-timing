"""
Config-driven runner for the canonical scenario bundle.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

from build_paper_figures import (
    build_paper_figure_bundle,
    load_switching_policy_results,
    paper_figure_paths,
    resolve_switching_policy_dir,
)
from grid_config import model_results_path
from harvest_timing_model import ModelParameters
from harvest_timing_model import main as run_model
from plot_results import main as plot_results
from plot_utility_histograms import load_utility_scenario_results
from plot_utility_histograms import main as plot_utility_histograms
from run_all_config import (
    DEFAULT_RUN_ALL_CONFIG_PATH,
    ModelRunJob,
    PaperFigureJob,
    PlotJob,
    RunAllConfig,
    UtilityJob,
    describe_run_all_config,
    load_run_all_config,
)
from scenario_registry import get_model_scenario, get_utility_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical harvest timing workflow")
    parser.add_argument(
        "--config",
        default=DEFAULT_RUN_ALL_CONFIG_PATH,
        help=f"Path to workflow config (default: {DEFAULT_RUN_ALL_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded workflow and exit without running jobs.",
    )
    return parser.parse_args()


def _print_job_header(title: str) -> None:
    print(f"\n\n{'#' * 80}")
    print(title)
    print(f"{'#' * 80}\n")


def _ensure_result_available(
    run_name: str,
    grid_size: int,
    produced_results: set[tuple[str, int]],
    consumer: str,
) -> None:
    if (run_name, grid_size) in produced_results:
        return

    pickle_path = model_results_path(run_name, grid_size)
    if os.path.exists(pickle_path):
        return

    raise FileNotFoundError(
        f"{consumer} requires {pickle_path}, but that result was not produced earlier "
        "in this workflow and does not already exist on disk."
    )


def run_model_job(job: ModelRunJob) -> None:
    scenario = get_model_scenario(job.scenario)
    _print_job_header(
        f"RUNNING MODEL SCENARIO: {job.scenario} -> {job.run_name} "
        f"({job.grid_size}x{job.grid_size})"
    )

    params = ModelParameters(
        N_pt=job.grid_size,
        N_pc=job.grid_size,
        **scenario["overrides"],
    )
    args = Namespace(
        temp_dir=job.run_name,
        grid_size=job.grid_size,
        model_scenario=job.scenario,
        sanity_checks=False,
    )
    run_model(args=args, params=params)


def run_utility_job(job: UtilityJob, produced_results: set[tuple[str, int]]) -> None:
    for scenario in get_utility_scenarios(job.scenario_set):
        _ensure_result_available(
            scenario["default_run_name"],
            job.grid_size,
            produced_results,
            consumer=f"utility job '{job.scenario_set}'",
        )

    _print_job_header(
        f"RUNNING UTILITY JOB: {job.scenario_set} ({job.grid_size}x{job.grid_size})"
    )
    plot_utility_histograms(
        Namespace(
            scenario_set=job.scenario_set,
            rerun=True,
            grid_size=job.grid_size,
            output_dir=job.output_dir,
            pickle_path=None,
        )
    )


def run_plot_job(job: PlotJob, produced_results: set[tuple[str, int]]) -> None:
    _ensure_result_available(
        job.run_name,
        job.grid_size,
        produced_results,
        consumer=f"plot job '{job.kind}:{job.run_name}'",
    )

    _print_job_header(
        f"RUNNING PLOT JOB: {job.kind}:{job.run_name} ({job.grid_size}x{job.grid_size})"
    )
    if job.kind == "results":
        plot_results(
            Namespace(
                temp_dir=job.run_name,
                grid_size=job.grid_size,
                pickle_path=None,
                output_dir=job.output_dir,
            )
        )
        return

    raise ValueError(f"Unsupported plot job kind: {job.kind}")


def resolve_paper_switching_policy_dir(
    job: PaperFigureJob,
    config: RunAllConfig,
) -> str:
    if job.switching_policy_dir is not None:
        return job.switching_policy_dir

    matching_runs = tuple(
        model_run
        for model_run in config.model_runs
        if model_run.scenario == "switching-policy" and model_run.grid_size == job.grid_size
    )
    if len(matching_runs) == 1:
        return os.path.dirname(model_results_path(matching_runs[0].run_name, job.grid_size))
    if len(matching_runs) > 1:
        run_names = ", ".join(model_run.run_name for model_run in matching_runs)
        raise ValueError(
            "paper figure job has multiple switching-policy model runs at "
            f"{job.grid_size}x{job.grid_size} ({run_names}). "
            "Set paper_figure_jobs[].switching_policy_dir explicitly."
        )

    return resolve_switching_policy_dir(None, job.grid_size)


def run_paper_figure_job(job: PaperFigureJob, config: RunAllConfig) -> None:
    paper_output_dir = job.output_dir or os.path.join("outputs", "paper_figures")
    paths = paper_figure_paths(paper_output_dir, job.grid_size)
    switch_dir = resolve_paper_switching_policy_dir(job, config)
    switching_policy_pickle, _ = load_switching_policy_results(
        switch_dir,
        job.grid_size,
    )

    _print_job_header(
        f"REFRESHING PAPER UTILITY CACHES: paper ({job.grid_size}x{job.grid_size})"
    )
    load_utility_scenario_results(
        scenario_set="paper",
        grid_size=job.grid_size,
        pickle_path=switching_policy_pickle,
        cache_dir=paths["utility_cache"],
        rerun=True,
    )

    _print_job_header(
        f"RUNNING PAPER FIGURE JOB: paper_bundle ({job.grid_size}x{job.grid_size})"
    )
    build_paper_figure_bundle(
        output_dir=paper_output_dir,
        grid_size=job.grid_size,
        switching_policy_dir=switch_dir,
    )


def execute_workflow(config: RunAllConfig) -> None:
    produced_results: set[tuple[str, int]] = set()

    for job in config.model_runs:
        run_model_job(job)
        produced_results.add((job.run_name, job.grid_size))

    for job in config.utility_jobs:
        run_utility_job(job, produced_results)

    for job in config.paper_figure_jobs:
        run_paper_figure_job(job, config)

    for job in config.plot_jobs:
        run_plot_job(job, produced_results)



def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    config = load_run_all_config(args.config)
    print(f"Loaded workflow config from {args.config}")

    if getattr(args, "dry_run", False):
        print(describe_run_all_config(config))
        return

    execute_workflow(config)


if __name__ == "__main__":
    main()
