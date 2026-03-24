"""
Build the paper-ready figure bundle.
"""

from __future__ import annotations

import argparse
import os

from grid_config import grid_tag, load_results_pickle, switching_policy_output_dir
from paper_style import configure_paper_style
from plot_results import plot_accounting_comparison, plot_price_paths, plot_value_function
from plot_switching_policy import (
    LATER_ROTATION_PHASE_DIAGRAM_AGES,
    LATER_ROTATION_PHASE_DIAGRAM_TITLE_SUFFIXES,
    MAIN_PHASE_DIAGRAM_ANNOTATIONS,
    MAIN_PHASE_DIAGRAM_TITLE_SUFFIXES,
    MAIN_PHASE_DIAGRAM_AGES,
    compute_switch_share_table,
    plot_age_carbon_frontier,
    plot_phase_diagram_panels,
    plot_switch_share_by_age,
)
from plot_utility_histograms import (
    load_utility_scenario_results,
    print_summary_statistics,
    save_utility_figures,
)
from scenario_registry import resolve_results_model_scenario


DEFAULT_PAPER_FIGURE_GRID_SIZE = 201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready figures")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "paper_figures"),
        help="Root directory for the paper figure bundle",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_PAPER_FIGURE_GRID_SIZE,
        help=(
            "Number of timber and carbon price states "
            f"(default: {DEFAULT_PAPER_FIGURE_GRID_SIZE})"
        ),
    )
    parser.add_argument(
        "--switching-policy-dir",
        default=None,
        help=(
            "Existing directory containing the switching-policy model_results.pkl "
            "cache to use as read-only input. "
            "Defaults to outputs/switching_policy/grid_<N>x<N> for the chosen grid."
        ),
    )
    return parser.parse_args()


def paper_figure_paths(base_dir: str, grid_size: int) -> dict[str, str]:
    root_dir = os.path.join(base_dir, grid_tag(grid_size))
    appendix_dir = os.path.join(root_dir, "appendix")
    paths = {
        "root": root_dir,
        "main": os.path.join(root_dir, "main"),
        "appendix": appendix_dir,
        "utility_cache": os.path.join(appendix_dir, "utility_cache"),
    }
    return paths


def ensure_dirs(base_dir: str, grid_size: int) -> dict[str, str]:
    paths = paper_figure_paths(base_dir, grid_size)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def resolve_switching_policy_dir(
    switching_dir: str | None,
    grid_size: int,
) -> str:
    if switching_dir is not None:
        return switching_dir
    return switching_policy_output_dir(grid_size)


def load_switching_policy_results(
    switching_dir: str,
    grid_size: int,
) -> tuple[str, dict]:
    pickle_path = os.path.join(switching_dir, "model_results.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            "--switching-policy-dir requires an existing switching-policy cache at "
            f"{pickle_path}."
        )

    results = load_results_pickle(pickle_path, expected_grid_size=grid_size)
    try:
        model_scenario = resolve_results_model_scenario(results)
    except ValueError as exc:
        raise ValueError(
            "--switching-policy-dir requires a switching-policy scenario cache, "
            f"but {pickle_path} could not be classified: {exc}"
        ) from exc
    if model_scenario != "switching-policy":
        raise ValueError(
            "--switching-policy-dir requires a switching-policy scenario cache, found "
            f"{model_scenario} in {pickle_path}."
        )
    return pickle_path, results


def build_switching_figures(
    main_dir: str,
    switching_results: dict,
) -> None:
    params = switching_results["params"]
    price_data = switching_results["price_data"]
    state_space = switching_results["state_space"]
    sigma = switching_results["sigma"]

    plot_phase_diagram_panels(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        ages=MAIN_PHASE_DIAGRAM_AGES,
        regime=0,
        rotation=1,
        output_stem=os.path.join(main_dir, "main_phase_diagram_averaging_first_rotation"),
        panel_title_prefix="Stand age",
        panel_title_suffixes=MAIN_PHASE_DIAGRAM_TITLE_SUFFIXES,
        panel_annotations=MAIN_PHASE_DIAGRAM_ANNOTATIONS,
        smooth_regions=True,
    )
    plot_age_carbon_frontier(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        regime=0,
        rotation=1,
        output_stem=os.path.join(main_dir, "switching_frontier_by_age"),
    )

    plot_phase_diagram_panels(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        ages=LATER_ROTATION_PHASE_DIAGRAM_AGES,
        regime=0,
        rotation=2,
        output_stem=os.path.join(main_dir, "phase_diagram_later_rotations"),
        panel_title_prefix="Stand age",
        panel_title_suffixes=LATER_ROTATION_PHASE_DIAGRAM_TITLE_SUFFIXES,
        smooth_regions=True,
    )

    share_df = compute_switch_share_table(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        regime=0,
    )
    plot_switch_share_by_age(
        share_df=share_df,
        params=params,
        output_stem=os.path.join(main_dir, "switch_region_share_by_age"),
    )


def build_appendix_figures(
    appendix_dir: str,
    utility_results: list[dict],
    switching_results: dict,
) -> None:
    params = switching_results["params"]
    state_space = switching_results["state_space"]
    price_data = switching_results["price_data"]
    values = switching_results["V"]

    plot_price_paths(
        params=params,
        n_paths=1000,
        n_periods=51,
        save_path=os.path.join(appendix_dir, "price_paths_distribution.png"),
    )
    plot_value_function(
        V=values,
        state_space=state_space,
        params=params,
        price_data=price_data,
        max_age_plot=50,
        save_path=os.path.join(appendix_dir, "value_function.png"),
    )
    print_summary_statistics(utility_results)
    save_utility_figures(
        utility_results,
        scenario_set="paper",
        output_dir=appendix_dir,
    )


def build_paper_figure_bundle(
    *,
    output_dir: str,
    grid_size: int,
    switching_policy_dir: str | None = None,
) -> str:
    configure_paper_style()
    paths = paper_figure_paths(output_dir, grid_size)
    switch_dir = resolve_switching_policy_dir(switching_policy_dir, grid_size)
    switching_pickle_path, switching_results = load_switching_policy_results(
        switch_dir,
        grid_size,
    )
    utility_results, _ = load_utility_scenario_results(
        scenario_set="paper",
        grid_size=grid_size,
        pickle_path=switching_pickle_path,
        cache_dir=paths["utility_cache"],
        require_cached=True,
    )

    ensure_dirs(output_dir, grid_size)
    build_switching_figures(paths["main"], switching_results)
    plot_accounting_comparison(
        excel_path=os.path.join("data", "growth_curves.xlsx"),
        params=switching_results["params"],
        save_path=os.path.join(paths["main"], "accounting_comparison.png"),
    )
    build_appendix_figures(
        appendix_dir=paths["appendix"],
        utility_results=utility_results,
        switching_results=switching_results,
    )
    print(f"Paper figures written to {paths['root']}")
    return paths["root"]


def main() -> None:
    args = parse_args()
    try:
        build_paper_figure_bundle(
            output_dir=args.output_dir,
            grid_size=args.grid_size,
            switching_policy_dir=args.switching_policy_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
