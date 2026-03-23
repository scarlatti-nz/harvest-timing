"""
Build the paper-ready figure bundle.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from grid_config import grid_tag
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
    solve_or_load_results,
)
from plot_utility_histograms import main as plot_utility_histograms


DEFAULT_PAPER_FIGURE_GRID_SIZE = 201


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready figures")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "paper_figures"),
        help="Root directory for the paper figure bundle",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun cached model solve and utility simulations",
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
    return parser.parse_args()


def ensure_dirs(base_dir: str, grid_size: int) -> dict[str, str]:
    root_dir = os.path.join(base_dir, grid_tag(grid_size))
    paths = {
        "root": root_dir,
        "main": os.path.join(root_dir, "main"),
        "appendix": os.path.join(root_dir, "appendix"),
        "switching_policy": os.path.join(root_dir, "switching_policy"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def build_switching_figures(
    main_dir: str,
    switching_dir: str,
    grid_size: int,
    rerun: bool,
) -> dict:
    results = solve_or_load_results(
        output_dir=switching_dir,
        grid_size=grid_size,
        rerun_model=rerun,
    )
    params = results["params"]
    price_data = results["price_data"]
    state_space = results["state_space"]
    sigma = results["sigma"]

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
    return results


def build_appendix_figures(
    appendix_dir: str,
    switching_dir: str,
    switching_results: dict,
    grid_size: int,
    rerun: bool,
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
    plot_utility_histograms(
        SimpleNamespace(
            rerun=rerun,
            scenario_set="baseline",
            output_dir=appendix_dir,
            grid_size=grid_size,
            pickle_path=os.path.join(switching_dir, "model_results.pkl"),
        )
    )


def main() -> None:
    args = parse_args()
    configure_paper_style()
    paths = ensure_dirs(args.output_dir, args.grid_size)

    switching_results = build_switching_figures(
        paths["main"],
        switching_dir=paths["switching_policy"],
        grid_size=args.grid_size,
        rerun=args.rerun,
    )
    plot_accounting_comparison(
        excel_path=os.path.join("data", "growth_curves.xlsx"),
        params=switching_results["params"],
        save_path=os.path.join(paths["main"], "accounting_comparison.png"),
    )
    build_appendix_figures(
        appendix_dir=paths["appendix"],
        switching_dir=paths["switching_policy"],
        switching_results=switching_results,
        grid_size=args.grid_size,
        rerun=args.rerun,
    )
    print(f"Paper figures written to {paths['root']}")


if __name__ == "__main__":
    main()
