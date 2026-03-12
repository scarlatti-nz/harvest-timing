"""
Generate a focused switching-policy figure package for the harvest timing model.

Outputs:
- Phase diagrams in timber-price / carbon-price space for the paper's four selected ages
- Age/carbon policy maps at fixed timber-price slices
- Switch-region share by age
- Markdown + CSV summaries of switching thresholds and region shares
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import asdict
from typing import Dict, List, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid_config import (
    DEFAULT_PRICE_GRID_SIZE,
    build_results_metadata,
    grid_tag,
    load_results_pickle,
)
from harvest_timing_model import (
    ACTION_DO_NOTHING,
    ACTION_HARVEST_REPLANT,
    ACTION_SWITCH_PERMANENT,
    ModelParameters,
    build_price_grids,
    build_reward_matrix,
    build_state_space,
    build_transition_matrix,
    compute_carbon_curve,
    compute_carbon_flows_averaging,
    compute_carbon_flows_permanent,
    compute_price_quality_factor,
    compute_volume_from_carbon,
    solve_model,
)
from paper_style import (
    ACTION_COLOR_LIST,
    LIGHT_GREY,
    TEXT_GREY,
    add_action_legend,
    add_panel_condition,
    add_panel_label,
    configure_paper_style,
    save_figure,
    style_axes,
)


ACTION_TO_PLOT_CODE = {
    ACTION_DO_NOTHING: 0,
    ACTION_HARVEST_REPLANT: 1,
    ACTION_SWITCH_PERMANENT: 2,
}
NPG_SWITCH = "#4DBBD5"
PLOT_COLORS = ACTION_COLOR_LIST
TIMBER_SLICE_LABELS = ["Low timber price", "Median timber price", "High timber price"]
PANEL_LABELS_2X2 = ["a", "b", "c", "d"]
PANEL_LABELS_1X3 = ["a", "b", "c"]
MAIN_PHASE_DIAGRAM_AGES = [10, 16, 22, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the switching-policy figure and analysis package."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("outputs", "switching_policy"),
        help="Directory for figures, tables, and cached model results.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_PRICE_GRID_SIZE,
        help="Number of timber and carbon price states.",
    )
    parser.add_argument(
        "--rerun-model",
        action="store_true",
        help="Rebuild the model even if a cached pickle exists.",
    )
    return parser.parse_args()


def ensure_output_dirs(base_dir: str, grid_size: int) -> Dict[str, str]:
    root_dir = os.path.join(base_dir, grid_tag(grid_size))
    paths = {
        "root": root_dir,
        "figures": os.path.join(root_dir, "figures"),
        "tables": os.path.join(root_dir, "tables"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def solve_or_load_results(output_dir: str, grid_size: int, rerun_model: bool) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = os.path.join(output_dir, "model_results.pkl")
    if os.path.exists(pickle_path) and not rerun_model:
        try:
            return load_results_pickle(pickle_path, expected_grid_size=grid_size)
        except ValueError as exc:
            print(f"Cached results at {pickle_path} are invalid for grid size {grid_size}: {exc}")
            print("Rebuilding switching-policy results...")

    params = ModelParameters(N_pc=grid_size, N_pt=grid_size)
    price_data = build_price_grids(params)
    state_space = build_state_space(params)
    carbon_curve = compute_carbon_curve(params)
    volume_curve = compute_volume_from_carbon(carbon_curve, params)
    delta_avg = compute_carbon_flows_averaging(carbon_curve, params)
    delta_perm = compute_carbon_flows_permanent(carbon_curve)
    quality_curve = compute_price_quality_factor(params)
    reward_matrix = build_reward_matrix(
        params,
        state_space,
        price_data,
        volume_curve,
        carbon_curve,
        delta_avg,
        delta_perm,
        quality_curve,
    )
    transition_matrix, s_indices, a_indices = build_transition_matrix(
        params, state_space, price_data
    )
    values, sigma = solve_model(
        reward_matrix,
        transition_matrix,
        params.beta,
        method="policy_iteration",
        s_indices=s_indices,
        a_indices=a_indices,
    )

    results = {
        "params": params,
        "price_data": price_data,
        "state_space": state_space,
        "V": values,
        "sigma": sigma,
        "metadata": {
            **build_results_metadata(params, run_name="switching_policy"),
            "parameter_snapshot": asdict(params),
        },
    }
    with open(pickle_path, "wb") as handle:
        pickle.dump(results, handle)
    return results


def stationary_distribution(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    dist = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(max_iter):
        next_dist = dist @ P
        if np.max(np.abs(next_dist - dist)) < tol:
            return next_dist
        dist = next_dist
    raise RuntimeError("Stationary distribution failed to converge.")


def grid_edges(grid: np.ndarray) -> np.ndarray:
    if len(grid) == 1:
        return np.array([grid[0] - 0.5, grid[0] + 0.5])
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    first_edge = grid[0] - (midpoints[0] - grid[0])
    last_edge = grid[-1] + (grid[-1] - midpoints[-1])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def get_plot_code(
    sigma: np.ndarray,
    state_space,
    age: int,
    i_pc: int,
    i_pt: int,
    regime: int,
    rotation: int,
) -> int:
    state = state_space.tuple_to_state[(age, i_pc, i_pt, regime, rotation)]
    return ACTION_TO_PLOT_CODE[int(sigma[state])]


def price_phase_matrix(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    age: int,
    regime: int,
    rotation: int,
) -> np.ndarray:
    matrix = np.zeros((params.N_pc, params.N_pt), dtype=int)
    for i_pc in range(params.N_pc):
        for i_pt in range(params.N_pt):
            matrix[i_pc, i_pt] = get_plot_code(
                sigma, state_space, age, i_pc, i_pt, regime, rotation
            )
    return matrix


def age_carbon_matrix(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    regime: int,
    rotation: int,
    timber_index: int,
) -> np.ndarray:
    matrix = np.zeros((params.N_pc, params.N_a), dtype=int)
    for age in range(params.N_a):
        for i_pc in range(params.N_pc):
            matrix[i_pc, age] = get_plot_code(
                sigma, state_space, age, i_pc, timber_index, regime, rotation
            )
    return matrix


def draw_action_boundaries(
    ax: plt.Axes,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    matrix: np.ndarray,
    color: str = TEXT_GREY,
    linewidth: float = 0.4,
) -> None:
    n_rows, n_cols = matrix.shape
    for row in range(n_rows):
        for col in range(1, n_cols):
            if matrix[row, col - 1] != matrix[row, col]:
                ax.plot(
                    [x_edges[col], x_edges[col]],
                    [y_edges[row], y_edges[row + 1]],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle="butt",
                )
    for row in range(1, n_rows):
        for col in range(n_cols):
            if matrix[row - 1, col] != matrix[row, col]:
                ax.plot(
                    [x_edges[col], x_edges[col + 1]],
                    [y_edges[row], y_edges[row]],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle="butt",
                )


def timber_slice_indices(params: ModelParameters) -> List[int]:
    return [0, params.N_pt // 2, params.N_pt - 1]


def timber_slice_note(price_data: Dict, slice_indices: Sequence[int]) -> str:
    slice_values = [f"${price_data['pt_grid'][idx]:.0f}" for idx in slice_indices]
    return (
        "Slices correspond to "
        + ", ".join(slice_values[:-1])
        + f" and {slice_values[-1]} per m³ from left to right"
    )


def plot_phase_diagram_panels(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    price_data: Dict,
    ages: Sequence[int],
    regime: int,
    rotation: int,
    output_stem: str,
    panel_title_prefix: str | None = None,
) -> None:
    pt_grid = price_data["pt_grid"]
    pc_grid = price_data["pc_grid"]
    pt_edges = grid_edges(pt_grid)
    pc_edges = grid_edges(pc_grid)
    cmap = mcolors.ListedColormap(PLOT_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, age, panel_label in zip(axes.flatten(), ages, PANEL_LABELS_2X2):
        clamped_age = min(age, params.A_max)
        matrix = price_phase_matrix(
            sigma, state_space, params, clamped_age, regime=regime, rotation=rotation
        )
        ax.pcolormesh(
            pt_edges,
            pc_edges,
            matrix,
            cmap=cmap,
            norm=norm,
            shading="flat",
            antialiased=False,
            edgecolors="none",
            linewidth=0.0,
            rasterized=True,
        )
        draw_action_boundaries(ax, pt_edges, pc_edges, matrix)
        add_panel_label(ax, panel_label)
        if panel_title_prefix is None:
            add_panel_condition(ax, f"{clamped_age} years")
        else:
            ax.set_title(f"{panel_title_prefix}: {clamped_age} years", loc="left", pad=4)
        ax.set_xlabel("Timber price ($ per m³)")
        ax.set_ylabel(r"Carbon price (\$ per tCO$_2$)")
        style_axes(ax)

    fig.subplots_adjust(left=0.10, right=0.995, top=0.98, bottom=0.125, wspace=0.12, hspace=0.22)
    add_action_legend(fig, left=0.10, bottom=0.015, width=0.895, height=0.035)
    save_figure(fig, output_stem)


def plot_age_carbon_frontier(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    price_data: Dict,
    regime: int,
    rotation: int,
    output_stem: str,
) -> None:
    pc_grid = price_data["pc_grid"]
    pc_edges = grid_edges(pc_grid)
    age_edges = np.arange(params.N_a + 1) - 0.5
    cmap = mcolors.ListedColormap(PLOT_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    slice_indices = timber_slice_indices(params)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, label, timber_index, panel_label in zip(
        axes, TIMBER_SLICE_LABELS, slice_indices, PANEL_LABELS_1X3
    ):
        matrix = age_carbon_matrix(
            sigma,
            state_space,
            params,
            regime=regime,
            rotation=rotation,
            timber_index=timber_index,
        )
        ax.pcolormesh(
            age_edges,
            pc_edges,
            matrix,
            cmap=cmap,
            norm=norm,
            shading="flat",
            antialiased=False,
            edgecolors="none",
            linewidth=0.0,
            rasterized=True,
        )
        draw_action_boundaries(ax, age_edges, pc_edges, matrix)
        ax.axvline(
            params.carbon_credit_max_age,
            color=LIGHT_GREY,
            linewidth=0.6,
            linestyle="--",
            alpha=1.0,
        )
        add_panel_label(ax, panel_label)
        add_panel_condition(ax, label)
        ax.set_xlabel("Stand age (years)")
        style_axes(ax)
    axes[0].set_ylabel(r"Carbon price (\$ per tCO$_2$)")

    fig.text(
        0.5,
        0.05,
        timber_slice_note(price_data, slice_indices),
        ha="center",
        va="bottom",
        fontsize=6,
        color=TEXT_GREY,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.95, bottom=0.21, wspace=0.06)
    add_action_legend(fig, left=0.10, bottom=0.05, width=0.895, height=0.035)
    save_figure(fig, output_stem)


def compute_switch_share_table(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    price_data: Dict,
    regime: int = 0,
) -> pd.DataFrame:
    pc_stationary = stationary_distribution(price_data["Pc"])
    pt_stationary = stationary_distribution(price_data["Pt"])
    joint_weights = np.outer(pc_stationary, pt_stationary)
    records = []

    for rotation in (1, 2):
        for age in range(params.N_a):
            matrix = price_phase_matrix(
                sigma,
                state_space,
                params,
                age,
                regime=regime,
                rotation=rotation,
            )
            switch_mask = matrix == ACTION_TO_PLOT_CODE[ACTION_SWITCH_PERMANENT]
            records.append(
                {
                    "rotation": rotation,
                    "age": age,
                    "switch_share_unweighted": float(np.mean(switch_mask)),
                    "switch_share_weighted": float(np.sum(joint_weights[switch_mask])),
                }
            )

    return pd.DataFrame.from_records(records)


def plot_switch_share_by_age(
    share_df: pd.DataFrame,
    params: ModelParameters,
    output_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = {1: "First rotation", 2: "Later rotations"}
    colors = {1: TEXT_GREY, 2: NPG_SWITCH}
    linestyles = {1: "-", 2: "--"}

    for rotation in (1, 2):
        data = share_df.loc[share_df["rotation"] == rotation]
        ax.plot(
            data["age"],
            data["switch_share_weighted"],
            label=labels[rotation],
            color=colors[rotation],
            linestyle=linestyles[rotation],
            linewidth=1.1,
        )

    ax.axvline(
        params.carbon_credit_max_age,
        color=LIGHT_GREY,
        linewidth=0.6,
        linestyle="--",
        alpha=1.0,
    )
    ax.set_xlabel("Stand age (years)")
    ax.set_ylabel("Probability mass in switch region")
    ax.set_xlim(0, params.A_max)
    ax.set_ylim(0, None)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=TEXT_GREY, handlelength=2.3)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.98, bottom=0.20)
    save_figure(fig, output_stem)


def extract_threshold_table(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
    price_data: Dict,
    ages: Sequence[int],
    regime: int = 0,
) -> pd.DataFrame:
    pc_grid = price_data["pc_grid"]
    pt_grid = price_data["pt_grid"]
    pc_stationary = stationary_distribution(price_data["Pc"])
    slice_indices = timber_slice_indices(params)
    records = []

    for rotation in (1, 2):
        for age in ages:
            clamped_age = min(age, params.A_max)
            for timber_label, timber_index in zip(TIMBER_SLICE_LABELS, slice_indices):
                actions = np.array(
                    [
                        get_plot_code(
                            sigma,
                            state_space,
                            clamped_age,
                            i_pc,
                            timber_index,
                            regime,
                            rotation,
                        )
                        for i_pc in range(params.N_pc)
                    ]
                )
                switch_idx = np.where(actions == ACTION_TO_PLOT_CODE[ACTION_SWITCH_PERMANENT])[0]
                harvest_idx = np.where(actions == ACTION_TO_PLOT_CODE[ACTION_HARVEST_REPLANT])[0]
                hold_idx = np.where(actions == ACTION_TO_PLOT_CODE[ACTION_DO_NOTHING])[0]
                records.append(
                    {
                        "rotation": rotation,
                        "age": clamped_age,
                        "timber_slice": timber_label,
                        "timber_price": float(pt_grid[timber_index]),
                        "min_switch_carbon_price": (
                            float(pc_grid[switch_idx.min()]) if switch_idx.size else np.nan
                        ),
                        "max_switch_carbon_price": (
                            float(pc_grid[switch_idx.max()]) if switch_idx.size else np.nan
                        ),
                        "max_harvest_carbon_price": (
                            float(pc_grid[harvest_idx.max()]) if harvest_idx.size else np.nan
                        ),
                        "min_hold_carbon_price": (
                            float(pc_grid[hold_idx.min()]) if hold_idx.size else np.nan
                        ),
                        "switch_share_along_carbon_grid": (
                            float(switch_idx.size / params.N_pc) if params.N_pc else 0.0
                        ),
                        "switch_share_stationary_carbon": (
                            float(pc_stationary[switch_idx].sum()) if switch_idx.size else 0.0
                        ),
                    }
                )

    return pd.DataFrame.from_records(records)


def find_rotation_difference_ages(
    sigma: np.ndarray,
    state_space,
    params: ModelParameters,
) -> List[int]:
    diff_ages: List[int] = []
    for age in range(params.N_a):
        first_rotation = price_phase_matrix(
            sigma, state_space, params, age, regime=0, rotation=1
        )
        later_rotation = price_phase_matrix(
            sigma, state_space, params, age, regime=0, rotation=2
        )
        if np.any(first_rotation != later_rotation):
            diff_ages.append(age)
    return diff_ages


def choose_later_rotation_ages(
    rotation_difference_ages: Sequence[int],
    params: ModelParameters,
) -> List[int]:
    if not rotation_difference_ages:
        return [0, params.carbon_credit_max_age // 2, params.carbon_credit_max_age, min(30, params.A_max)]

    candidate_ages = [
        rotation_difference_ages[0],
        rotation_difference_ages[len(rotation_difference_ages) // 2],
        rotation_difference_ages[-1],
        params.carbon_credit_max_age,
    ]
    chosen: List[int] = []
    for age in candidate_ages:
        if age not in chosen:
            chosen.append(age)

    fill_age = 0
    while len(chosen) < 4 and fill_age <= params.A_max:
        if fill_age not in chosen:
            chosen.append(fill_age)
        fill_age += 1

    return chosen[:4]


def summarize_switch_shares(share_df: pd.DataFrame, rotation: int) -> Dict[str, float]:
    subset = share_df.loc[share_df["rotation"] == rotation].copy()
    positive = subset.loc[subset["switch_share_weighted"] > 0]
    if positive.empty:
        return {
            "first_age": np.nan,
            "last_age": np.nan,
            "peak_age": np.nan,
            "peak_share": 0.0,
        }

    peak_row = subset.loc[subset["switch_share_weighted"].idxmax()]
    return {
        "first_age": int(positive["age"].min()),
        "last_age": int(positive["age"].max()),
        "peak_age": int(peak_row["age"]),
        "peak_share": float(peak_row["switch_share_weighted"]),
    }


def format_currency(value: float) -> str:
    if np.isnan(value):
        return "Not present"
    return f"${value:,.0f}"


def write_summary_markdown(
    output_path: str,
    params: ModelParameters,
    figure_dir: str,
    share_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    rotation_difference_ages: Sequence[int],
) -> None:
    rot1 = summarize_switch_shares(share_df, rotation=1)
    rot2 = summarize_switch_shares(share_df, rotation=2)
    focus_rows = threshold_df.loc[
        (threshold_df["rotation"] == 1)
        & (threshold_df["age"].isin([16, 22, 30]))
    ].copy()
    focus_rows["switch_band"] = focus_rows.apply(
        lambda row: (
            "Not present"
            if np.isnan(row["min_switch_carbon_price"])
            else f"{format_currency(row['min_switch_carbon_price'])} to "
            f"{format_currency(row['max_switch_carbon_price'])}"
        ),
        axis=1,
    )

    lines = [
        "# Switching policy package",
        "",
        "## Scenario",
        "- Switching-enabled baseline using `ModelParameters` defaults.",
        f"- Price grid: {params.N_pt} timber states x {params.N_pc} carbon states.",
        f"- Switching cost: ${params.switch_cost:.0f}/ha.",
        f"- Permanent-regime harvest penalty: ${params.harvest_penalty_per_m3:.0f}/m³.",
        f"- Averaging threshold age: {params.carbon_credit_max_age}.",
        "",
        "## Included figures",
        f"- Main phase diagram: `{os.path.join(figure_dir, 'main_phase_diagram_averaging_first_rotation.png')}`",
        f"- Switching frontier by age: `{os.path.join(figure_dir, 'switching_frontier_by_age.png')}`",
        f"- Phase diagram after the first harvest: `{os.path.join(figure_dir, 'phase_diagram_later_rotations.png')}`",
        f"- Switch region share by age: `{os.path.join(figure_dir, 'switch_region_share_by_age.png')}`",
        "",
        "## Key switching diagnostics",
        (
            f"- First rotation: the switch region appears at age {rot1['first_age']}, "
            f"last appears at age {rot1['last_age']}, and peaks at age {rot1['peak_age']} "
            f"with stationary weight {rot1['peak_share']:.3f}."
        ),
        (
            f"- Later rotations: the switch region appears at age {rot2['first_age']}, "
            f"last appears at age {rot2['last_age']}, and peaks at age {rot2['peak_age']} "
            f"with stationary weight {rot2['peak_share']:.3f}."
        ),
        (
            "- Under the current parameterization, first and later rotations differ only at ages "
            + ", ".join(str(age) for age in rotation_difference_ages)
            + "."
            if rotation_difference_ages
            else "- Under the current parameterization, first and later rotations produce the same policy map."
        ),
        "- The later-rotation map should be interpreted relative to the age-16 shortfall penalty for switching before the averaging threshold.",
        "",
        "## Switching bands at selected ages",
        "| Age | Timber slice | Switch carbon-price band | Max harvest carbon price |",
        "| --- | --- | --- | --- |",
    ]

    for _, row in focus_rows.iterrows():
        lines.append(
            f"| {int(row['age'])} | {row['timber_slice']} | {row['switch_band']} | "
            f"{format_currency(row['max_harvest_carbon_price'])} |"
        )

    lines.extend(
        [
            "",
            "## Data tables",
            "- `tables/switch_region_share_by_age.csv`",
            "- `tables/switch_thresholds.csv`",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_paper_style()
    paths = ensure_output_dirs(args.output_dir, args.grid_size)
    results = solve_or_load_results(
        output_dir=paths["root"],
        grid_size=args.grid_size,
        rerun_model=args.rerun_model,
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
        output_stem=os.path.join(
            paths["figures"], "main_phase_diagram_averaging_first_rotation"
        ),
        panel_title_prefix="Stand age",
    )

    plot_age_carbon_frontier(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        regime=0,
        rotation=1,
        output_stem=os.path.join(paths["figures"], "switching_frontier_by_age"),
    )

    rotation_difference_ages = find_rotation_difference_ages(
        sigma=sigma,
        state_space=state_space,
        params=params,
    )
    later_rotation_ages = choose_later_rotation_ages(
        rotation_difference_ages=rotation_difference_ages,
        params=params,
    )
    plot_phase_diagram_panels(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        ages=later_rotation_ages,
        regime=0,
        rotation=2,
        output_stem=os.path.join(paths["figures"], "phase_diagram_later_rotations"),
    )

    share_df = compute_switch_share_table(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        regime=0,
    )
    share_df.to_csv(
        os.path.join(paths["tables"], "switch_region_share_by_age.csv"),
        index=False,
    )
    plot_switch_share_by_age(
        share_df=share_df,
        params=params,
        output_stem=os.path.join(paths["figures"], "switch_region_share_by_age"),
    )

    threshold_df = extract_threshold_table(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        ages=sorted(set(MAIN_PHASE_DIAGRAM_AGES).union(later_rotation_ages)),
        regime=0,
    )
    threshold_df.to_csv(
        os.path.join(paths["tables"], "switch_thresholds.csv"),
        index=False,
    )
    write_summary_markdown(
        output_path=os.path.join(paths["root"], "switching_policy_summary.md"),
        params=params,
        figure_dir=paths["figures"],
        share_df=share_df,
        threshold_df=threshold_df,
        rotation_difference_ages=rotation_difference_ages,
    )

    print(f"Switching-policy package written to {paths['root']}")


if __name__ == "__main__":
    main()
