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
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd

from grid_config import (
    DEFAULT_PRICE_GRID_SIZE,
    build_results_metadata,
    grid_tag,
    infer_run_name_from_pickle_path,
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
    build_transition_representation,
    compute_carbon_curve,
    compute_carbon_flows_averaging,
    compute_carbon_flows_permanent,
    compute_price_quality_factor,
    compute_volume_from_carbon,
    solve_model,
    solve_model_matrix_free,
)
from paper_style import (
    ACTION_COLORS,
    ACTION_COLOR_LIST,
    LIGHT_GREY,
    TEXT_BLACK,
    TEXT_GREY,
    add_action_legend,
    add_panel_condition,
    add_panel_label,
    configure_paper_style,
    save_figure,
    style_axes,
)
from scenario_registry import resolve_results_model_scenario


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
MAIN_PHASE_DIAGRAM_TITLE_SUFFIXES = [
    "0-15 years",
    "16 years",
    "22 years",
    "30 years",
]
MAIN_PHASE_DIAGRAM_ANNOTATIONS = [
    None,
    "Optimal to switch to permanent\nat end of averaging period\nunless carbon price is low",
    [
        {
            "text": "Switch if carbon price is high\nand timber price is sufficiently low",
            "x": 0.28,
            "y": 0.88,
            "ha": "center",
            "va": "top",
            "fontsize": 7.1,
        },
        {
            "text": "Harvest only if\nthe price is very high",
            "x": 0.67,
            "y": 0.50,
            "ha": "center",
            "va": "center",
            "fontsize": 7.1,
            "line_from": (0.76, 0.50),
            "line_to": (0.90, 0.52),
        },
    ],
    {
        "text": "Harvest when the price is\nsufficiently high",
        "x": 0.75,
        "y": 0.50,
        "ha": "center",
        "va": "center",
        "fontsize": 7.1,
    },
]
LATER_ROTATION_PHASE_DIAGRAM_AGES = [1, 4, 15, 16]
LATER_ROTATION_PHASE_DIAGRAM_TITLE_SUFFIXES = [
    "1 years",
    "4 years",
    "5-15 years",
    "16 years",
]
TIMBER_PRICE_MIN = 100.0
TIMBER_PRICE_MAX = 220.0
MAIN_PHASE_DIAGRAM_Y_MIN = 0.0
MAIN_PHASE_DIAGRAM_Y_MAX = 150.0
MAIN_PHASE_SWITCH_LOWESS_FRAC = 0.24
MAIN_PHASE_HARVEST_LOWESS_FRAC = 0.28
MAIN_PHASE_SMOOTH_POINTS = 1400


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
            results = load_results_pickle(pickle_path, expected_grid_size=grid_size)
            model_scenario = resolve_results_model_scenario(results)
            if model_scenario != "switching-policy":
                raise ValueError(
                    f"expected switching-policy scenario cache, found {model_scenario}"
                )
            return results
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
    transition_matrix, s_indices, a_indices, used_matrix_free = build_transition_representation(
        params, state_space, price_data
    )
    if used_matrix_free:
        values, sigma = solve_model_matrix_free(
            reward_matrix,
            transition_matrix,
            params.beta,
        )
    else:
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
            **build_results_metadata(
                params,
                run_name=infer_run_name_from_pickle_path(pickle_path),
                model_scenario="switching-policy",
            ),
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


def _compress_constant_segments(
    u_vals: np.ndarray,
    v_vals: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(u_vals) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    groups = []
    start = 0
    for idx in range(1, len(v_vals) + 1):
        if idx == len(v_vals) or abs(v_vals[idx] - v_vals[start]) > tol:
            u_group = u_vals[start:idx]
            groups.append((u_group[0], u_group[-1], u_group.mean(), v_vals[start]))
            start = idx

    u_lo = np.array([group[0] for group in groups], dtype=float)
    u_hi = np.array([group[1] for group in groups], dtype=float)
    u_mid = np.array([group[2] for group in groups], dtype=float)
    v_mid = np.array([group[3] for group in groups], dtype=float)
    return u_lo, u_hi, u_mid, v_mid


def _visible_row_slice(
    y_edges: np.ndarray,
    *,
    y_min: float,
    y_max: float,
) -> tuple[slice, np.ndarray]:
    row_mask = (y_edges[:-1] < y_max) & (y_edges[1:] > y_min)
    row_indices = np.flatnonzero(row_mask)
    if row_indices.size == 0:
        return slice(0, len(y_edges) - 1), y_edges
    row_start = int(row_indices.min())
    row_stop = int(row_indices.max()) + 1
    return slice(row_start, row_stop), y_edges[row_start : row_stop + 1]


def _extract_switch_boundary(
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> dict[str, np.ndarray | str] | None:
    n_rows = matrix.shape[0]
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    xs: list[float] = []
    ys: list[float] = []
    cols: list[int] = []
    lower_count = 0
    upper_count = 0
    for col in range(matrix.shape[1]):
        rows = np.flatnonzero(matrix[:, col] == ACTION_TO_PLOT_CODE[ACTION_SWITCH_PERMANENT])
        if rows.size == 0:
            continue
        xs.append(float(x_centers[col]))
        cols.append(col)
        touches_bottom = int(rows.min()) == 0
        touches_top = int(rows.max()) == n_rows - 1
        if touches_bottom and not touches_top:
            lower_count += 1
        elif touches_top and not touches_bottom:
            upper_count += 1

    if not xs:
        return None

    if lower_count > upper_count:
        orientation = "lower"
    elif upper_count > lower_count:
        orientation = "upper"
    else:
        switch_rows, _ = np.where(matrix == ACTION_TO_PLOT_CODE[ACTION_SWITCH_PERMANENT])
        orientation = (
            "lower"
            if switch_rows.size and float(switch_rows.mean()) < 0.5 * (n_rows - 1)
            else "upper"
        )

    xs = []
    ys = []
    cols = []
    for col in range(matrix.shape[1]):
        rows = np.flatnonzero(matrix[:, col] == ACTION_TO_PLOT_CODE[ACTION_SWITCH_PERMANENT])
        if rows.size == 0:
            continue
        xs.append(float(x_centers[col]))
        if orientation == "lower":
            ys.append(float(y_edges[int(rows.max()) + 1]))
        else:
            ys.append(float(y_edges[int(rows.min())]))
        cols.append(col)

    col_idx = np.array(cols, dtype=int)
    return {
        "u": np.array(xs, dtype=float),
        "v": np.array(ys, dtype=float),
        "u_lo": np.array([x_edges[int(col_idx.min())]], dtype=float),
        "u_hi": np.array([x_edges[int(col_idx.max()) + 1]], dtype=float),
        "orientation": orientation,
    }


def _extract_harvest_left_boundary(
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> dict[str, np.ndarray] | None:
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ys: list[float] = []
    xs: list[float] = []
    rows_seen: list[int] = []
    for row in range(matrix.shape[0]):
        cols = np.flatnonzero(matrix[row] == ACTION_TO_PLOT_CODE[ACTION_HARVEST_REPLANT])
        if cols.size == 0:
            continue
        ys.append(float(y_centers[row]))
        xs.append(float(x_edges[int(cols[0])]))
        rows_seen.append(row)

    if not ys:
        return None

    row_idx = np.array(rows_seen, dtype=int)
    return {
        "u": np.array(ys, dtype=float),
        "v": np.array(xs, dtype=float),
        "u_lo": np.array([y_edges[int(row_idx.min())]], dtype=float),
        "u_hi": np.array([y_edges[int(row_idx.max()) + 1]], dtype=float),
    }


def _fit_lowess_v_of_u(
    boundary: dict[str, np.ndarray] | None,
    *,
    frac: float,
    n_dense: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if boundary is None:
        return None

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "statsmodels is required for smoothed main phase diagrams. "
            "Install it with `./.venv/bin/pip install -r requirements.txt`."
        ) from exc

    u_vals = boundary["u"]
    v_vals = boundary["v"]
    u_support_lo = float(boundary["u_lo"][0])
    u_support_hi = float(boundary["u_hi"][0])
    _, _, u_mid, v_mid = _compress_constant_segments(u_vals, v_vals)
    u_dense = np.linspace(u_support_lo, u_support_hi, n_dense)

    if len(u_mid) == 1:
        v_dense = np.full_like(u_dense, v_mid[0], dtype=float)
        return u_dense, v_dense

    fitted = lowess(v_mid, u_mid, frac=min(frac, 0.99), it=0, return_sorted=True)
    fit_u = fitted[:, 0]
    fit_v = fitted[:, 1]
    v_dense = np.interp(u_dense, fit_u, fit_v)
    slope_lo = (fit_v[1] - fit_v[0]) / (fit_u[1] - fit_u[0])
    slope_hi = (fit_v[-1] - fit_v[-2]) / (fit_u[-1] - fit_u[-2])
    lo_mask = u_dense < fit_u[0]
    hi_mask = u_dense > fit_u[-1]
    v_dense[lo_mask] = fit_v[0] + slope_lo * (u_dense[lo_mask] - fit_u[0])
    v_dense[hi_mask] = fit_v[-1] + slope_hi * (u_dense[hi_mask] - fit_u[-1])
    return u_dense, v_dense


def _add_clipped_patch(ax: plt.Axes, patch) -> None:
    patch.set_clip_on(True)
    patch.set_clip_path(ax.patch)
    ax.add_patch(patch)


def _bring_axes_to_front(ax: plt.Axes) -> None:
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_zorder(20)
    ax.xaxis.set_zorder(20)
    ax.yaxis.set_zorder(20)
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.tick1line.set_zorder(21)
        tick.tick2line.set_zorder(21)
        tick.label1.set_zorder(21)
        tick.label2.set_zorder(21)
    ax.xaxis.label.set_zorder(21)
    ax.yaxis.label.set_zorder(21)
    ax.title.set_zorder(21)


def _plot_smoothed_main_phase_regions(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    row_slice, visible_y_edges = _visible_row_slice(
        y_edges,
        y_min=MAIN_PHASE_DIAGRAM_Y_MIN,
        y_max=MAIN_PHASE_DIAGRAM_Y_MAX,
    )
    visible_matrix = matrix[row_slice, :]
    x_min = float(x_edges[0])
    x_max = float(x_edges[-1])
    y_min = MAIN_PHASE_DIAGRAM_Y_MIN
    y_max = MAIN_PHASE_DIAGRAM_Y_MAX

    _add_clipped_patch(
        ax,
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=ACTION_COLORS["Hold"],
            edgecolor="none",
            zorder=0.1,
        ),
    )

    switch_meta = _extract_switch_boundary(visible_matrix, x_edges, visible_y_edges)
    switch_boundary = _fit_lowess_v_of_u(
        switch_meta,
        frac=MAIN_PHASE_SWITCH_LOWESS_FRAC,
        n_dense=MAIN_PHASE_SMOOTH_POINTS,
    )
    if switch_boundary is not None:
        switch_x, switch_y = switch_boundary
        switch_x = np.clip(switch_x, x_min, x_max)
        switch_y = np.clip(switch_y, y_min, y_max)
        assert switch_meta is not None
        if switch_meta["orientation"] == "lower":
            switch_points = np.column_stack(
                [
                    np.concatenate(([switch_x[0]], switch_x, [switch_x[-1]])),
                    np.concatenate(([y_min], switch_y, [y_min])),
                ]
            )
        else:
            switch_points = np.column_stack(
                [
                    np.concatenate(([switch_x[0]], switch_x, [switch_x[-1]])),
                    np.concatenate(([y_max], switch_y, [y_max])),
                ]
            )
        _add_clipped_patch(
            ax,
            Polygon(
                switch_points,
                closed=True,
                facecolor=ACTION_COLORS["Switch"],
                edgecolor="none",
                zorder=0.2,
            ),
        )

    harvest_boundary = _fit_lowess_v_of_u(
        _extract_harvest_left_boundary(visible_matrix, x_edges, visible_y_edges),
        frac=MAIN_PHASE_HARVEST_LOWESS_FRAC,
        n_dense=MAIN_PHASE_SMOOTH_POINTS,
    )
    if harvest_boundary is not None:
        harvest_y, harvest_x = harvest_boundary
        harvest_x = np.clip(harvest_x, x_min, x_max)
        harvest_y = np.clip(harvest_y, y_min, y_max)
        harvest_points = np.column_stack(
            [
                np.concatenate((harvest_x, [x_max, x_max])),
                np.concatenate((harvest_y, [harvest_y[-1], harvest_y[0]])),
            ]
        )
        _add_clipped_patch(
            ax,
            Polygon(
                harvest_points,
                closed=True,
                facecolor=ACTION_COLORS["Harvest"],
                edgecolor="none",
                zorder=0.3,
            ),
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
    panel_title_suffixes: Sequence[str] | None = None,
    panel_annotations: Sequence[object | None] | None = None,
    smooth_regions: bool = False,
) -> None:
    pt_grid = price_data["pt_grid"]
    pc_grid = price_data["pc_grid"]
    pc_edges = grid_edges(pc_grid)
    cmap = mcolors.ListedColormap(PLOT_COLORS)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    timber_mask = (pt_grid >= TIMBER_PRICE_MIN) & (pt_grid <= TIMBER_PRICE_MAX)
    if not timber_mask.any():
        timber_mask = np.ones_like(pt_grid, dtype=bool)

    pt_grid_for_plot = pt_grid[timber_mask]
    pt_edges_for_plot = grid_edges(pt_grid_for_plot)
    pt_edges_for_plot[0] = TIMBER_PRICE_MIN
    pt_edges_for_plot[-1] = TIMBER_PRICE_MAX

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    if panel_title_suffixes is not None and len(panel_title_suffixes) != len(ages):
        raise ValueError("panel_title_suffixes must match the number of ages")
    if panel_annotations is not None and len(panel_annotations) != len(ages):
        raise ValueError("panel_annotations must match the number of ages")

    for i, (ax, age, panel_label) in enumerate(zip(axes.flatten(), ages, PANEL_LABELS_2X2)):
        clamped_age = min(age, params.A_max)
        matrix = price_phase_matrix(
            sigma, state_space, params, clamped_age, regime=regime, rotation=rotation
        )
        matrix_for_plot = matrix[:, timber_mask]
        if matrix_for_plot.size == 0:
            matrix_for_plot = matrix
        matrix_for_plot = np.vstack((matrix_for_plot[:1, :], matrix_for_plot))
        pc_edges_for_plot = np.concatenate(([0.0], pc_edges))
        if smooth_regions:
            _plot_smoothed_main_phase_regions(
                ax,
                matrix_for_plot,
                pt_edges_for_plot,
                pc_edges_for_plot,
            )
        else:
            ax.pcolormesh(
                pt_edges_for_plot,
                pc_edges_for_plot,
                matrix_for_plot,
                cmap=cmap,
                norm=norm,
                shading="flat",
                antialiased=False,
                edgecolors="none",
                linewidth=0.0,
                rasterized=True,
            )
            draw_action_boundaries(ax, pt_edges_for_plot, pc_edges_for_plot, matrix_for_plot)
        add_panel_label(ax, panel_label)
        if panel_title_prefix is None:
            add_panel_condition(ax, f"{clamped_age} years")
        else:
            title_suffix = (
                panel_title_suffixes[i]
                if panel_title_suffixes is not None
                else f"{clamped_age} years"
            )
            ax.set_title(f"{panel_title_prefix}: {title_suffix}", loc="left", pad=4)
        if np.all(matrix_for_plot == ACTION_TO_PLOT_CODE[ACTION_DO_NOTHING]):
            ax.text(
                0.5,
                0.5,
                "Always hold",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=TEXT_BLACK,
            )
        annotation = panel_annotations[i] if panel_annotations is not None else None
        if annotation:
            annotation_items = annotation if isinstance(annotation, (list, tuple)) else [annotation]
            for annotation_item in annotation_items:
                if isinstance(annotation_item, str):
                    annotation_kwargs = {
                        "text": annotation_item,
                        "x": 0.5,
                        "y": 0.5,
                        "ha": "center",
                        "va": "center",
                        "fontsize": 7.3,
                    }
                else:
                    annotation_kwargs = {
                        "text": annotation_item["text"],
                        "x": annotation_item.get("x", 0.5),
                        "y": annotation_item.get("y", 0.5),
                        "ha": annotation_item.get("ha", "center"),
                        "va": annotation_item.get("va", "center"),
                        "fontsize": annotation_item.get("fontsize", 7.3),
                    }
                ax.text(
                    annotation_kwargs["x"],
                    annotation_kwargs["y"],
                    annotation_kwargs["text"],
                    transform=ax.transAxes,
                    ha=annotation_kwargs["ha"],
                    va=annotation_kwargs["va"],
                    fontsize=annotation_kwargs["fontsize"],
                    color=TEXT_BLACK,
                )
                if isinstance(annotation_item, dict) and "line_from" in annotation_item and "line_to" in annotation_item:
                    ax.plot(
                        [annotation_item["line_from"][0], annotation_item["line_to"][0]],
                        [annotation_item["line_from"][1], annotation_item["line_to"][1]],
                        transform=ax.transAxes,
                        color=TEXT_BLACK,
                        linewidth=0.6,
                        solid_capstyle="round",
                    )
        ax.set_ylim(0, 150)
        ax.set_yticks(np.arange(0, 151, 25))
        ax.set_xlim(TIMBER_PRICE_MIN, TIMBER_PRICE_MAX)
        ax.set_xticks(np.arange(TIMBER_PRICE_MIN, TIMBER_PRICE_MAX + 1, 20))
        ax.set_xlabel("Timber price ($ per m³)")
        ax.set_ylabel(r"Carbon price (\$ per tCO$_2$)")
        style_axes(ax)
        if smooth_regions:
            _bring_axes_to_front(ax)

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
        output_stem=os.path.join(paths["figures"], "switching_frontier_by_age"),
    )

    rotation_difference_ages = find_rotation_difference_ages(
        sigma=sigma,
        state_space=state_space,
        params=params,
    )
    plot_phase_diagram_panels(
        sigma=sigma,
        state_space=state_space,
        params=params,
        price_data=price_data,
        ages=LATER_ROTATION_PHASE_DIAGRAM_AGES,
        regime=0,
        rotation=2,
        output_stem=os.path.join(paths["figures"], "phase_diagram_later_rotations"),
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
        ages=sorted(set(MAIN_PHASE_DIAGRAM_AGES).union(LATER_ROTATION_PHASE_DIAGRAM_AGES)),
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
