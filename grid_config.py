"""
Shared helpers for price-grid defaults, output paths, and results validation.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, Mapping, Optional


DEFAULT_PRICE_GRID_SIZE = 11
RESULTS_SCHEMA_VERSION = 1


def normalize_grid_size(grid_size: int) -> int:
    size = int(grid_size)
    if size < 1:
        raise ValueError(f"Grid size must be positive, got {grid_size}.")
    return size


def grid_tag(grid_size: int) -> str:
    size = normalize_grid_size(grid_size)
    return f"grid_{size}x{size}"


def require_square_price_grid(params: Any) -> int:
    n_pc = getattr(params, "N_pc", None)
    n_pt = getattr(params, "N_pt", None)
    if n_pc is None or n_pt is None:
        raise ValueError("Results params are missing N_pc/N_pt.")

    n_pc = int(n_pc)
    n_pt = int(n_pt)
    if n_pc != n_pt:
        raise ValueError(
            f"Expected aligned timber/carbon price grids, got N_pc={n_pc}, N_pt={n_pt}."
        )
    return n_pc


def output_root(run_name: str, grid_size: int, outputs_root: str = "outputs") -> str:
    return os.path.join(outputs_root, run_name, grid_tag(grid_size))


def switching_policy_output_dir(
    grid_size: int,
    outputs_root: str = "outputs",
) -> str:
    return output_root("switching_policy", grid_size, outputs_root)


def model_results_path(
    run_name: str,
    grid_size: int,
    outputs_root: str = "outputs",
) -> str:
    return os.path.join(output_root(run_name, grid_size, outputs_root), "model_results.pkl")


def model_parameters_path(
    run_name: str,
    grid_size: int,
    timestamp: str,
    outputs_root: str = "outputs",
) -> str:
    return os.path.join(
        output_root(run_name, grid_size, outputs_root),
        f"model_parameters_{timestamp}.txt",
    )


def scenario_cache_path(cache_root: str, scenario_name: str) -> str:
    return os.path.join(cache_root, f"{scenario_name}_realized_npvs.csv")


def results_plot_output_dir(
    run_name: str,
    grid_size: int,
    plots_root: str = "plots",
) -> str:
    return os.path.join(plots_root, "results", run_name, grid_tag(grid_size))


def utility_plot_output_dir(
    scenario_set: str,
    grid_size: int,
    plots_root: str = "plots",
) -> str:
    return os.path.join(plots_root, "utility", scenario_set, grid_tag(grid_size))


def build_results_metadata(
    params: Any,
    run_name: Optional[str] = None,
    model_scenario: Optional[str] = None,
) -> dict[str, Any]:
    size = require_square_price_grid(params)
    metadata: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "grid_size": size,
        "grid_tag": grid_tag(size),
    }
    if run_name is not None:
        metadata["run_name"] = run_name
    if model_scenario is not None:
        metadata["model_scenario"] = model_scenario
    return metadata


def get_results_grid_size(results: Mapping[str, Any]) -> int:
    params = results.get("params")
    if params is None:
        raise ValueError("Results payload is missing params.")

    size = require_square_price_grid(params)
    metadata = results.get("metadata") or {}

    meta_size = metadata.get("grid_size")
    if meta_size is not None and int(meta_size) != size:
        raise ValueError(
            f"Grid size mismatch between params ({size}) and metadata ({meta_size})."
        )

    meta_tag = metadata.get("grid_tag")
    if meta_tag is not None and meta_tag != grid_tag(size):
        raise ValueError(
            f"Grid tag mismatch for grid size {size}: expected {grid_tag(size)}, got {meta_tag}."
        )

    return size


def get_results_run_name(
    results: Mapping[str, Any],
    default: Optional[str] = None,
) -> Optional[str]:
    metadata = results.get("metadata") or {}
    run_name = metadata.get("run_name")
    if run_name:
        return str(run_name)
    return default


def infer_run_name_from_pickle_path(pickle_path: str) -> str:
    abs_pickle_path = os.path.abspath(pickle_path)
    parent_dir = os.path.basename(os.path.dirname(abs_pickle_path))
    if re.fullmatch(r"grid_\d+x\d+", parent_dir):
        grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(abs_pickle_path)))
        if grandparent_dir:
            return grandparent_dir
    return parent_dir


def load_results_pickle(
    pickle_path: str,
    expected_grid_size: Optional[int] = None,
) -> dict[str, Any]:
    with open(pickle_path, "rb") as handle:
        results = pickle.load(handle)

    actual_grid_size = get_results_grid_size(results)
    if expected_grid_size is not None:
        expected_grid_size = normalize_grid_size(expected_grid_size)
        if actual_grid_size != expected_grid_size:
            raise ValueError(
                f"Expected grid size {expected_grid_size}, found {actual_grid_size} in {pickle_path}."
            )

    return results
