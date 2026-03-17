"""
Helpers for loading and validating the config-driven run_all workflow.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from grid_config import normalize_grid_size
from scenario_registry import has_model_scenario, has_utility_scenario_set


DEFAULT_RUN_ALL_CONFIG_PATH = os.path.join("configs", "run_all.json")


@dataclass(frozen=True)
class ModelRunJob:
    scenario: str
    run_name: str
    grid_size: int


@dataclass(frozen=True)
class UtilityJob:
    scenario_set: str
    grid_size: int
    output_dir: str | None = None


@dataclass(frozen=True)
class PlotJob:
    kind: str
    run_name: str
    grid_size: int
    output_dir: str | None = None


@dataclass(frozen=True)
class RunAllConfig:
    model_runs: tuple[ModelRunJob, ...]
    utility_jobs: tuple[UtilityJob, ...]
    plot_jobs: tuple[PlotJob, ...]


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list.")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, context)


def _require_grid_size(value: Any, context: str) -> int:
    try:
        return normalize_grid_size(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a positive integer.") from exc


def _parse_model_run(job_data: Any, index: int) -> ModelRunJob:
    job = _require_mapping(job_data, f"model_runs[{index}]")
    scenario = _require_string(job.get("scenario"), f"model_runs[{index}].scenario")
    if not has_model_scenario(scenario):
        raise ValueError(f"model_runs[{index}] references unknown scenario '{scenario}'.")

    return ModelRunJob(
        scenario=scenario,
        run_name=_require_string(job.get("run_name"), f"model_runs[{index}].run_name"),
        grid_size=_require_grid_size(job.get("grid_size"), f"model_runs[{index}].grid_size"),
    )


def _parse_utility_job(job_data: Any, index: int) -> UtilityJob:
    job = _require_mapping(job_data, f"utility_jobs[{index}]")
    scenario_set = _require_string(job.get("scenario_set"), f"utility_jobs[{index}].scenario_set")
    if not has_utility_scenario_set(scenario_set):
        raise ValueError(
            f"utility_jobs[{index}] references unknown scenario_set '{scenario_set}'."
        )

    return UtilityJob(
        scenario_set=scenario_set,
        grid_size=_require_grid_size(job.get("grid_size"), f"utility_jobs[{index}].grid_size"),
        output_dir=_optional_string(job.get("output_dir"), f"utility_jobs[{index}].output_dir"),
    )


def _parse_plot_job(job_data: Any, index: int) -> PlotJob:
    job = _require_mapping(job_data, f"plot_jobs[{index}]")
    kind = _require_string(job.get("kind"), f"plot_jobs[{index}].kind")
    if kind != "results":
        raise ValueError(
            f"plot_jobs[{index}] has unsupported kind '{kind}'. Only 'results' is supported."
        )

    return PlotJob(
        kind=kind,
        run_name=_require_string(job.get("run_name"), f"plot_jobs[{index}].run_name"),
        grid_size=_require_grid_size(job.get("grid_size"), f"plot_jobs[{index}].grid_size"),
        output_dir=_optional_string(job.get("output_dir"), f"plot_jobs[{index}].output_dir"),
    )


def _validate_uniqueness(config: RunAllConfig) -> None:
    seen_model_runs: set[tuple[str, int]] = set()
    for job in config.model_runs:
        key = (job.run_name, job.grid_size)
        if key in seen_model_runs:
            raise ValueError(
                f"Duplicate model run for run_name='{job.run_name}' at {job.grid_size}x{job.grid_size}."
            )
        seen_model_runs.add(key)

    seen_utility_jobs: set[tuple[str, int, str | None]] = set()
    for job in config.utility_jobs:
        key = (job.scenario_set, job.grid_size, job.output_dir)
        if key in seen_utility_jobs:
            raise ValueError(
                f"Duplicate utility job for scenario_set='{job.scenario_set}' at "
                f"{job.grid_size}x{job.grid_size}."
            )
        seen_utility_jobs.add(key)

    seen_plot_jobs: set[tuple[str, str, int, str | None]] = set()
    for job in config.plot_jobs:
        key = (job.kind, job.run_name, job.grid_size, job.output_dir)
        if key in seen_plot_jobs:
            raise ValueError(
                f"Duplicate plot job for kind='{job.kind}', run_name='{job.run_name}' "
                f"at {job.grid_size}x{job.grid_size}."
            )
        seen_plot_jobs.add(key)


def load_run_all_config(config_path: str = DEFAULT_RUN_ALL_CONFIG_PATH) -> RunAllConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    data = _require_mapping(raw, "run_all config")
    model_runs = tuple(
        _parse_model_run(job, index)
        for index, job in enumerate(_require_list(data.get("model_runs"), "model_runs"))
    )
    utility_jobs = tuple(
        _parse_utility_job(job, index)
        for index, job in enumerate(_require_list(data.get("utility_jobs"), "utility_jobs"))
    )
    plot_jobs = tuple(
        _parse_plot_job(job, index)
        for index, job in enumerate(_require_list(data.get("plot_jobs"), "plot_jobs"))
    )

    config = RunAllConfig(
        model_runs=model_runs,
        utility_jobs=utility_jobs,
        plot_jobs=plot_jobs,
    )
    _validate_uniqueness(config)
    return config


def describe_run_all_config(config: RunAllConfig) -> str:
    lines = ["Execution plan:"]

    if config.model_runs:
        lines.append("  Model runs:")
        for job in config.model_runs:
            lines.append(
                f"    - {job.scenario} -> {job.run_name} @ {job.grid_size}x{job.grid_size}"
            )

    if config.utility_jobs:
        lines.append("  Utility jobs:")
        for job in config.utility_jobs:
            output_suffix = f" -> {job.output_dir}" if job.output_dir else ""
            lines.append(
                f"    - {job.scenario_set} @ {job.grid_size}x{job.grid_size}{output_suffix}"
            )

    if config.plot_jobs:
        lines.append("  Plot jobs:")
        for job in config.plot_jobs:
            output_suffix = f" -> {job.output_dir}" if job.output_dir else ""
            lines.append(
                f"    - {job.kind}:{job.run_name} @ {job.grid_size}x{job.grid_size}{output_suffix}"
            )

    return "\n".join(lines)
