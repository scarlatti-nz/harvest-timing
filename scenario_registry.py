"""
Shared scenario registry for model runs and utility comparison bundles.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import Any, Mapping

from harvest_timing_model import ModelParameters
from paper_style import REGIME_COLORS


MODEL_SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "overrides": {
            "switch_cost": 1e9,
            "harvest_penalty_per_m3": 10000.0,
        },
    },
    "low-volatility": {
        "overrides": {
            "switch_cost": 1e9,
            "harvest_penalty_per_m3": 10000.0,
            "pc_rho": 0.8,
            "pc_sigma": 0.05,
        },
    },
    "low-expectations": {
        "overrides": {
            "switch_cost": 1e9,
            "harvest_penalty_per_m3": 10000.0,
            "pc_0": 50.0,
            "pc_mean": 10.0,
            "pc_rho": 0.96,
            "pc_sigma": 0.2,
        },
    },
    "high-expectations": {
        "overrides": {
            "switch_cost": 1e9,
            "harvest_penalty_per_m3": 10000.0,
            "pc_0": 50.0,
            "pc_mean": 300.0,
            "pc_rho": 0.96,
            "pc_sigma": 0.2,
        },
    },
    "stock-change-bank-credit": {
        "overrides": {
            "switch_cost": 1e9,
            "harvest_penalty_per_m3": 10000.0,
            "carbon_credit_max_age": 10,
        },
    },
    "switching-policy": {
        "overrides": {},
    },
}

UTILITY_SCENARIO_SETS: dict[str, list[dict[str, Any]]] = {
    "baseline": [
        {
            "name": "Averaging",
            "model_scenario": "baseline",
            "default_run_name": "baseline",
            "regime": 0,
            "rotation": 1,
            "color": REGIME_COLORS["averaging"],
        },
        {
            "name": "Permanent",
            "model_scenario": "baseline",
            "default_run_name": "baseline",
            "regime": 1,
            "rotation": 1,
            "color": REGIME_COLORS["permanent"],
        },
        {
            "name": "Stock change",
            "model_scenario": "baseline",
            "default_run_name": "baseline",
            "regime": 2,
            "rotation": 1,
            "color": REGIME_COLORS["stock_change"],
        },
    ],
    "suboptimal": [
        {
            "name": "Averaging",
            "model_scenario": "baseline",
            "default_run_name": "baseline",
            "regime": 0,
            "rotation": 1,
            "color": REGIME_COLORS["averaging"],
        },
        {
            "name": "Stock change (force harvest at age 28)",
            "model_scenario": "baseline",
            "default_run_name": "baseline",
            "regime": 2,
            "rotation": 1,
            "color": REGIME_COLORS["stock_change"],
            "force_age": 28,
        },
        {
            "name": "Stock change (bank credits)",
            "model_scenario": "stock-change-bank-credit",
            "default_run_name": "stock-change-bank-credit",
            "regime": 0,
            "rotation": 1,
            "color": "#7E6148",
        },
    ],
    "paper": [
        {
            "name": "Averaging",
            "model_scenario": "switching-policy",
            "default_run_name": "switching_policy",
            "regime": 0,
            "rotation": 1,
            "color": REGIME_COLORS["averaging"],
        },
        {
            "name": "Permanent",
            "model_scenario": "switching-policy",
            "default_run_name": "switching_policy",
            "regime": 1,
            "rotation": 1,
            "color": REGIME_COLORS["permanent"],
        },
        {
            "name": "Stock change",
            "model_scenario": "switching-policy",
            "default_run_name": "switching_policy",
            "regime": 2,
            "rotation": 1,
            "color": REGIME_COLORS["stock_change"],
        },
    ],
}

_SCENARIO_MATCH_IGNORED_FIELDS = frozenset({"N_pc", "N_pt"})


def _extract_parameter_values(params: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in fields(ModelParameters):
        try:
            values[field.name] = getattr(params, field.name)
        except AttributeError as exc:
            raise ValueError(
                f"Results params are missing required field '{field.name}'."
            ) from exc
    return values


def list_matching_model_scenarios(params: Any) -> tuple[str, ...]:
    actual = _extract_parameter_values(params)
    default_values = _extract_parameter_values(ModelParameters())
    matches: list[str] = []

    for name, scenario in MODEL_SCENARIOS.items():
        expected = dict(default_values)
        expected.update(scenario["overrides"])
        if all(
            actual[field_name] == expected[field_name]
            for field_name in actual
            if field_name not in _SCENARIO_MATCH_IGNORED_FIELDS
        ):
            matches.append(name)

    return tuple(matches)


def resolve_results_model_scenario(results: Mapping[str, Any]) -> str:
    params = results.get("params")
    if params is None:
        raise ValueError("Results payload is missing params.")

    matches = list_matching_model_scenarios(params)
    metadata = results.get("metadata") or {}
    explicit_model_scenario = metadata.get("model_scenario")
    if explicit_model_scenario is not None:
        explicit_model_scenario = str(explicit_model_scenario)
        if not has_model_scenario(explicit_model_scenario):
            raise ValueError(
                f"Results metadata declares unknown model_scenario "
                f"'{explicit_model_scenario}'."
            )
        if explicit_model_scenario not in matches:
            if matches:
                raise ValueError(
                    f"Results metadata declares model_scenario "
                    f"'{explicit_model_scenario}', but parameters match "
                    f"{', '.join(matches)}."
                )
            raise ValueError(
                f"Results metadata declares model_scenario "
                f"'{explicit_model_scenario}', but parameters do not match any "
                "known model scenario."
            )
        return explicit_model_scenario

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not infer model scenario from results parameters."
        )

    raise ValueError(
        "Could not infer a unique model scenario from results parameters. "
        f"Matches: {', '.join(matches)}."
    )


def list_model_scenarios() -> tuple[str, ...]:
    return tuple(MODEL_SCENARIOS)


def list_utility_scenario_sets() -> tuple[str, ...]:
    return tuple(UTILITY_SCENARIO_SETS)


def has_model_scenario(name: str) -> bool:
    return name in MODEL_SCENARIOS


def has_utility_scenario_set(name: str) -> bool:
    return name in UTILITY_SCENARIO_SETS


def get_model_scenario(name: str) -> dict[str, Any]:
    try:
        return deepcopy(MODEL_SCENARIOS[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown model scenario '{name}'. "
            f"Known scenarios: {', '.join(list_model_scenarios())}"
        ) from exc


def get_utility_scenarios(name: str) -> list[dict[str, Any]]:
    try:
        return deepcopy(UTILITY_SCENARIO_SETS[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown utility scenario set '{name}'. "
            f"Known sets: {', '.join(list_utility_scenario_sets())}"
        ) from exc
