"""
Shared scenario registry for model runs and utility comparison bundles.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

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
}

UTILITY_SCENARIO_SETS: dict[str, list[dict[str, Any]]] = {
    "baseline": [
        {
            "name": "Averaging",
            "run_name": "baseline",
            "regime": 0,
            "rotation": 1,
            "color": REGIME_COLORS["averaging"],
        },
        {
            "name": "Permanent",
            "run_name": "baseline",
            "regime": 1,
            "rotation": 1,
            "color": REGIME_COLORS["permanent"],
        },
        {
            "name": "Stock change",
            "run_name": "baseline",
            "regime": 2,
            "rotation": 1,
            "color": REGIME_COLORS["stock_change"],
        },
    ],
    "suboptimal": [
        {
            "name": "Averaging",
            "run_name": "baseline",
            "regime": 0,
            "rotation": 1,
            "color": REGIME_COLORS["averaging"],
        },
        {
            "name": "Stock change (force harvest at age 28)",
            "run_name": "baseline",
            "regime": 2,
            "rotation": 1,
            "color": REGIME_COLORS["stock_change"],
            "force_age": 28,
        },
        {
            "name": "Stock change (bank credits)",
            "run_name": "stock-change-bank-credit",
            "regime": 0,
            "rotation": 1,
            "color": "#7E6148",
        },
    ],
}


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
