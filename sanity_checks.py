"""
Sanity Checks for Forest Harvest Timing Model

This script runs detailed sanity checks on the harvest timing model
to verify correctness of the implementation.
"""

import numpy as np
from typing import Optional, Dict

# Import all necessary components from the main model
from harvest_timing_model import (
    ModelParameters,
    StateSpace,
    compute_carbon_curve,
    compute_volume_from_carbon,
    compute_carbon_flows_averaging,
    compute_carbon_flows_permanent,
    compute_price_quality_factor,
    build_price_grids,
    build_state_space,
    build_reward_matrix,
    build_transition_matrix,
    solve_model,
    ACTION_DO_NOTHING,
    ACTION_HARVEST_REPLANT,
    ACTION_SWITCH_PERMANENT,
)


def _print_scenario_details(
    name: str,
    params: ModelParameters,
    price_data: Dict,
    C_age: np.ndarray,
    V_age: np.ndarray,
    DeltaC: np.ndarray,
    R: np.ndarray,
    V: np.ndarray,
    sigma: np.ndarray,
    state_space: StateSpace,
    regime: int = 0,
    rotation: int = 1
):
    """Print detailed information for a sanity check scenario."""
    pc = price_data['pc_grid'][0]
    pt = price_data['pt_grid'][0]
    
    print(f"\n  Parameters:")
    print(f"    Carbon price: ${pc:.2f}/tCO₂")
    print(f"    Timber price: ${pt:.2f}/m³")
    print(f"    Harvest cost: ${params.harvest_cost_per_m3:.2f}/m³")
    print(f"    Flat harvest cost: ${params.harvest_cost_flat_per_ha:.2f}/ha")
    print(f"    Harvest penalty: ${params.harvest_penalty_per_m3:.2f}/m³")
    print(f"    Replant cost: ${params.replant_cost:.2f}")
    print(f"    Maintenance: ${params.maintenance_cost:.2f}/yr")
    print(f"    Discount rate: {params.discount_rate:.1%}")
    print(f"    Carbon credits max age: {params.carbon_credit_max_age}")
    
    regime_name = "averaging" if regime == 0 else "permanent"
    rot_name = "first" if rotation == 1 else "later"
    
    print(f"\n  State trajectory starting from (regime={regime_name}, rotation={rot_name}):")
    print(f"    V(s) = Value function = expected PV of all future cashflows under optimal policy")
    print(f"    Note: R values shown are for current state; after Switch, next state is in permanent regime")
    print(f"    {'Age':>4} {'Carbon':>10} {'Volume':>10} {'ΔCarbon':>10} {'R(hold)':>10} {'R(harv)':>10} {'R(switch)':>10} {'V(s)':>12} {'Action':>8}")
    print(f"    {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    
    # Find optimal harvest age and first switch age
    optimal_age = None
    first_switch_age = None
    for a in range(min(params.N_a, 50)):
        s = state_space.tuple_to_state[(a, 0, 0, regime, rotation)]
        action = sigma[s]
        action_name = ['Hold', 'Harvest', 'Switch'][action]
        
        # Mark first switch or harvest
        marker = ""
        if action == ACTION_SWITCH_PERMANENT and first_switch_age is None and regime == 0:
            first_switch_age = a
            marker = " ← SWITCH"
        elif action == ACTION_HARVEST_REPLANT and optimal_age is None:
            optimal_age = a
            marker = " ← HARVEST"
        
        print(f"    {a:4d} {C_age[a]:10.1f} {V_age[a]:10.1f} {DeltaC[a]:10.2f} "
              f"{R[s, ACTION_DO_NOTHING]:10.2f} {R[s, ACTION_HARVEST_REPLANT]:10.2f} "
              f"{R[s, ACTION_SWITCH_PERMANENT]:10.2f} {V[s]:12.2f} {action_name:>8}{marker}")
    
    # Summary
    if first_switch_age is not None and regime == 0:
        print(f"\n  → Optimal first action: SWITCH to permanent at age {first_switch_age}")
        print(f"    After switching, you enter permanent regime where:")
        print(f"    - Carbon credits continue indefinitely (not capped at age {params.carbon_credit_max_age})")
        print(f"    - Harvesting incurs carbon liability = {params.carbon_liability_npv_factor():.1%} × carbon_price × carbon_stock")
        
        # Show what permanent regime looks like after switch
        print(f"\n  What happens AFTER switching (permanent regime trajectory):")
        print(f"    {'Age':>4} {'ΔC(perm)':>10} {'R(hold)':>12} {'R(harvest)':>12} {'Action':>8}")
        print(f"    {'-'*4} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
        
        for a in range(first_switch_age + 1, min(first_switch_age + 15, params.N_a)):
            s_perm = state_space.tuple_to_state[(a, 0, 0, 1, rotation)]  # regime=1 (permanent)
            action_perm = sigma[s_perm]
            action_name_perm = ['Hold', 'Harvest', 'Switch'][action_perm]
            # Get permanent regime delta C
            delta_c_perm = C_age[a] - C_age[a-1] if a > 0 else 0
            print(f"    {a:4d} {delta_c_perm:10.2f} {R[s_perm, ACTION_DO_NOTHING]:12.2f} "
                  f"{R[s_perm, ACTION_HARVEST_REPLANT]:12.2f} {action_name_perm:>8}")
    
    elif optimal_age is not None:
        print(f"\n  → Optimal harvest age: {optimal_age} years")
        
        # Show harvest payoff breakdown at optimal age
        a = optimal_age
        volume = V_age[a]
        carbon = C_age[a]
        harvest_cost = (params.harvest_cost_per_m3 + params.harvest_penalty_per_m3) * volume
        timber_revenue = pt * volume
        net_timber = timber_revenue - harvest_cost - params.harvest_cost_flat_per_ha - params.replant_cost
        
        print(f"\n  Harvest payoff breakdown at age {a}:")
        print(f"    Timber revenue: ${timber_revenue:,.2f} ({volume:.1f} m³ × ${pt:.2f})")
        print(f"    Harvest cost (per m³):   -${harvest_cost:,.2f} ({volume:.1f} m³ × ${params.harvest_cost_per_m3 + params.harvest_penalty_per_m3:.2f})")
        print(f"    Flat harvest cost:       -${params.harvest_cost_flat_per_ha:,.2f}/ha")
        print(f"    Replant cost:   -${params.replant_cost:,.2f}")
        print(f"    Net timber:     ${net_timber:,.2f}")
        
        if regime == 1:  # Permanent
            liability_factor = params.carbon_liability_npv_factor()
            liability = pc * carbon * liability_factor
            print(f"    Carbon liability: -${liability:,.2f} ({carbon:.1f} tCO₂ × ${pc:.2f} × {liability_factor:.3f})")
            print(f"    Total harvest reward: ${net_timber - liability:,.2f}")
        else:
            print(f"    Total harvest reward: ${net_timber:,.2f}")
    else:
        print(f"\n  → No harvest optimal up to age 50")
    
    return optimal_age


def run_sanity_checks(base_params: Optional[ModelParameters] = None):
    """
    Run sanity checks with detailed logging.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    
    # ==========================================================================
    # Check 1: Baseline with deterministic prices
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CHECK 1: Baseline - Averaging regime, first rotation")
    print("-" * 70)
    
    params_det = ModelParameters(N_pc=1, N_pt=1, A_max=60)
    
    C_age = compute_carbon_curve(params_det)
    V_age = compute_volume_from_carbon(C_age, params_det)
    DeltaC_avg = compute_carbon_flows_averaging(C_age, params_det)
    DeltaC_perm = compute_carbon_flows_permanent(C_age)
    price_quality_factor = compute_price_quality_factor(params_det)
    price_data = build_price_grids(params_det)
    state_space = build_state_space(params_det)
    
    print(f"\n  State space size: {state_space.N_states}")
    
    R = build_reward_matrix(params_det, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
    Q = build_transition_matrix(params_det, state_space, price_data)
    V, sigma = solve_model(R, Q, params_det.beta)
    
    _print_scenario_details(
        "Baseline", params_det, price_data, C_age, V_age, DeltaC_avg,
        R, V, sigma, state_space, regime=0, rotation=1
    )
    
    # Also show later rotation with switch penalty details
    print("\n  --- Later Rotation (2nd+) with Switch Penalty ---")
    print(f"    When switching from 2nd+ rotation averaging to permanent:")
    print(f"    - If age < {params_det.carbon_credit_max_age}: must pay back shortfall = carbon_price × (C@16 - current_C)")
    print(f"    - If age >= {params_det.carbon_credit_max_age}: no penalty, just start accruing")
    print(f"    Carbon at age {params_det.carbon_credit_max_age}: {C_age[params_det.carbon_credit_max_age]:.1f} tCO₂")
    print()
    print(f"    {'Age':>4} {'Carbon':>10} {'Shortfall':>10} {'R(switch)':>12} {'Action':>8}")
    print(f"    {'-'*4} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    
    carbon_at_16 = C_age[params_det.carbon_credit_max_age]
    opt_age_rot2 = None
    for a in range(min(30, params_det.N_a)):
        s = state_space.tuple_to_state[(a, 0, 0, 0, 2)]  # regime=0, rotation=2
        action = sigma[s]
        action_name = ['Hold', 'Harvest', 'Switch'][action]
        # Shortfall = how much less carbon you have than age-16 level
        shortfall = max(0, carbon_at_16 - C_age[a])
        
        if action == ACTION_HARVEST_REPLANT and opt_age_rot2 is None:
            opt_age_rot2 = a
        
        print(f"    {a:4d} {C_age[a]:10.1f} {shortfall:10.1f} {R[s, ACTION_SWITCH_PERMANENT]:12.2f} {action_name:>8}")
    
    print(f"\n  → Optimal harvest age (averaging, later rotations): {opt_age_rot2} years")
    
    # ==========================================================================
    # Check 2: No carbon credits
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CHECK 2: No carbon credits (baseline comparison)")
    print("-" * 70)
    
    DeltaC_zero = np.zeros_like(DeltaC_avg)
    R_no_carbon = build_reward_matrix(params_det, state_space, price_data, V_age, C_age, DeltaC_zero, DeltaC_zero, price_quality_factor)
    V_nc, sigma_nc = solve_model(R_no_carbon, Q, params_det.beta)
    
    _print_scenario_details(
        "No Carbon", params_det, price_data, C_age, V_age, DeltaC_zero,
        R_no_carbon, V_nc, sigma_nc, state_space, regime=0, rotation=1
    )
    
    # ==========================================================================
    # Check 3: With harvest penalty
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CHECK 3: With $10/m³ harvest penalty")
    print("-" * 70)
    
    params_penalty = ModelParameters(N_pc=1, N_pt=1, A_max=60, harvest_penalty_per_m3=10.0)
    
    C_age_p = compute_carbon_curve(params_penalty)
    V_age_p = compute_volume_from_carbon(C_age_p, params_penalty)
    DeltaC_avg_p = compute_carbon_flows_averaging(C_age_p, params_penalty)
    DeltaC_perm_p = compute_carbon_flows_permanent(C_age_p)
    price_quality_factor_p = compute_price_quality_factor(params_penalty)
    price_data_p = build_price_grids(params_penalty)
    state_space_p = build_state_space(params_penalty)
    
    R_p = build_reward_matrix(params_penalty, state_space_p, price_data_p, V_age_p, C_age_p, DeltaC_avg_p, DeltaC_perm_p, price_quality_factor_p)
    Q_p = build_transition_matrix(params_penalty, state_space_p, price_data_p)
    V_p, sigma_p = solve_model(R_p, Q_p, params_penalty.beta)
    
    _print_scenario_details(
        "With Penalty", params_penalty, price_data_p, C_age_p, V_age_p, DeltaC_avg_p,
        R_p, V_p, sigma_p, state_space_p, regime=0, rotation=1
    )
    
    # ==========================================================================
    # Check 4: Permanent regime (carbon credits continue, but liability on harvest)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CHECK 4: Permanent regime (carbon indefinitely, liability on harvest)")
    print("-" * 70)
    
    print(f"\n  Carbon liability NPV factor: {params_det.carbon_liability_npv_factor():.4f}")
    print(f"    (50% instant + 50% over {params_det.carbon_liability_years} years)")
    print(f"  Carbon credits: continue indefinitely (unlike averaging which stops at age {params_det.carbon_credit_max_age})")
    
    _print_scenario_details(
        "Permanent", params_det, price_data, C_age, V_age, DeltaC_perm,
        R, V, sigma, state_space, regime=1, rotation=1
    )
    
    print("\n" + "=" * 70)
    print("✓ SANITY CHECKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("REAL OPTIONS MODEL - SANITY CHECKS ONLY")
    print("=" * 70)
    run_sanity_checks()

