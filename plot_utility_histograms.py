"""
Script to generate histograms of simulated utilities (NPV) at time 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from paper_style import (
    REGIME_COLORS,
    REFERENCE_COLOR,
    configure_paper_style,
    save_figure,
    style_axes,
)
from harvest_timing_model import (
    ModelParameters, StateSpace, simulate_single_trajectory,
    compute_carbon_curve, compute_volume_from_carbon,
    compute_carbon_flows_averaging, compute_carbon_flows_permanent,
    compute_price_quality_factor, build_reward_matrix, build_transition_representation,
    estimate_transition_nnz,
    ACTION_DO_NOTHING, ACTION_HARVEST_REPLANT, ACTION_SWITCH_PERMANENT, N_ACTIONS
)
from grid_config import (
    DEFAULT_PRICE_GRID_SIZE,
    load_results_pickle,
    model_results_path,
    plot_output_dir,
    scenario_cache_path,
)

SCENARIOS_BASELINE = [
    {
        'name': 'Averaging',
        'dir': 'baseline',
        'regime': 0,
        'rotation': 1,
        'color': REGIME_COLORS['averaging'],
    },
    {
        'name': 'Permanent',
        'dir': 'baseline',
        'regime': 1,
        'rotation': 1,
        'color': REGIME_COLORS['permanent'],
    },
    {
        'name': 'Stock change',
        'dir': 'baseline',
        'regime': 2,
        'rotation': 1,
        'color': REGIME_COLORS['stock_change'],
    },
]

SCENARIOS_SUBOPTIMAL = [
    {
        'name': 'Averaging',
        'dir': 'baseline',
        'regime': 0,
        'rotation': 1,
        'color': REGIME_COLORS['averaging'],
    },
    {
        'name': 'Stock change (force harvest at age 28)',
        'dir': 'baseline',
        'regime': 2,
        'rotation': 1,
        'color': REGIME_COLORS['stock_change'],
        'force_age': 28,
    },
    {
        'name': 'Stock change (bank credits)',
        'dir': 'stock-change-bank-credit',
        'regime': 0,
        'rotation': 1,
        'color': '#7E6148',
    }
]

REQUIRED_CACHE_COLUMNS = {'carbon', 'timber', 'total', 'avg_pc', 'avg_pt', 'harvest_age'}


def build_reward_components(
    params: ModelParameters,
    state_space: StateSpace,
    price_data,
    V_age,
    C_age,
    DeltaC_avg,
    DeltaC_perm,
    price_quality_factor
):
    """
    Build carbon and timber components matching build_reward_matrix logic.
    Returns two arrays of shape (N_states, N_ACTIONS).
    """
    carbon_component = np.zeros((state_space.N_states, N_ACTIONS), dtype=np.float32)
    timber_component = np.zeros_like(carbon_component)

    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    liability_npv_factor = params.carbon_liability_npv_factor()

    for s in range(state_space.N_states):
        a, i_pc, i_pt, regime, rotation = state_space.state_to_tuple[s]

        pc = pc_grid[i_pc]
        pt = pt_grid[i_pt]
        volume = V_age[a]
        carbon_stock = C_age[a]

        # Carbon flow depends on regime/rotation
        if regime == 0:  # Averaging
            delta_C = DeltaC_avg[a] if rotation == 1 else 0.0
        else:  # Permanent (regime=1) or Permanent No Penalty (regime=2)
            delta_C = DeltaC_perm[a]

        # Shared pieces
        R_carbon = pc * delta_C
        R_cost = -params.maintenance_cost
        effective_pt = pt * price_quality_factor[a]

        # Action 0: Do nothing
        carbon_component[s, ACTION_DO_NOTHING] = R_carbon
        timber_component[s, ACTION_DO_NOTHING] = R_cost

        # Action 1: Harvest and replant
        # Harvest penalty only applies to permanent (regime=1), not to
        # averaging (regime=0) or pre-2023 stock change (regime=2)
        harvest_cost_per_m3 = params.harvest_cost_per_m3
        if regime == 1:  # Permanent incurs penalty
            harvest_cost_per_m3 += params.harvest_penalty_per_m3

        R_timber = effective_pt * volume - harvest_cost_per_m3 * volume
        R_replant = -params.replant_cost
        R_harvest_flat = -params.harvest_cost_flat_per_ha

        # Carbon liability applies to both permanent (regime=1) and pre-2023 stock-change (regime=2)
        carbon_liability = 0.0
        if regime >= 1:
            carbon_liability = pc * carbon_stock * liability_npv_factor

        carbon_component[s, ACTION_HARVEST_REPLANT] = -carbon_liability
        timber_component[s, ACTION_HARVEST_REPLANT] = R_timber + R_replant + R_harvest_flat

        # Action 2: Switch to permanent (only from averaging)
        # Cannot switch into pre-2023 stock change from any regime
        if regime == 0:
            switch_penalty = params.switch_cost
            carbon_shortfall_penalty = 0.0
            if rotation >= 2 and a < params.carbon_credit_max_age:
                carbon_at_16 = C_age[params.carbon_credit_max_age]
                carbon_shortfall = max(0.0, carbon_at_16 - carbon_stock)
                carbon_shortfall_penalty = pc * carbon_shortfall
                switch_penalty += carbon_shortfall_penalty

            carbon_component[s, ACTION_SWITCH_PERMANENT] = R_carbon - carbon_shortfall_penalty
            timber_component[s, ACTION_SWITCH_PERMANENT] = R_cost - params.switch_cost
        else:
            # Permanent or pre-2023 stock change regime: identical to do-nothing
            carbon_component[s, ACTION_SWITCH_PERMANENT] = carbon_component[s, ACTION_DO_NOTHING]
            timber_component[s, ACTION_SWITCH_PERMANENT] = timber_component[s, ACTION_DO_NOTHING]

    return carbon_component, timber_component

def run_simulations(
    n_sims: int,
    regime: int,
    rotation: int,
    params,
    state_space,
    price_data,
    R,
    Q,
    V,
    sigma,
    C_age,
    reward_components,
    n_years: int = 200
):
    """Run N simulations and return list of NPVs, harvest ages and average prices."""
    npvs = []
    carbon_npvs = []
    timber_npvs = []
    harvest_ages = []
    avg_carbon_prices = []
    avg_timber_prices = []

    carbon_component, timber_component = reward_components
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # Pre-generate seeds to ensure reproducibility but independence
    np.random.seed(23665616)
    seeds = np.random.randint(0, 1000000, size=n_sims)
    
    print(f"Simulating {n_sims} trajectories for Regime={regime}, Rotation={rotation}...")
    
    for i in range(n_sims):
        # Run simulation
        # Start at Age 0 (bare land/just planted) to capture full lifecycle cost/benefit
        data = simulate_single_trajectory(
            params, state_space, price_data, R, Q, V, sigma, C_age,
            n_years=n_years,
            seed=seeds[i],
            initial_age=0,
            initial_regime=regime,
            initial_rotation=rotation
        )
        
        # Calculate NPV
        # NPV = sum(beta^t * reward_t)
        rewards = data['net_revenue']
        discounts = params.beta ** data['years']
        
        # Truncate if lengths differ (shouldn't happen based on code)
        min_len = min(len(rewards), len(discounts))
        npv = np.sum(rewards[:min_len] * discounts[:min_len])
        
        npvs.append(npv)

        # Record average prices
        avg_carbon_prices.append(np.mean(data['carbon_price']))
        avg_timber_prices.append(np.mean(data['timber_price']))

        # Record first harvest age
        harvest_indices = np.where(data['action'] == ACTION_HARVEST_REPLANT)[0]
        if len(harvest_indices) > 0:
            first_harvest_idx = harvest_indices[0]
            harvest_ages.append(data['age'][first_harvest_idx])
        else:
            harvest_ages.append(np.nan)

        # Decompose into carbon vs timber components using the recorded states/actions
        carbon_flows = np.zeros(min_len)
        timber_flows = np.zeros(min_len)
        for t in range(min_len):
            age = data['age'][t]
            regime_t = data['regime'][t]
            rotation_t = data['rotation'][t]
            action = data['action'][t]

            i_pc = (np.abs(pc_grid - data['carbon_price'][t])).argmin()
            i_pt = (np.abs(pt_grid - data['timber_price'][t])).argmin()

            s = state_space.tuple_to_state[(age, i_pc, i_pt, regime_t, rotation_t)]
            carbon_flows[t] = carbon_component[s, action]
            timber_flows[t] = timber_component[s, action]

        carbon_npvs.append(np.sum(carbon_flows * discounts[:min_len]))
        timber_npvs.append(np.sum(timber_flows * discounts[:min_len]))
        
    return np.array(npvs), np.array(carbon_npvs), np.array(timber_npvs), np.array(harvest_ages), np.array(avg_carbon_prices), np.array(avg_timber_prices)


def load_cached_scenario_results(csv_path, scenario):
    data = np.genfromtxt(csv_path, delimiter=',', names=True, skip_header=0)
    available_cols = set(data.dtype.names or ())
    if not REQUIRED_CACHE_COLUMNS.issubset(available_cols):
        missing_cols = sorted(REQUIRED_CACHE_COLUMNS - available_cols)
        raise ValueError(f"missing required columns: {', '.join(missing_cols)}")

    return {
        'scenario': scenario,
        'npvs': data['total'],
        'carbon': data['carbon'],
        'timber': data['timber'],
        'harvest_ages': data['harvest_age'],
        'avg_pc': data['avg_pc'],
        'avg_pt': data['avg_pt']
    }

def main(args=None):
    configure_paper_style()

    parser = argparse.ArgumentParser(description="Generate histograms of simulated utilities")
    parser.add_argument('--rerun', action='store_true', help='Force rerun of simulations even if cached CSVs exist')
    parser.add_argument(
        '--scenario-set',
        choices=['baseline', 'suboptimal'],
        default='baseline',
        help='Which scenario bundle to run'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save figures (defaults to a grid-specific plots path)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=DEFAULT_PRICE_GRID_SIZE,
        help=f'Grid size used to resolve default pickles (default: {DEFAULT_PRICE_GRID_SIZE})'
    )
    parser.add_argument(
        '--pickle-path',
        default=None,
        help='Optional model_results.pkl to use for baseline scenarios'
    )
    if args is None:
        args = parser.parse_args()

    scenario_set = getattr(args, 'scenario_set', 'baseline')
    grid_size = getattr(args, 'grid_size', DEFAULT_PRICE_GRID_SIZE)
    pickle_path_arg = getattr(args, 'pickle_path', None)
    output_dir = getattr(args, 'output_dir', None)
    scenarios = SCENARIOS_SUBOPTIMAL if scenario_set == 'suboptimal' else SCENARIOS_BASELINE
    default_output_grid_size = grid_size

    if pickle_path_arg is not None:
        if not os.path.exists(pickle_path_arg):
            print(f"Error: {pickle_path_arg} not found.")
            return
        try:
            explicit_results = load_results_pickle(pickle_path_arg)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        default_output_grid_size = explicit_results['params'].N_pc

    if output_dir is None:
        output_dir = plot_output_dir(f"utility_{scenario_set}", default_output_grid_size)

    results_storage = []
    
    # Simulation settings
    N_SIMS = 5000
    N_YEARS = 50 

    for sc in scenarios:
        if pickle_path_arg and sc['dir'] == 'baseline':
            pickle_path = pickle_path_arg
            expected_grid_size = None
        else:
            pickle_path = model_results_path(sc['dir'], default_output_grid_size)
            expected_grid_size = default_output_grid_size

        cache_root = os.path.dirname(pickle_path) or "."
        csv_path = scenario_cache_path(cache_root, sc['name'])

        if not os.path.exists(pickle_path):
            print(f"Warning: {pickle_path} not found. Skipping.")
            continue

        try:
            res = load_results_pickle(pickle_path, expected_grid_size=expected_grid_size)
        except ValueError as exc:
            print(f"  Warning: {exc}. Skipping.")
            continue
        pickle_mtime_ns = os.stat(pickle_path).st_mtime_ns

        if not getattr(args, 'rerun', False) and os.path.exists(csv_path):
            csv_mtime_ns = os.stat(csv_path).st_mtime_ns
            if csv_mtime_ns >= pickle_mtime_ns:
                print(f"\n--- Loading Cached Scenario: {sc['name']} ---")
                try:
                    cached_results = load_cached_scenario_results(csv_path, sc)
                except Exception as e:
                    print(f"  Error loading cache {csv_path}: {e}. Rerunning...")
                else:
                    results_storage.append(cached_results)
                    print(f"  Successfully loaded {len(cached_results['npvs'])} samples from {csv_path}")
                    continue
            else:
                print(
                    f"\n--- Processing Scenario: {sc['name']} ---\n"
                    f"  Cache {csv_path} is stale because {pickle_path} is newer. Rerunning..."
                )
        else:
            print(f"\n--- Processing Scenario: {sc['name']} ---")

        print(f"Loading results from {pickle_path}...")
        
        params = res['params']
        state_space = res['state_space']
        price_data = res['price_data']
        V = res['V']
        sigma = res['sigma']
        
        # Override policy if force_age is specified
        if 'force_age' in sc:
            print(f"  Overriding policy to force harvest at age {sc['force_age']}...")
            new_sigma = np.zeros_like(sigma)
            target_age = sc['force_age']
            for s_idx in range(state_space.N_states):
                a, _, _, _, _ = state_space.state_to_tuple[s_idx]
                # Force harvest at target_age, do nothing otherwise
                if a == target_age:
                    new_sigma[s_idx] = ACTION_HARVEST_REPLANT
                else:
                    new_sigma[s_idx] = ACTION_DO_NOTHING
            sigma = new_sigma

        print("Rebuilding matrices...")
        C_age = compute_carbon_curve(params)
        V_age = compute_volume_from_carbon(C_age, params)
        DeltaC_avg = compute_carbon_flows_averaging(C_age, params)
        DeltaC_perm = compute_carbon_flows_permanent(C_age)
        price_quality_factor = compute_price_quality_factor(params)
        
        R = build_reward_matrix(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
        Q, _, _, used_matrix_free = build_transition_representation(params, state_space, price_data)
        if used_matrix_free:
            print(
                "  Using matrix-free transition representation "
                f"(skipping {estimate_transition_nnz(params):,} explicit non-zeros)"
            )
        reward_components = build_reward_components(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
        
        npvs, carbon_npvs, timber_npvs, harvest_ages, avg_pc, avg_pt = run_simulations(
            N_SIMS, 
            regime=sc['regime'], 
            rotation=sc['rotation'],
            params=params,
            state_space=state_space,
            price_data=price_data,
            R=R, Q=Q, V=V, sigma=sigma, C_age=C_age,
            reward_components=reward_components,
            n_years=N_YEARS
        )
        
        results_storage.append({
            'scenario': sc,
            'npvs': npvs,
            'carbon': carbon_npvs,
            'timber': timber_npvs,
            'harvest_ages': harvest_ages,
            'avg_pc': avg_pc,
            'avg_pt': avg_pt
        })

        # Save realized NPVs and prices to CSV (including harvest_age for caching)
        os.makedirs(cache_root, exist_ok=True)
        data_to_save = np.column_stack((carbon_npvs, timber_npvs, npvs, avg_pc, avg_pt, harvest_ages))
        np.savetxt(csv_path, data_to_save, delimiter=',', header='carbon,timber,total,avg_pc,avg_pt,harvest_age', comments='')
        print(f"  Saved realized results to {csv_path}")

    if not results_storage:
        print("No results loaded. Exiting.")
        return

    # Print summary statistics
    print("\n" + "=" * 50)
    print(f"{'Scenario':<25} | {'Mean NPV':<12} | {'Std Dev':<12}")
    print("-" * 50)
    for res in results_storage:
        sc = res['scenario']
        npvs = res['npvs']
        mean_val = np.mean(npvs)
        std_val = np.std(npvs)
        print(f"{sc['name']:<25} | ${mean_val:>10,.0f} | ${std_val:>10,.0f}")
    print("=" * 50 + "\n")

    # 1. Main NPV Distributions
    print("\nGenerating main histogram plot...")
    if len(results_storage) <=4 :
        n_cols = 1
        n_rows = len(results_storage)
    else:
        n_cols = 2
        n_rows = int(np.ceil(len(results_storage)/2))
    
    fig_width = 3.5 if n_cols == 1 else 7.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 2.2 * n_rows), sharex=False)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Determine common bins
    all_npvs = np.concatenate([r['npvs'] for r in results_storage])
    min_val = np.min(all_npvs)
    max_val = 80000 # Cap at 80,000 as requested
    bins = np.linspace(min_val, max_val, 50)
    
    for i, res in enumerate(results_storage):
        sc = res['scenario']
        npvs = res['npvs']
        ax = axes[i]
        
        ax.hist(npvs, bins=bins, color=sc['color'], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(npvs), color=REFERENCE_COLOR, linestyle='dashed', linewidth=0.9, 
                  label=f'Mean: ${np.mean(npvs):,.0f}')
        ax.set_title(f'{sc["name"]}')
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=min_val, right=80000)
        ax.legend(frameon=False)
        style_axes(ax)
        
    for ax in axes:
        ax.set_xlabel('Net present value ($ per ha)')
    
    for ax in axes[len(results_storage):]:
        ax.remove()

    fig.subplots_adjust(left=0.18 if n_cols == 1 else 0.10, right=0.98, top=0.98, bottom=0.16, hspace=0.32, wspace=0.22)
    output_path = os.path.join(output_dir, f'utility_histograms_{scenario_set}.png')
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, output_path)
    print(f"Main plot saved to {output_path}")

    # 1b. CDF of NPVs (All scenarios on one plot)
    print("Generating CDF plot...")
    fig_cdf, ax_cdf = plt.subplots(figsize=(3.5, 2.6))
    for res in results_storage:
        sc = res['scenario']
        npvs = np.sort(res['npvs'])
        cdf = np.arange(1, len(npvs) + 1) / len(npvs)
        ls = sc.get('linestyle', '-')
        ax_cdf.plot(npvs, cdf, label=sc['name'], color=sc['color'], linewidth=1.2, linestyle=ls)
    
    ax_cdf.set_xlabel('Net present value ($ per ha)')
    ax_cdf.set_ylabel('Cumulative probability')
    ax_cdf.set_xlim(0, 80000)
    ax_cdf.legend(frameon=False)
    style_axes(ax_cdf)
    fig_cdf.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.18)

    output_path_cdf = os.path.join(output_dir, f'utility_cdf_{scenario_set}.png')
    save_figure(fig_cdf, output_path_cdf)
    print(f"CDF plot saved to {output_path_cdf}")

    # 3. Age at First Harvest
    print("Generating harvest age plot...")
    # Filter out Permanent scenario for harvest age plots as requested
    harvest_results = [res for res in results_storage if res['scenario']['name'] != 'Permanent']
    n_harvest_plots = len(harvest_results)

    if n_harvest_plots <= 4:
        n_cols_h = 1
        n_rows_h = n_harvest_plots
    else:
        n_cols_h = 2
        n_rows_h = int(np.ceil(n_harvest_plots / 2))

    fig_width_h = 3.5 if n_cols_h == 1 else 7.2
    fig3, axes3 = plt.subplots(n_rows_h, n_cols_h, figsize=(fig_width_h, 2.2 * n_rows_h), sharex=False)
    if isinstance(axes3, np.ndarray):
        axes3 = axes3.flatten()
    else:
        axes3 = [axes3]

    # Determine common bins for ages
    # Filter out NaNs for bin calculation
    all_ages = np.concatenate([r['harvest_ages'][~np.isnan(r['harvest_ages'])] for r in harvest_results])
    if len(all_ages) > 0:
        min_age = np.min(all_ages)
        max_age = np.max(all_ages)
        # Use integer bins for ages
        age_bins = np.arange(min(0, min_age), max_age + 2) - 0.5
    else:
        age_bins = 20

    for i, res in enumerate(harvest_results):
        sc = res['scenario']
        ages = res['harvest_ages']
        # Remove NaNs for plotting
        valid_ages = ages[~np.isnan(ages)]
        n_no_harvest = np.sum(np.isnan(ages))
        
        ax = axes3[i]
        if len(valid_ages) > 0:
            ax.hist(valid_ages, bins=age_bins, color=sc['color'], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(valid_ages), color=REFERENCE_COLOR, linestyle='dashed', linewidth=0.9, 
                      label=f'Mean Age: {np.mean(valid_ages):.1f}')
        
        title = f'{sc["name"]}'
        if n_no_harvest > 0:
            title += f' ({n_no_harvest}/{len(ages)} never harvested)'
            
        ax.set_title(title)
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
        style_axes(ax)
        
    for ax in axes3:
        ax.set_xlabel('Age at first harvest (years)')

    for ax in axes3[len(harvest_results):]:
        ax.remove()

    fig3.subplots_adjust(left=0.18 if n_cols_h == 1 else 0.10, right=0.98, top=0.98, bottom=0.16, hspace=0.32, wspace=0.22)
    output_path_ages = os.path.join(output_dir, f'harvest_age_histograms_{scenario_set}.png')
    save_figure(fig3, output_path_ages)
    print(f"Harvest age plot saved to {output_path_ages}")

if __name__ == "__main__":
    main()
