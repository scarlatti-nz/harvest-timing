"""
Script to generate histograms of simulated utilities (NPV) at time 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from tqdm import tqdm
from harvest_timing_model import (
    ModelParameters, StateSpace, simulate_single_trajectory,
    compute_carbon_curve, compute_volume_from_carbon,
    compute_carbon_flows_averaging, compute_carbon_flows_permanent,
    compute_price_quality_factor, build_reward_matrix, build_transition_matrix,
    ACTION_DO_NOTHING, ACTION_HARVEST_REPLANT, ACTION_SWITCH_PERMANENT, N_ACTIONS
)

SCENARIOS_BASELINE = [
    {
        'name': 'Averaging',
        'dir': 'baseline',
        'regime': 0,
        'rotation': 1,
        'color': '#3498db',
        'color_carbon': '#16a085',
        'color_timber': '#2980b9'
    },
    {
        'name': 'Permanent',
        'dir': 'baseline',
        'regime': 1,
        'rotation': 1,
        'color': '#e67e22',
        'color_carbon': '#d35400',
        'color_timber': '#f39c12'
    },
    {
        'name': 'Stock change',
        'dir': 'baseline',
        'regime': 2,
        'rotation': 1,
        'color': '#8e44ad',
        'color_carbon': '#27ae60',
        'color_timber': '#8e44ad'
    },
]

SCENARIOS_SUBOPTIMAL = [
    {
        'name': 'Averaging',
        'dir': 'baseline',
        'regime': 0,
        'rotation': 1,
        'color': '#3498db',
        'color_carbon': '#16a085',
        'color_timber': '#2980b9'
    },
    {
        'name': 'Stock change (force harvest at age 28)',
        'dir': 'baseline',
        'regime': 2,
        'rotation': 1,
        'color': '#27ae60',
        'color_carbon': '#16a085',
        'color_timber': '#2980b9',
        'force_age': 28,
    },
    {
        'name': 'Stock change (bank credits)',
        'dir': 'stock-change-bank-credit',
        'regime': 0,
        'rotation': 1,
        'color': '#633a01',
        'color_carbon': '#16a085',
        'color_timber': '#2980b9',
    }
]


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
    
    for i in tqdm(range(n_sims)):
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

def main():
    # Set global font sizes for all plots
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'figure.titlesize': 28
    })

    parser = argparse.ArgumentParser(description="Generate histograms of simulated utilities")
    parser.add_argument('--rerun', action='store_true', help='Force rerun of simulations even if cached CSVs exist')
    parser.add_argument(
        '--scenario-set',
        choices=['baseline', 'suboptimal'],
        default='baseline',
        help='Which scenario bundle to run'
    )
    args = parser.parse_args()

    scenarios = SCENARIOS_SUBOPTIMAL if args.scenario_set == 'suboptimal' else SCENARIOS_BASELINE

    results_storage = []
    
    # Simulation settings
    N_SIMS = 5000
    N_YEARS = 50 

    for sc in scenarios:
        csv_path = os.path.join('outputs', sc['dir'], f"{sc['name']}_realized_npvs.csv")
        
        # Try to load from cache if not rerunning
        if not args.rerun and os.path.exists(csv_path):
            print(f"\n--- Loading Cached Scenario: {sc['name']} ---")
            try:
                # Use genfromtxt to handle headers and potential NaNs in harvest_age
                data = np.genfromtxt(csv_path, delimiter=',', names=True, skip_header=0)
                
                # Check if all required columns are present
                required_cols = {'carbon', 'timber', 'total', 'avg_pc', 'avg_pt', 'harvest_age'}
                if required_cols.issubset(set(data.dtype.names)):
                    results_storage.append({
                        'scenario': sc,
                        'npvs': data['total'],
                        'carbon': data['carbon'],
                        'timber': data['timber'],
                        'harvest_ages': data['harvest_age'],
                        'avg_pc': data['avg_pc'],
                        'avg_pt': data['avg_pt']
                    })
                    print(f"  Successfully loaded {len(data)} samples from {csv_path}")
                    continue
                else:
                    print(f"  Warning: Cache file {csv_path} is missing required columns. Rerunning...")
            except Exception as e:
                print(f"  Error loading cache {csv_path}: {e}. Rerunning...")

        # If we reach here, we either forced rerun or cache was invalid/missing
        pickle_path = os.path.join('outputs', sc['dir'], 'model_results.pkl')
        if not os.path.exists(pickle_path):
            print(f"Warning: {pickle_path} not found. Skipping.")
            continue

        print(f"\n--- Processing Scenario: {sc['name']} ---")
        print(f"Loading results from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            res = pickle.load(f)
        
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
        Q_sa, _, _ = build_transition_matrix(params, state_space, price_data)
        reward_components = build_reward_components(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
        
        npvs, carbon_npvs, timber_npvs, harvest_ages, avg_pc, avg_pt = run_simulations(
            N_SIMS, 
            regime=sc['regime'], 
            rotation=sc['rotation'],
            params=params,
            state_space=state_space,
            price_data=price_data,
            R=R, Q=Q_sa, V=V, sigma=sigma, C_age=C_age,
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
        os.makedirs(os.path.join('outputs', sc['dir']), exist_ok=True)
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows), sharex=False)
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
        ax.axvline(np.mean(npvs), color='red', linestyle='dashed', linewidth=2, 
                  label=f'Mean: ${np.mean(npvs):,.0f}')
        ax.set_title(f'{sc["name"]}', fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=min_val, right=80000)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    for ax in axes:
        ax.set_xlabel('Net present value ($/ha)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Distribution of NPV at planting under optimal decision-making (n=5000)", y=0.98, fontweight='bold')
    output_path = 'plots/utility_histograms.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Main plot saved to {output_path}")

    # 1b. CDF of NPVs (All scenarios on one plot)
    print("Generating CDF plot...")
    plt.figure(figsize=(16, 9))
    for res in results_storage:
        sc = res['scenario']
        npvs = np.sort(res['npvs'])
        cdf = np.arange(1, len(npvs) + 1) / len(npvs)
        ls = sc.get('linestyle', '-')
        plt.plot(npvs, cdf, label=sc['name'], color=sc['color'], linewidth=3, linestyle=ls)
    
    plt.title('Cumulative distribution of realized utilities (NPV)', fontweight='bold')
    plt.xlabel('Net present value ($/ha)')
    plt.ylabel('Cumulative probability')
    plt.xlim(0, 80000)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path_cdf = 'plots/utility_cdf.png'
    plt.savefig(output_path_cdf, dpi=150)
    plt.close()
    print(f"CDF plot saved to {output_path_cdf}")

    # 2. Carbon vs Timber Components
    print("Generating component plot...")
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows), sharex=True)
    if isinstance(axes2, np.ndarray):
        axes2 = axes2.flatten()
    else:
        axes2 = [axes2]

    def plot_components(ax, carbon_vals, timber_vals, title, color_carbon, color_timber):
        combined_min = min(np.min(carbon_vals), np.min(timber_vals))
        combined_max = 80000 # Cap at 80,000 as requested
        bins = np.linspace(combined_min, combined_max, 50)

        ax.hist(timber_vals, bins=bins, color=color_timber, alpha=0.6, edgecolor='black', label='Timber utility')
        ax.axvline(np.mean(timber_vals), color=color_timber, linestyle='dashed', linewidth=2, label=f'Timber mean: ${np.mean(timber_vals):,.0f}')

        ax.hist(carbon_vals, bins=bins, color=color_carbon, alpha=0.6, edgecolor='black', label='Carbon utility')
        ax.axvline(np.mean(carbon_vals), color=color_carbon, linestyle='dashed', linewidth=2, label=f'Carbon mean: ${np.mean(carbon_vals):,.0f}')

        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=combined_min, right=80000)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for i, res in enumerate(results_storage):
        sc = res['scenario']
        plot_components(
            axes2[i], res['carbon'], res['timber'],
            title=f'Utility Components: {sc["name"]}',
            color_carbon=sc['color_carbon'], color_timber=sc['color_timber']
        )
    
    for ax in axes2:
        ax.set_xlabel('Net present value ($/ha)')

    plt.tight_layout()
    output_path_components = 'plots/utility_histograms_components.png'
    plt.savefig(output_path_components, dpi=150)
    print(f"Component plot saved to {output_path_components}")

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

    fig3, axes3 = plt.subplots(n_rows_h, n_cols_h, figsize=(16, 5 * n_rows_h), sharex=False)
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
            ax.axvline(np.mean(valid_ages), color='red', linestyle='dashed', linewidth=2, 
                      label=f'Mean Age: {np.mean(valid_ages):.1f}')
        
        title = f'{sc["name"]}'
        if n_no_harvest > 0:
            title += f' ({n_no_harvest}/{len(ages)} never harvested)'
            
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlim(left=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    for ax in axes3:
        ax.set_xlabel('Age at first harvest (years)')

    fig3.suptitle("Distribution of harvest age under optimal decision-making (n=5000)", y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path_ages = 'plots/harvest_age_histograms.png'
    plt.savefig(output_path_ages, dpi=150)
    print(f"Harvest age plot saved to {output_path_ages}")

if __name__ == "__main__":
    main()
