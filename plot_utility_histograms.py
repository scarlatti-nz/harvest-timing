"""
Script to generate histograms of simulated utilities (NPV) at time 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from harvest_timing_model import (
    ModelParameters, StateSpace, simulate_single_trajectory,
    compute_carbon_curve, compute_volume_from_carbon,
    compute_carbon_flows_averaging, compute_carbon_flows_permanent,
    compute_price_quality_factor, build_reward_matrix, build_transition_matrix
)

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
    n_years: int = 200
):
    """Run N simulations and return list of NPVs."""
    npvs = []
    
    # Pre-generate seeds to ensure reproducibility but independence
    np.random.seed(42 + regime * 100 + rotation)
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
        
    return np.array(npvs)

def main():
    pickle_path = 'noswitch/model_results.pkl'
    if not os.path.exists(pickle_path):
        print("Model results not found. Please run harvest_timing_model.py first.")
        return

    print(f"Loading results from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    params = results['params']
    state_space = results['state_space']
    price_data = results['price_data']
    V = results['V']
    sigma = results['sigma']
    
    # Rebuild matrices if needed (usually R and Q are large, maybe not pickled? 
    # Check pickle content in harvest_timing_model.py. 
    # It pickles V, sigma, but R and Q are not in the list of saved items in harvest_timing_model.py lines 1118-1126.
    # So we need to rebuild them.)
    
    print("Rebuilding matrices...")
    C_age = compute_carbon_curve(params)
    V_age = compute_volume_from_carbon(C_age, params)
    DeltaC_avg = compute_carbon_flows_averaging(C_age, params)
    DeltaC_perm = compute_carbon_flows_permanent(C_age)
    price_quality_factor = compute_price_quality_factor(params)
    
    R = build_reward_matrix(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
    Q = build_transition_matrix(params, state_space, price_data)
    
    # Simulation settings
    N_SIMS = 1000
    N_YEARS = 200 # Long horizon to approximate infinite horizon NPV
    
    # 1. Averaging, First Rotation
    npvs_avg = run_simulations(
        N_SIMS, 
        regime=0, 
        rotation=1,
        params=params,
        state_space=state_space,
        price_data=price_data,
        R=R, Q=Q, V=V, sigma=sigma, C_age=C_age,
        n_years=N_YEARS
    )
    
    # 2. Permanent
    npvs_perm = run_simulations(
        N_SIMS, 
        regime=1, 
        rotation=1, # Rotation index doesn't matter much for permanent, but 1 is fine
        params=params,
        state_space=state_space,
        price_data=price_data,
        R=R, Q=Q, V=V, sigma=sigma, C_age=C_age,
        n_years=N_YEARS
    )
    
    # Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Determine common bins
    all_data = np.concatenate([npvs_avg, npvs_perm])
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    bins = np.linspace(min_val, max_val, 50)
    
    # Panel 1: Averaging
    axes[0].hist(npvs_avg, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(npvs_avg), color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${np.mean(npvs_avg):,.0f}')
    axes[0].set_title('Distribution of Realized Utilities: First Rotation Averaging', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Permanent
    axes[1].hist(npvs_perm, bins=bins, color='#8e44ad', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(npvs_perm), color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${np.mean(npvs_perm):,.0f}')
    axes[1].set_title('Distribution of Realized Utilities: Permanent Regime', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Net Present Value ($/ha)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'plots/utility_histograms.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()


