import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from typing import Dict, List

# Import definitions from main model
from harvest_timing_model import (
    ModelParameters, 
    StateSpace, 
    simulate_price_paths,
    compute_carbon_curve,
    compute_volume_from_carbon,
    ACTION_HARVEST_REPLANT
)

# Transcribed data from NEFD-2022 Table 4: Forest area by annual age class as at 1 April 2022
# Age (1 to 54) and Area (ha)
AGE_DISTRIBUTION = {
    1: 42319, 2: 53696, 3: 55758, 4: 44725, 5: 42312, 6: 42230, 7: 42994, 8: 43680, 9: 48295, 10: 59098,
    11: 54557, 12: 40792, 13: 36161, 14: 35143, 15: 35161, 16: 33035, 17: 37814, 18: 47595, 19: 56250, 20: 58802,
    21: 64987, 22: 64453, 23: 66625, 24: 74914, 25: 81492, 26: 96055, 27: 81825, 28: 93424, 29: 57748, 30: 43274,
    31: 12796, 32: 11943, 33: 9783, 34: 6645, 35: 6513, 36: 9519, 37: 10523, 38: 8058, 39: 6283, 40: 4829,
    41: 4599, 42: 5066, 43: 3569, 44: 4083, 45: 2244, 46: 1853, 47: 1942, 48: 1379, 49: 940, 50: 845,
    51: 566, 52: 1110, 53: 788, 54: 463
}

def simulate_forest_inventory(
    results: Dict,
    regime: int,
    n_years: int = 20,
    target_total_area: float = 1.7e6,
    seed: int = 42
):
    params = results['params']
    state_space = results['state_space']
    price_data = results['price_data']
    V_results = results['V']
    sigma = results['sigma']
    
    # 1. Initialize Forest Blocks
    # Each age class from the table is a "block" with a specific area
    current_sum = sum(AGE_DISTRIBUTION.values())
    scale_factor = target_total_area / current_sum
    
    # blocks list: each element is [age, area, rotation]
    blocks = []
    for age, area in AGE_DISTRIBUTION.items():
        blocks.append({
            'age': age,
            'area': area * scale_factor,
            'rotation': 1  # Assume all start in first rotation
        })
    
    # 2. Generate Price Trajectories
    # Use params from model to ensure consistency
    pc_path = simulate_price_paths(
        params.pc_mean, params.pc_rho, params.pc_sigma,
        n_paths=1, n_periods=n_years + 1, seed=seed, p0=params.pc_0
    ).flatten()
    
    pt_path = simulate_price_paths(
        params.pt_mean, params.pt_rho, params.pt_sigma,
        n_paths=1, n_periods=n_years + 1, seed=seed + 1, p0=params.pt_mean # Assume starting at mean if p0 not specified
    ).flatten()
    
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # 3. Growth Curves (for volume calculation)
    C_age = compute_carbon_curve(params)
    V_age = compute_volume_from_carbon(C_age, params)
    
    # 4. Simulation Loop
    annual_harvest_vol = []
    
    for t in range(n_years):
        yearly_vol = 0
        
        # Current prices at start of year t
        pc_t = pc_path[t]
        pt_t = pt_path[t]
        
        # Find nearest price indices in grid
        i_pc = (np.abs(pc_grid - pc_t)).argmin()
        i_pt = (np.abs(pt_grid - pt_t)).argmin()
        
        for block in blocks:
            age = block['age']
            rotation = block['rotation']
            
            # Find state index
            # State tuple: (age, i_pc, i_pt, regime, rotation)
            # Clamp age to A_max
            a_state = min(age, params.A_max)
            state_tuple = (a_state, i_pc, i_pt, regime, rotation)
            s = state_space.tuple_to_state[state_tuple]
            
            # Get optimal action
            action = sigma[s]
            
            if action == ACTION_HARVEST_REPLANT:
                # Harvest
                vol_ha = V_age[age] if age < len(V_age) else V_age[-1]
                yearly_vol += vol_ha * block['area']
                
                # Reset block
                block['age'] = 0
                block['rotation'] = 2
            else:
                # Grow
                block['age'] = min(age + 1, params.A_max) # Technically we can go beyond A_max in reality, but model caps at A_max
                
        annual_harvest_vol.append(yearly_vol)
        
    return {
        'years': np.arange(1, n_years + 1),
        'harvest_vol': annual_harvest_vol,
        'carbon_prices': pc_path[:n_years],
        'timber_prices': pt_path[:n_years]
    }

def main():
    parser = argparse.ArgumentParser(description="Simulate 1.7m ha forest inventory over 20 years.")
    parser.add_argument('--pickle-path', type=str, default='outputs/baseline/model_results.pkl',
                        help='Path to model results pickle file')
    parser.add_argument('--regime', type=int, choices=[0, 1, 2], default=0,
                        help='Regime for all forests (0: averaging, 1: permanent, 2: pre-2023 stock-change)')
    parser.add_argument('--n-years', type=int, default=50, help='Number of years to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for price paths')
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.pickle_path):
        print(f"Error: {args.pickle_path} not found.")
        return
        
    with open(args.pickle_path, 'rb') as f:
        results = pickle.load(f)
        
    regime_names = {0: 'Averaging', 1: 'Permanent', 2: 'Pre-2023 Stock-Change'}
    regime_name = regime_names[args.regime]
    
    # Run simulation
    sim_results = simulate_forest_inventory(results, args.regime, n_years=args.n_years, seed=args.seed)
    
    # Plotting
    os.makedirs('plots', exist_ok=True)
    
    # Chart 1: Harvest Volume
    plt.figure(figsize=(10, 6))
    plt.plot(sim_results['years'], np.array(sim_results['harvest_vol']) / 1e6, marker='o', color='forestgreen')
    plt.title(f'Annual Harvest Volume - {regime_name} Regime', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Harvest Volume (Million m³)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(sim_results['years'])
    
    chart1_path = f'plots/harvest_volume_{regime_name.lower().replace(" ", "_")}.png'
    plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {chart1_path}")
    
    # Chart 2: Harvest Volume + Prices
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Volume
    ax1.plot(sim_results['years'], np.array(sim_results['harvest_vol']) / 1e6, marker='o', color='forestgreen', label='Harvest Volume')
    ax1.set_ylabel('Volume (Million m³)', fontsize=12)
    ax1.set_title(f'Simulation Results - {regime_name} Regime', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Subplot 2: Prices
    ax2_timber = ax2
    ax2_carbon = ax2.twinx()
    
    p1, = ax2_timber.plot(sim_results['years'], sim_results['timber_prices'], color='brown', label='Timber Price')
    p2, = ax2_carbon.plot(sim_results['years'], sim_results['carbon_prices'], color='blue', label='Carbon Price')
    
    ax2_timber.set_ylabel('Timber Price ($/m³)', color='brown', fontsize=12)
    ax2_carbon.set_ylabel('Carbon Price ($/tCO₂)', color='blue', fontsize=12)
    ax2_timber.set_xlabel('Year', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax2.legend(handles=[p1, p2], loc='upper left')
    
    plt.xticks(sim_results['years'])
    plt.tight_layout()
    
    chart2_path = f'plots/harvest_volume_with_prices_{regime_name.lower().replace(" ", "_")}.png'
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"Chart with prices saved to {chart2_path}")

if __name__ == "__main__":
    main()


