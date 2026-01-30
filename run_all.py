"""
Script to run multiple model scenarios by overriding parameters.
"""

import argparse
from harvest_timing_model import main as run_model
from harvest_timing_model import ModelParameters
from plot_utility_histograms import main as plot_utility_histograms
from plot_results import main as plot_results

class MockArgs:
    def __init__(self, temp_dir='temp', grid_size=9):
        self.temp_dir = temp_dir
        self.grid_size = grid_size
        self.sanity_checks = False

def run_scenarios(grid_size=9):
    scenarios = [
        {
            'name': 'baseline',
            # switch_cost 1e9 forces you to stay in averaging regime, high harvest penalty forces permanent regime not to harvest
            'overrides': {
                'switch_cost': 1e9,
                'harvest_penalty_per_m3': 10000.0
            }
        },
        {
            'name': 'low-volatility',
            # much lower carbon price volatility
            'overrides': {
                'switch_cost': 1e9,
                'harvest_penalty_per_m3': 10000.0,
                'pc_rho': 0.8,
                'pc_sigma': 0.05
            }
        },
        {
            'name': 'low-expectations',
            # carbon price expected to trend downwards over time to $10/CO2
            'overrides': {
                'switch_cost': 1e9,
                'harvest_penalty_per_m3': 10000.0,
                'pc_0': 50.0,
                'pc_mean': 10.0,
                'pc_rho': 0.96,
                'pc_sigma': 0.2
            }
        },
        {
            'name': 'high-expectations',
            # carbon price expected to trend upwards over time to $300t/CO2
            'overrides': {
                'switch_cost': 1e9,
                'harvest_penalty_per_m3': 10000.0,
                'pc_0': 50.0,
                'pc_mean': 300.0,
                'pc_rho': 0.96,
                'pc_sigma': 0.2
            }
        },
        {
            'name': 'stock-change-bank-credit',
            'overrides': {
                'switch_cost': 1e9,
                'harvest_penalty_per_m3': 10000.0,
                'carbon_credit_max_age': 10, # Bank credits beyond age 10 to surrender at harvest - functionally no carbon revenue
            }
        }
    ]

    for sc in scenarios:
        print(f"\n\n{'#'*80}")
        print(f"RUNNING SCENARIO: {sc['name']}")
        print(f"{'#'*80}\n")
        
        # Create parameters with specific overrides
        # We use grid_size from args for consistency
        params = ModelParameters(
            N_pt=grid_size,
            N_pc=grid_size,
            **sc['overrides']
        )
        
        # Setup mock arguments for the main function
        args = MockArgs(
            temp_dir=sc['name'],
            grid_size=grid_size
        )
        
        # Run the model
        run_model(args=args, params=params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple harvest timing scenarios")
    parser.add_argument('--grid-size', type=int, default=9, help='Grid size for price discretization')
    args = parser.parse_args()
    
    run_scenarios(grid_size=args.grid_size)
    plot_utility_histograms()
    plot_results()
