"""
Real Options Model for Forest Harvest Timing

Infinite-horizon DP using QuantEcon's DiscreteDP. State includes age, prices, 
ETS regime, and rotation number. Carbon credits earned only in first rotation
(averaging) up to age 16. Permanent regime has carbon liability on harvest.
"""

import numpy as np
from quantecon.markov import tauchen, DiscreteDP
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import warnings
import argparse


# =============================================================================
# 1. Configuration / Parameter Block
# =============================================================================

@dataclass
class ModelParameters:
    """Economic and biophysical parameters for the harvest timing model."""
    
    # Discounting
    discount_rate: float = 0.06
    
    # Age parameters
    A_max: int = 100  # Maximum age in state space (no forced harvest)
    carbon_credit_max_age: int = 16  # Carbon credits stop at this age
    
    # Price grid sizes
    N_pc: int = 7   # Number of carbon price states
    N_pt: int = 7   # Number of timber price states
    
    # Price process parameters (AR(1) in logs)
    # Carbon price
    pc_mean: float = 50.0      # Mean carbon price ($/tCO2)
    pc_rho: float = 0.9        # AR(1) persistence
    pc_sigma: float = 0.15     # Volatility of log price
    
    # Timber price
    pt_mean: float = 150.0     # Mean timber price ($/m³)
    pt_rho: float = 0.85       # AR(1) persistence
    pt_sigma: float = 0.20     # Volatility of log price
    
    # Cost parameters
    harvest_cost_per_m3: float = 75.0   # $/m³
    replant_cost: float = 2000.0        # $ per hectare
    maintenance_cost: float = 50.0      # $ per year per hectare
    switch_cost: float = 500.0          # Admin cost to switch to permanent
    
    # Optional harvest penalty ($/m³) - set to 0 to disable
    harvest_penalty_per_m3: float = 0.0
    
    # Permanent regime carbon liability parameters
    # On harvest: pay carbon_price * carbon_stock
    # 50% paid instantly, 50% paid in equal increments over 10 years
    carbon_liability_years: int = 10
    carbon_liability_instant_fraction: float = 0.5
    
    # Growth parameters
    C_max: float = 2622.0    # Maximum carbon stock (tCO2/ha)
    k_carbon: float = 0.038  # Carbon accumulation rate
    timber_per_tonne_carbon: float = 1.2  # Timber volume (m³) per tonne of carbon
    
    @property
    def beta(self) -> float:
        """Discount factor."""
        return 1.0 / (1.0 + self.discount_rate)
    
    @property
    def N_a(self) -> int:
        """Number of age states."""
        return self.A_max + 1
    
    @property
    def N_regimes(self) -> int:
        """Number of ETS regimes (averaging=0, permanent=1)."""
        return 2
    
    @property
    def N_rotations(self) -> int:
        """Number of rotation groups (first=1, later=2)."""
        return 2
    
    def carbon_liability_npv_factor(self) -> float:
        """
        NPV factor for carbon liability on permanent harvest.
        
        Liability = 50% instant + 50% spread over 10 years.
        NPV = 0.5 + 0.5 * sum_{t=1}^{10} (1/10) * beta^t
            = 0.5 + 0.05 * sum_{t=1}^{10} beta^t
            = 0.5 + 0.05 * beta * (1 - beta^10) / (1 - beta)
        """
        instant = self.carbon_liability_instant_fraction
        spread = 1.0 - instant
        annual_payment = spread / self.carbon_liability_years
        
        # PV of annuity over liability_years
        pv_annuity = 0.0
        for t in range(1, self.carbon_liability_years + 1):
            pv_annuity += annual_payment * (self.beta ** t)
        
        return instant + pv_annuity


# =============================================================================
# 2. Growth and Carbon Functions
# =============================================================================

def compute_carbon_curve(params: ModelParameters) -> np.ndarray:
    """
    Compute carbon stock as function of age.
    C(a) = C_max * (1 - exp(-k * a))^2
    
    Returns:
        C_age: Array of shape (N_a,) with carbon at each age.
    """
    ages = np.arange(params.N_a)
    C_age = params.C_max * (1 - np.exp(-params.k_carbon * ages)) ** 2
    return C_age


def compute_volume_from_carbon(C_age: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Compute timber volume as a constant factor of carbon stock.
    V(a) = timber_per_tonne_carbon * C(a)
    
    Returns:
        V_age: Array of shape (N_a,) with volume at each age.
    """
    return params.timber_per_tonne_carbon * C_age


def compute_carbon_flows_averaging(
    C_age: np.ndarray, 
    params: ModelParameters
) -> np.ndarray:
    """
    Compute annual carbon credit flows for AVERAGING regime.
    Credits only earned up to carbon_credit_max_age (first rotation only).
    
    ΔC(a) = C(a) - C(a-1) for a <= carbon_credit_max_age, else 0
    
    Returns:
        DeltaC: Array of shape (N_a,) with annual carbon flows.
    """
    DeltaC = np.zeros_like(C_age)
    # Credits only earned for ages 1 through carbon_credit_max_age
    max_age = min(params.carbon_credit_max_age, len(C_age) - 1)
    DeltaC[1:max_age + 1] = np.diff(C_age)[:max_age]
    return DeltaC


def compute_carbon_flows_permanent(C_age: np.ndarray) -> np.ndarray:
    """
    Compute annual carbon credit flows for PERMANENT regime.
    Credits earned indefinitely (all ages).
    
    ΔC(a) = C(a) - C(a-1)
    
    Returns:
        DeltaC: Array of shape (N_a,) with annual carbon flows.
    """
    DeltaC = np.zeros_like(C_age)
    DeltaC[1:] = np.diff(C_age)
    return DeltaC


# =============================================================================
# 3. Price Process Discretization
# =============================================================================

def discretize_price_process(
    mean: float,
    rho: float,
    sigma: float,
    n_states: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize an AR(1) process in logs using Tauchen's method.
    
    log(P_t) = (1-rho)*log(mean) + rho*log(P_{t-1}) + sigma*epsilon
    
    Returns:
        grid: Price levels (in levels, not logs)
        P: Transition matrix (n_states x n_states)
    """
    if n_states == 1:
        # Degenerate case: single price state
        return np.array([mean]), np.array([[1.0]])
    
    # Convert to log scale for Tauchen
    log_mean = np.log(mean)
    
    # Unconditional mean of log process: mu in AR(1) is mu = (1-rho)*E[y]
    mu = log_mean * (1 - rho)
    
    # Use Tauchen method: tauchen(n, rho, sigma, mu=0.0, n_std=3)
    mc = tauchen(n_states, rho, sigma, mu=mu, n_std=3)
    
    # Convert grid from logs to levels
    grid = np.exp(mc.state_values)
    P = mc.P
    
    return grid, P


def build_price_grids(params: ModelParameters) -> Dict:
    """
    Build price grids and transition matrices for carbon and timber prices.
    
    Returns:
        Dictionary with grids and transition matrices.
    """
    # Carbon price
    pc_grid, Pc = discretize_price_process(
        params.pc_mean, params.pc_rho, params.pc_sigma, params.N_pc
    )
    
    # Timber price
    pt_grid, Pt = discretize_price_process(
        params.pt_mean, params.pt_rho, params.pt_sigma, params.N_pt
    )
    
    return {
        'pc_grid': pc_grid,
        'pt_grid': pt_grid,
        'Pc': Pc,
        'Pt': Pt
    }


def simulate_price_paths(
    mean: float,
    rho: float,
    sigma: float,
    n_paths: int = 1000,
    n_periods: int = 100,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simulate price paths from an AR(1) process in logs.
    
    log(P_t) = (1-rho)*log(mean) + rho*log(P_{t-1}) + sigma*epsilon
    
    Args:
        mean: Long-run mean price level
        rho: AR(1) persistence parameter
        sigma: Standard deviation of innovations
        n_paths: Number of paths to simulate
        n_periods: Number of time periods
        seed: Random seed for reproducibility
    
    Returns:
        paths: Array of shape (n_paths, n_periods) with price levels
    """
    if seed is not None:
        np.random.seed(seed)
    
    log_mean = np.log(mean)
    mu = log_mean * (1 - rho)  # Intercept term
    
    # Initialize log prices at the unconditional mean
    log_prices = np.zeros((n_paths, n_periods))
    log_prices[:, 0] = log_mean
    
    # Generate innovations
    innovations = np.random.normal(0, sigma, (n_paths, n_periods - 1))
    
    # Simulate AR(1) process
    for t in range(1, n_periods):
        log_prices[:, t] = mu + rho * log_prices[:, t - 1] + innovations[:, t - 1]
    
    # Convert to levels
    prices = np.exp(log_prices)
    
    return prices


def plot_price_paths(
    params: ModelParameters,
    n_paths: int = 1000,
    n_periods: int = 100,
    save_path: Optional[str] = None
):
    """
    Plot simulated price paths for carbon and timber prices.
    
    Shows the stochastic nature of the AR(1) price processes.
    """
    # Simulate paths
    carbon_paths = simulate_price_paths(
        params.pc_mean, params.pc_rho, params.pc_sigma,
        n_paths=n_paths, n_periods=n_periods, seed=42
    )
    timber_paths = simulate_price_paths(
        params.pt_mean, params.pt_rho, params.pt_sigma,
        n_paths=n_paths, n_periods=n_periods, seed=123
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    years = np.arange(n_periods)
    
    # --- Carbon price paths ---
    ax = axes[0]
    
    # Plot individual paths with transparency
    for i in range(n_paths):
        ax.plot(years, carbon_paths[i, :], color='#3498db', alpha=0.03, linewidth=0.5)
    
    # Plot percentiles
    p5 = np.percentile(carbon_paths, 5, axis=0)
    p25 = np.percentile(carbon_paths, 25, axis=0)
    p50 = np.percentile(carbon_paths, 50, axis=0)
    p75 = np.percentile(carbon_paths, 75, axis=0)
    p95 = np.percentile(carbon_paths, 95, axis=0)
    
    ax.fill_between(years, p5, p95, alpha=0.2, color='#3498db', label='5th-95th percentile')
    ax.fill_between(years, p25, p75, alpha=0.3, color='#3498db', label='25th-75th percentile')
    ax.plot(years, p50, color='#2c3e50', linewidth=2, label='Median')
    ax.axhline(params.pc_mean, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean = ${params.pc_mean:.0f}')
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Carbon Price ($/tCO₂)', fontsize=11)
    ax.set_title(f'Carbon Price Paths (n={n_paths})\nρ={params.pc_rho}, σ={params.pc_sigma}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, n_periods - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    # --- Timber price paths ---
    ax = axes[1]
    
    # Plot individual paths with transparency
    for i in range(n_paths):
        ax.plot(years, timber_paths[i, :], color='#27ae60', alpha=0.03, linewidth=0.5)
    
    # Plot percentiles
    p5 = np.percentile(timber_paths, 5, axis=0)
    p25 = np.percentile(timber_paths, 25, axis=0)
    p50 = np.percentile(timber_paths, 50, axis=0)
    p75 = np.percentile(timber_paths, 75, axis=0)
    p95 = np.percentile(timber_paths, 95, axis=0)
    
    ax.fill_between(years, p5, p95, alpha=0.2, color='#27ae60', label='5th-95th percentile')
    ax.fill_between(years, p25, p75, alpha=0.3, color='#27ae60', label='25th-75th percentile')
    ax.plot(years, p50, color='#2c3e50', linewidth=2, label='Median')
    ax.axhline(params.pt_mean, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean = ${params.pt_mean:.0f}')
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Timber Price ($/m³)', fontsize=11)
    ax.set_title(f'Timber Price Paths (n={n_paths})\nρ={params.pt_rho}, σ={params.pt_sigma}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, n_periods - 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Price paths plot saved to: {save_path}")
    
    plt.show()


# =============================================================================
# 4. State Space Construction
# =============================================================================

@dataclass
class StateSpace:
    """Container for state space mappings."""
    N_states: int
    state_to_tuple: Dict[int, Tuple[int, int, int, int, int]]
    tuple_to_state: Dict[Tuple[int, int, int, int, int], int]
    
    # Dimension info
    N_a: int
    N_pc: int
    N_pt: int
    N_regimes: int
    N_rotations: int


def build_state_space(params: ModelParameters) -> StateSpace:
    """
    Build state space enumeration.
    
    State tuple: (age, i_pc, i_pt, regime, rotation)
    - age: 0..A_max
    - i_pc: 0..N_pc-1 (carbon price index)
    - i_pt: 0..N_pt-1 (timber price index)
    - regime: 0=averaging, 1=permanent
    - rotation: 1=first, 2=later
    
    Returns:
        StateSpace object with mappings.
    """
    state_to_tuple = {}
    tuple_to_state = {}
    
    s = 0
    for a in range(params.N_a):
        for i_pc in range(params.N_pc):
            for i_pt in range(params.N_pt):
                for regime in range(params.N_regimes):
                    for rotation in range(1, params.N_rotations + 1):
                        state_tuple = (a, i_pc, i_pt, regime, rotation)
                        state_to_tuple[s] = state_tuple
                        tuple_to_state[state_tuple] = s
                        s += 1
    
    N_states = s
    
    return StateSpace(
        N_states=N_states,
        state_to_tuple=state_to_tuple,
        tuple_to_state=tuple_to_state,
        N_a=params.N_a,
        N_pc=params.N_pc,
        N_pt=params.N_pt,
        N_regimes=params.N_regimes,
        N_rotations=params.N_rotations
    )


# =============================================================================
# 5. Action Space
# =============================================================================

# Action indices
ACTION_DO_NOTHING = 0
ACTION_HARVEST_REPLANT = 1
ACTION_SWITCH_PERMANENT = 2

N_ACTIONS = 3


# =============================================================================
# 6. Reward Matrix Construction
# =============================================================================

def build_reward_matrix(
    params: ModelParameters,
    state_space: StateSpace,
    price_data: Dict,
    V_age: np.ndarray,
    C_age: np.ndarray,
    DeltaC_avg: np.ndarray,
    DeltaC_perm: np.ndarray
) -> np.ndarray:
    """
    Construct reward matrix R[state, action].
    
    Carbon credits:
    - Averaging: first rotation only, up to age 16
    - Permanent: all rotations, all ages (indefinitely)
    
    Harvest costs:
    - Averaging: harvest_cost + optional penalty
    - Permanent: harvest_cost + optional penalty + carbon_liability (NPV)
    
    Returns:
        R: Array of shape (N_states, N_actions)
    """
    R = np.zeros((state_space.N_states, N_ACTIONS))
    
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # NPV factor for carbon liability
    liability_npv_factor = params.carbon_liability_npv_factor()
    
    for s in range(state_space.N_states):
        a, i_pc, i_pt, regime, rotation = state_space.state_to_tuple[s]
        
        # Get price levels
        pc = pc_grid[i_pc]
        pt = pt_grid[i_pt]
        
        # Get volume and carbon
        volume = V_age[a]
        carbon_stock = C_age[a]
        
        # Carbon credit flow depends on regime
        if regime == 0:  # Averaging
            # Only in first rotation, up to age 16
            if rotation == 1:
                delta_C = DeltaC_avg[a]
            else:
                delta_C = 0.0
        else:  # Permanent
            # Credits continue indefinitely (all ages, all rotations)
            delta_C = DeltaC_perm[a]
        
        # === Action 0: Do nothing ===
        R_carbon = pc * delta_C
        R_cost = -params.maintenance_cost
        R[s, ACTION_DO_NOTHING] = R_carbon + R_cost
        
        # === Action 1: Harvest and replant ===
        # Timber revenue minus harvest cost
        harvest_cost = params.harvest_cost_per_m3 + params.harvest_penalty_per_m3
        R_timber = pt * volume - harvest_cost * volume
        R_replant = -params.replant_cost
        
        # Carbon liability for permanent regime
        if regime == 1:  # Permanent
            # Must pay carbon_price * carbon_stock on harvest
            # 50% instant, 50% over 10 years (use NPV)
            carbon_liability = pc * carbon_stock * liability_npv_factor
            R[s, ACTION_HARVEST_REPLANT] = R_timber + R_replant - carbon_liability
        else:
            # Averaging: no carbon liability on harvest
            R[s, ACTION_HARVEST_REPLANT] = R_timber + R_replant
        
        # === Action 2: Switch to permanent ===
        if regime == 0:  # Only meaningful when in averaging
            # Pay switch cost, receive carbon flow as in do-nothing
            switch_penalty = params.switch_cost
            
            # For second+ rotation: if switching BEFORE age 16, must pay back
            # the shortfall (you received credits up to age-16 level in first rotation,
            # but your current carbon is less than that)
            # If switching AT or AFTER age 16, no penalty - just start accruing
            if rotation >= 2 and a < params.carbon_credit_max_age:
                carbon_at_16 = C_age[params.carbon_credit_max_age]
                carbon_shortfall = max(0.0, carbon_at_16 - carbon_stock)
                switch_penalty += pc * carbon_shortfall
            
            R[s, ACTION_SWITCH_PERMANENT] = R_carbon + R_cost - switch_penalty
        else:
            # Already permanent - make identical to do nothing
            R[s, ACTION_SWITCH_PERMANENT] = R[s, ACTION_DO_NOTHING]
    
    return R


# =============================================================================
# 7. Transition Matrix Construction
# =============================================================================

def build_transition_matrix(
    params: ModelParameters,
    state_space: StateSpace,
    price_data: Dict
) -> np.ndarray:
    """
    Construct transition tensor Q[action, state, next_state].
    
    No forced harvest - age just increments (capped at A_max for state space).
    
    Returns:
        Q: Array of shape (N_actions, N_states, N_states)
    """
    Q = np.zeros((N_ACTIONS, state_space.N_states, state_space.N_states))
    
    Pc = price_data['Pc']
    Pt = price_data['Pt']
    
    for s in range(state_space.N_states):
        a, i_pc, i_pt, regime, rotation = state_space.state_to_tuple[s]
        
        # Compute next state components for each action
        for action in range(N_ACTIONS):
            
            if action == ACTION_DO_NOTHING:
                # Age advances (capped at A_max for state space bounds)
                a_next = min(a + 1, params.A_max)
                regime_next = regime
                rotation_next = rotation
                
            elif action == ACTION_HARVEST_REPLANT:
                # Reset age to 0
                a_next = 0
                regime_next = regime
                # Move to second rotation if in first
                rotation_next = 2
                
            elif action == ACTION_SWITCH_PERMANENT:
                if regime == 0:  # Switching from averaging to permanent
                    a_next = min(a + 1, params.A_max)
                    regime_next = 1  # Now permanent
                    rotation_next = rotation
                else:
                    # Already permanent - same as do nothing
                    a_next = min(a + 1, params.A_max)
                    regime_next = regime
                    rotation_next = rotation
            
            # Now loop over all possible next price states
            for j_pc in range(params.N_pc):
                for j_pt in range(params.N_pt):
                    # Joint price transition probability (independent)
                    p_price = Pc[i_pc, j_pc] * Pt[i_pt, j_pt]
                    
                    # Get next state index
                    next_tuple = (a_next, j_pc, j_pt, regime_next, rotation_next)
                    s_next = state_space.tuple_to_state[next_tuple]
                    
                    Q[action, s, s_next] += p_price
    
    # Verify rows sum to 1
    row_sums = Q.sum(axis=2)
    if not np.allclose(row_sums, 1.0):
        warnings.warn("Transition matrix rows don't sum to 1. Normalizing.")
        for action in range(N_ACTIONS):
            for s in range(state_space.N_states):
                if row_sums[action, s] > 0:
                    Q[action, s, :] /= row_sums[action, s]
    
    return Q


# =============================================================================
# 8. Solve DP Model
# =============================================================================

def solve_model(
    R: np.ndarray,
    Q: np.ndarray,
    beta: float,
    method: str = 'policy_iteration'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve using QuantEcon's DiscreteDP (infinite horizon).
    
    Returns:
        V: Value function
        sigma: Optimal policy
    """
    # DiscreteDP expects Q in shape (N_states, N_actions, N_states)
    # We have (N_actions, N_states, N_states), so transpose
    Q_transposed = np.transpose(Q, (1, 0, 2))
    
    ddp = DiscreteDP(R, Q_transposed, beta)
    
    if method == 'policy_iteration':
        results = ddp.solve(method='policy_iteration')
    else:
        results = ddp.solve(method='value_iteration')
    
    return results.v, results.sigma


# =============================================================================
# 9. Analysis and Visualization
# =============================================================================

def analyze_policy(
    sigma: np.ndarray,
    state_space: StateSpace,
    params: ModelParameters,
    price_data: Dict
) -> Dict:
    """
    Analyze the optimal policy to extract interpretable results.
    
    Returns:
        Dictionary with analysis results.
    """
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # Find harvest thresholds by age for average prices
    mid_pc = params.N_pc // 2
    mid_pt = params.N_pt // 2
    
    results = {
        'harvest_by_age_avg_rot1': {},
        'harvest_by_age_avg_rot2': {},
        'harvest_by_age_perm': {},
    }
    
    # Analyze for regime=0 (averaging), rotation=1
    print("\n=== Optimal Policy Analysis ===")
    print(f"\nCarbon credits earned up to age {params.carbon_credit_max_age}")
    print(f"Harvest penalty: ${params.harvest_penalty_per_m3}/m³")
    print("\nHarvest decisions by age (averaging regime, first rotation, mid prices):")
    print("-" * 60)
    
    first_harvest_age_rot1 = None
    for a in range(params.N_a):
        state_tuple = (a, mid_pc, mid_pt, 0, 1)  # averaging, first rotation
        s = state_space.tuple_to_state[state_tuple]
        action = sigma[s]
        
        action_name = ['Hold', 'Harvest', 'Switch'][action]
        results['harvest_by_age_avg_rot1'][a] = action
        
        if action == ACTION_HARVEST_REPLANT and first_harvest_age_rot1 is None:
            first_harvest_age_rot1 = a
            print(f"  Age {a:2d}: {action_name} ← First harvest age")
        elif action != ACTION_DO_NOTHING:
            print(f"  Age {a:2d}: {action_name}")
    
    if first_harvest_age_rot1 is not None:
        print(f"\n  → Optimal harvest age (first rotation): {first_harvest_age_rot1} years")
    
    # Analyze for regime=0 (averaging), rotation=2
    print("\nHarvest decisions (averaging regime, later rotations, mid prices):")
    print("-" * 60)
    
    first_harvest_age_rot2 = None
    for a in range(params.N_a):
        state_tuple = (a, mid_pc, mid_pt, 0, 2)  # averaging, later rotation
        s = state_space.tuple_to_state[state_tuple]
        action = sigma[s]
        
        results['harvest_by_age_avg_rot2'][a] = action
        
        if action == ACTION_HARVEST_REPLANT and first_harvest_age_rot2 is None:
            first_harvest_age_rot2 = a
            action_name = ['Hold', 'Harvest', 'Switch'][action]
            print(f"  Age {a:2d}: {action_name} ← First harvest age")
    
    if first_harvest_age_rot2 is not None:
        print(f"\n  → Optimal harvest age (later rotations): {first_harvest_age_rot2} years")
    
    # Analyze for regime=1 (permanent)
    print("\nHarvest decisions (permanent regime, mid prices):")
    print("-" * 60)
    
    first_harvest_age_perm = None
    for a in range(params.N_a):
        state_tuple = (a, mid_pc, mid_pt, 1, 1)  # permanent, first rotation
        s = state_space.tuple_to_state[state_tuple]
        action = sigma[s]
        
        results['harvest_by_age_perm'][a] = action
        
        if action == ACTION_HARVEST_REPLANT and first_harvest_age_perm is None:
            first_harvest_age_perm = a
            action_name = ['Hold', 'Harvest', 'Switch'][action]
            print(f"  Age {a:2d}: {action_name} ← First harvest age")
    
    if first_harvest_age_perm is not None:
        print(f"\n  → Optimal harvest age (permanent): {first_harvest_age_perm} years")
    else:
        print("\n  → No harvest optimal in permanent regime at mid prices")
    
    return results


def plot_harvest_regions(
    sigma: np.ndarray,
    state_space: StateSpace,
    params: ModelParameters,
    price_data: Dict,
    max_age_plot: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot harvest/switch decision regions in (age, price) space.
    """
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for actions
    cmap = plt.cm.colors.ListedColormap(['#2ecc71', '#e74c3c', '#3498db'])
    
    # Plot for each regime/rotation combination
    configs = [
        (0, 1, 'Averaging Regime, First Rotation'),
        (0, 2, 'Averaging Regime, Later Rotations'),
        (1, 1, 'Permanent Regime, First Rotation'),
        (1, 2, 'Permanent Regime, Later Rotations'),
    ]
    
    plot_ages = min(max_age_plot + 1, params.N_a)
    
    for ax_idx, (regime, rotation, title) in enumerate(configs):
        ax = axes.flatten()[ax_idx]
        
        # Use middle timber price, vary carbon price
        mid_pt = params.N_pt // 2
        
        # Build decision matrix (age x carbon price)
        decision_matrix = np.zeros((plot_ages, params.N_pc))
        
        for a in range(plot_ages):
            for i_pc in range(params.N_pc):
                state_tuple = (a, i_pc, mid_pt, regime, rotation)
                s = state_space.tuple_to_state[state_tuple]
                decision_matrix[a, i_pc] = sigma[s]
        
        # Plot
        im = ax.imshow(
            decision_matrix.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=0,
            vmax=2,
            extent=[0, plot_ages - 1, pc_grid[0], pc_grid[-1]]
        )
        
        ax.set_xlabel('Stand Age (years)', fontsize=11)
        ax.set_ylabel('Carbon Price ($/tCO₂)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0.33, 1, 1.67])
    cbar.ax.set_yticklabels(['Hold', 'Harvest', 'Switch'])
    
    plt.suptitle('Optimal Decisions by State\n(at median timber price)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def plot_harvest_regions_by_price(
    sigma: np.ndarray,
    state_space: StateSpace,
    params: ModelParameters,
    price_data: Dict,
    save_path: Optional[str] = None
):
    """
    Plot harvest/switch decision regions in (timber price, carbon price) space
    for specific age/regime/rotation combinations.
    
    4 panels:
    - First rotation averaging, age 16
    - First rotation averaging, age 30
    - Second rotation averaging, age 5
    - Permanent, age 30
    """
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for actions
    cmap = plt.cm.colors.ListedColormap(['#2ecc71', '#e74c3c', '#3498db'])
    
    # Panel configurations: (age, regime, rotation, title)
    configs = [
        (16, 0, 1, 'Averaging, 1st Rotation, Age 16'),
        (30, 0, 1, 'Averaging, 1st Rotation, Age 30'),
        (5, 0, 2, 'Averaging, 2nd+ Rotation, Age 5'),
        (30, 1, 1, 'Permanent, Age 30'),
    ]
    
    for ax_idx, (age, regime, rotation, title) in enumerate(configs):
        ax = axes.flatten()[ax_idx]
        
        # Clamp age to valid range
        a = min(age, params.N_a - 1)
        
        # Build decision matrix (timber price x carbon price)
        decision_matrix = np.zeros((params.N_pt, params.N_pc))
        
        for i_pt in range(params.N_pt):
            for i_pc in range(params.N_pc):
                state_tuple = (a, i_pc, i_pt, regime, rotation)
                s = state_space.tuple_to_state[state_tuple]
                decision_matrix[i_pt, i_pc] = sigma[s]
        
        # Plot
        im = ax.imshow(
            decision_matrix.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=0,
            vmax=2,
            extent=[pt_grid[0], pt_grid[-1], pc_grid[0], pc_grid[-1]]
        )
        
        ax.set_xlabel('Timber Price ($/m³)', fontsize=11)
        ax.set_ylabel('Carbon Price ($/tCO₂)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0.33, 1, 1.67])
    cbar.ax.set_yticklabels(['Hold', 'Harvest', 'Switch'])
    
    plt.suptitle('Optimal Decisions by Price State', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def plot_value_function(
    V: np.ndarray,
    state_space: StateSpace,
    params: ModelParameters,
    price_data: Dict,
    max_age_plot: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot value function by age for different price states.
    """
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for regime=0, rotation=1, varying carbon price
    mid_pt = params.N_pt // 2
    colors = plt.cm.viridis(np.linspace(0, 1, params.N_pc))
    
    plot_ages = min(max_age_plot + 1, params.N_a)
    
    for i_pc in range(params.N_pc):
        values_by_age = []
        for a in range(plot_ages):
            state_tuple = (a, i_pc, mid_pt, 0, 1)
            s = state_space.tuple_to_state[state_tuple]
            values_by_age.append(V[s])
        
        ax.plot(range(plot_ages), values_by_age, 
                color=colors[i_pc], 
                label=f'P_c = ${pc_grid[i_pc]:.0f}',
                linewidth=2)
    
    ax.set_xlabel('Stand Age (years)', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.set_title('Value Function by Age\n(Averaging Regime, First Rotation, Median Timber Price)',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Carbon Price', loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


# =============================================================================
# 10. Sanity Checks
# =============================================================================

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
        net_timber = timber_revenue - harvest_cost - params.replant_cost
        
        print(f"\n  Harvest payoff breakdown at age {a}:")
        print(f"    Timber revenue: ${timber_revenue:,.2f} ({volume:.1f} m³ × ${pt:.2f})")
        print(f"    Harvest cost:   -${harvest_cost:,.2f} ({volume:.1f} m³ × ${params.harvest_cost_per_m3 + params.harvest_penalty_per_m3:.2f})")
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
    price_data = build_price_grids(params_det)
    state_space = build_state_space(params_det)
    
    print(f"\n  State space size: {state_space.N_states}")
    
    R = build_reward_matrix(params_det, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm)
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
    R_no_carbon = build_reward_matrix(params_det, state_space, price_data, V_age, C_age, DeltaC_zero, DeltaC_zero)
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
    price_data_p = build_price_grids(params_penalty)
    state_space_p = build_state_space(params_penalty)
    
    R_p = build_reward_matrix(params_penalty, state_space_p, price_data_p, V_age_p, C_age_p, DeltaC_avg_p, DeltaC_perm_p)
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


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real Options Model for Forest Harvest Timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harvest_timing_model.py                    # Run full model
  python harvest_timing_model.py --sanity-checks    # Run only sanity checks
  python harvest_timing_model.py --no-plots         # Run without generating plots
        """
    )
    parser.add_argument(
        '--sanity-checks', '-s',
        action='store_true',
        help='Run only sanity checks with detailed output'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--harvest-penalty',
        type=float,
        default=0.0,
        help='Additional harvest penalty in $/m³ (default: 0)'
    )
    return parser.parse_args()


def main():
    """
    Main function to run the harvest timing model.
    """
    args = parse_args()
    
    # If only sanity checks requested, run them and exit
    if args.sanity_checks:
        print("=" * 70)
        print("REAL OPTIONS MODEL - SANITY CHECKS ONLY")
        print("=" * 70)
        run_sanity_checks()
        return None, None, None
    
    print("=" * 70)
    print("REAL OPTIONS MODEL FOR FOREST HARVEST TIMING")
    print("=" * 70)
    
    # Initialize parameters
    params = ModelParameters(harvest_penalty_per_m3=args.harvest_penalty)
    
    print("\n--- Model Parameters ---")
    print(f"  Discount rate: {params.discount_rate:.1%}")
    print(f"  Maximum age in state space: {params.A_max} years")
    print(f"  Carbon credits stop at age: {params.carbon_credit_max_age}")
    print(f"  Carbon price states: {params.N_pc}")
    print(f"  Timber price states: {params.N_pt}")
    print(f"  Harvest penalty: ${params.harvest_penalty_per_m3}/m³")
    print(f"  Carbon liability NPV factor: {params.carbon_liability_npv_factor():.3f}")
    
    # Build growth curves
    print("\n--- Computing Growth Curves ---")
    C_age = compute_carbon_curve(params)
    V_age = compute_volume_from_carbon(C_age, params)
    DeltaC_avg = compute_carbon_flows_averaging(C_age, params)
    DeltaC_perm = compute_carbon_flows_permanent(C_age)
    
    print(f"  Carbon at age 16: {C_age[16]:.1f} tCO₂/ha")
    print(f"  Carbon at age 30: {C_age[30]:.1f} tCO₂/ha")
    print(f"  Volume at age 16: {V_age[16]:.1f} m³/ha (= {params.timber_per_tonne_carbon} × carbon)")
    print(f"  Volume at age 30: {V_age[30]:.1f} m³/ha")
    print(f"  Averaging credits (ages 1-{params.carbon_credit_max_age}): {DeltaC_avg.sum():.1f} tCO₂/ha")
    print(f"  Permanent credits (all ages): continue indefinitely")
    
    # Build price grids
    print("\n--- Discretizing Price Processes ---")
    price_data = build_price_grids(params)
    
    print(f"  Carbon price range: ${price_data['pc_grid'][0]:.0f} - ${price_data['pc_grid'][-1]:.0f}")
    print(f"  Timber price range: ${price_data['pt_grid'][0]:.0f} - ${price_data['pt_grid'][-1]:.0f}")
    
    # Plot simulated price paths
    if not args.no_plots:
        print("\n--- Simulating Price Paths ---")
        try:
            plot_price_paths(params, n_paths=1000, n_periods=100, 
                           save_path='price_paths.png')
        except Exception as e:
            print(f"  Could not generate price paths plot: {e}")
    
    # Build state space
    print("\n--- Building State Space ---")
    state_space = build_state_space(params)
    
    print(f"  Total states: {state_space.N_states:,}")
    print(f"  Actions: {N_ACTIONS}")
    
    # Build reward matrix
    print("\n--- Building Reward Matrix ---")
    R = build_reward_matrix(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm)
    print(f"  Shape: {R.shape}")
    
    # Build transition matrix
    print("\n--- Building Transition Matrix ---")
    Q = build_transition_matrix(params, state_space, price_data)
    print(f"  Shape: {Q.shape}")
    
    # Solve
    print("\n--- Solving DP (Infinite Horizon) ---")
    V, sigma = solve_model(R, Q, params.beta)
    print("  ✓ Solution found")
    
    # Analyze policy
    results = analyze_policy(sigma, state_space, params, price_data)
    
    # Run sanity checks
    run_sanity_checks(params)
    
    # Plot results
    if not args.no_plots:
        print("\n--- Generating Visualizations ---")
        try:
            plot_harvest_regions(sigma, state_space, params, price_data, 
                                max_age_plot=50,
                                save_path='harvest_regions.png')
            plot_harvest_regions_by_price(sigma, state_space, params, price_data,
                                         save_path='harvest_regions_by_price.png')
            plot_value_function(V, state_space, params, price_data,
                               max_age_plot=50,
                               save_path='value_function.png')
        except Exception as e:
            print(f"  Could not generate plots (may need display): {e}")
    
    print("\n" + "=" * 70)
    print("MODEL EXECUTION COMPLETE")
    print("=" * 70)
    
    return V, sigma, results


if __name__ == "__main__":
    main()
