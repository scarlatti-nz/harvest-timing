"""
Real Options Model for Forest Harvest Timing

Infinite-horizon DP using QuantEcon's DiscreteDP. State includes age, prices, 
ETS regime, and rotation number. Carbon credits earned only in first rotation
(averaging) up to age 16. Permanent regime has carbon liability on harvest.

Regimes:
- averaging (0): Carbon credits up to age 16 in first rotation only, no carbon 
  liability on harvest
- permanent (1): Carbon credits indefinitely, carbon liability on harvest, 
  harvest penalty applies
- pre-2023 stock-change (2): Same as permanent but ignores harvest penalty. 
  Cannot be switched into from any other regime.
"""

import numpy as np
from quantecon.markov import tauchen, DiscreteDP
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional
import warnings
import argparse
import os
import time
import datetime


# =============================================================================
# 1. Configuration / Parameter Block
# =============================================================================

@dataclass
class ModelParameters:
    """Economic and biophysical parameters for the harvest timing model."""
    
    # Discounting
    discount_rate: float = 0.06
    
    # Age parameters
    A_max: int = 52  # Maximum age in state space 
    carbon_credit_max_age: int = 16  # Carbon credits stop at this age under averaging
    
    # Price grid sizes
    N_pt: int = 9   # Number of timber price states
    N_pc: int = 9   # Number of carbon price states
    
    # Price process parameters (AR(1) in logs)
    # Carbon price
    pc_0: float = 50.0      # Initial carbon price ($/tCO2)
    pc_mean: float = 50.0      # Mean carbon price ($/tCO2)
    pc_rho: float = 0.96        # AR(1) persistence
    pc_sigma: float = 0.2     # Volatility of log price
    
    # Timber price
    pt_mean: float = 150.0     # Mean timber price ($/m³)
    pt_rho: float = 0.63       # AR(1) persistence
    pt_sigma: float = 0.15     # Volatility of log price
    
    # Cost parameters
    harvest_cost_per_m3: float = 46.0   # $/m³
    harvest_cost_flat_per_ha: float = 12500.0  # Flat harvest cost per hectare
    replant_cost: float = 2000.0        # $ per hectare
    maintenance_cost: float = 50.0      # $ per year per hectare
    switch_cost: float = 1e9            # hack to disable switching for utility comparison
    # switch_cost: float = 10.0          # Admin cost to switch to permanent - $10/ha based on reality of per-forest fee of $700 and area of 70ha. Basically negligible.
    
    # Optional harvest penalty ($/m³) - $10 reflects current legislated penalty but can set aribtrarily high to completely disallow harvest of permanent regime
    harvest_penalty_per_m3: float = 10000.0
    
    # Permanent regime carbon liability parameters
    # On harvest: pay carbon_price * carbon_stock
    # 50% paid instantly, 50% paid in equal increments over 10 years
    carbon_liability_years: int = 10
    carbon_liability_instant_fraction: float = 0.5
    
    # Growth parameters
    C_max: float = 2221.0    # Maximum carbon stock (tCO2/ha)
    k_carbon: float = 0.0395  # Carbon accumulation rate
    raw_timber_m3_per_tonne_carbon: float = 1.2  # Timber volume (m³) per tonne of carbon
    recovery_rate: float = 0.5  # Proportion of total timber volume that is recoverable after harvest, based on ratio of total volume to residual volume in carbon tables
    timber_per_tonne_carbon: float = raw_timber_m3_per_tonne_carbon * recovery_rate  # Timber volume (m³) per tonne of carbon
    
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
        """Number of ETS regimes (averaging=0, permanent=1, pre-2023 stock-change=2)."""
        return 3
    
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


def compute_price_quality_factor(params: ModelParameters) -> np.ndarray:
    """
    Compute price scaling factor based on age (wood quality).
    
    f(a) = 
      0.5 + 0.03(a - 10)    10 <= a < 15
      0.65 + 0.03(a - 15)   15 <= a < 20
      0.8 + 0.04(a - 20)    20 <= a < 25
      0.99                  25 <= a <= 35
      0.99 - 0.01(a - 35)   35 < a <= 45
      0.89 - 0.02(a - 45)   a > 45
      
    Assume 0 for a < 10 (unmarketable).
    """
    ages = np.arange(params.N_a)
    factors = np.zeros_like(ages, dtype=float)
    
    for i, a in enumerate(ages):
        if a < 10:
            factors[i] = 0.0
        elif 10 <= a < 15:
            factors[i] = 0.5 + 0.03 * (a - 10)
        elif 15 <= a < 20:
            factors[i] = 0.65 + 0.03 * (a - 15)
        elif 20 <= a < 25:
            factors[i] = 0.8 + 0.04 * (a - 20)
        elif 25 <= a <= 35:
            factors[i] = 0.99
        elif 35 < a <= 45:
            factors[i] = 0.99 - 0.01 * (a - 35)
        else: # a > 45
            factors[i] = 0.89 - 0.02 * (a - 45)
            
    # Ensure non-negative
    factors = np.maximum(factors, 0.0)
    return factors


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
    
    # Intercept of log process: mu in AR(1) is mu = (1-rho)*E[y]
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
    p0: Optional[float] = None,
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
    
    # Initialize log prices at the unconditional mean or a given initial price if supplied
    if p0 is not None:
        log_prices = np.zeros((n_paths, n_periods))
        log_prices[:, 0] = np.log(p0)
    else:
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


# Plotting functions have been moved to plot_results.py



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
    DeltaC_perm: np.ndarray,
    price_quality_factor: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Construct reward matrix R[state, action].
    
    Carbon credits:
    - Averaging: first rotation only, up to age 16
    - Permanent: all rotations, all ages (indefinitely)
    
    Harvest costs:
    - Averaging: harvest_cost_per_m3 only (no penalty, any rotation)
    - Permanent: harvest_cost_per_m3 + harvest_penalty_per_m3 + carbon_liability (NPV)
    
    Note: When an averaging forest switches to permanent (Action 2), future
    harvests will incur the penalty since they'll be in the permanent regime.
    
    Returns:
        R: Array of shape (N_states, N_actions)
    """
    R = np.zeros((state_space.N_states, N_ACTIONS), dtype=np.float32)
    
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # NPV factor for carbon liability
    liability_npv_factor = params.carbon_liability_npv_factor()
    
    # Default quality factor is 1.0 if not provided
    if price_quality_factor is None:
        price_quality_factor = np.ones(params.N_a)
    
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
        else:  # Permanent (regime=1) or pre-2023 stock-change (regime=2)
            # Credits continue indefinitely (all ages, all rotations)
            delta_C = DeltaC_perm[a]
        
        # === Action 0: Do nothing ===
        R_carbon = pc * delta_C
        R_cost = -params.maintenance_cost
        R[s, ACTION_DO_NOTHING] = R_carbon + R_cost
        
        # === Action 1: Harvest and replant ===
        # Timber revenue minus harvest cost
        # Harvest penalty only applies to permanent regime (regime=1), not to
        # averaging (regime=0) or pre-2023 stock-change (regime=2)
        if regime == 1:  # Permanent
            harvest_cost_per_m3 = params.harvest_cost_per_m3 + params.harvest_penalty_per_m3
        else:  # Averaging or pre-2023 stock-change
            harvest_cost_per_m3 = params.harvest_cost_per_m3
        
        # Apply price quality scaling factor
        effective_pt = pt * price_quality_factor[a]
        
        R_timber = effective_pt * volume - harvest_cost_per_m3 * volume
        R_replant = -params.replant_cost
        R_harvest_flat = -params.harvest_cost_flat_per_ha  # Flat harvest cost per hectare
        
        # Carbon liability for permanent regimes (both permanent and pre-2023 stock-change)
        if regime >= 1:  # Permanent or pre-2023 stock-change
            # Must pay carbon_price * carbon_stock on harvest
            # 50% instant, 50% over 10 years (use NPV)
            carbon_liability = pc * carbon_stock * liability_npv_factor
            R[s, ACTION_HARVEST_REPLANT] = R_timber + R_replant + R_harvest_flat - carbon_liability
        else:
            # Averaging: no carbon liability on harvest
            R[s, ACTION_HARVEST_REPLANT] = R_timber + R_replant + R_harvest_flat
        
        # === Action 2: Switch to permanent ===
        # Only meaningful when in averaging (regime=0). Cannot switch INTO
        # pre-2023 stock-change (regime=2) from any regime.
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
            # Already permanent or pre-2023 stock change - make identical to do nothing
            R[s, ACTION_SWITCH_PERMANENT] = R[s, ACTION_DO_NOTHING]
    
    return R


def save_reward_matrix_csv(
    R: np.ndarray,
    state_space: StateSpace,
    price_data: Dict,
    filename: str = 'reward_matrix.csv'
) -> None:
    """
    Save the reward matrix to a labeled CSV file for easier interpretation.
    
    Columns:
        state_idx: State index (row in R matrix)
        age: Forest age (years)
        carbon_price: Carbon price ($/tCO2) 
        timber_price: Timber price ($/m³)
        regime: 'averaging' or 'permanent'
        rotation: 'first' or 'later'
        R_do_nothing: Reward for action 0 (continue growing)
        R_harvest_replant: Reward for action 1 (harvest and replant)
        R_switch_permanent: Reward for action 2 (switch to permanent regime)
    
    Args:
        R: Reward matrix of shape (N_states, N_actions)
        state_space: StateSpace object with state mappings
        price_data: Dictionary with price grids
        filename: Output filename
    """
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    regime_names = {0: 'averaging', 1: 'permanent', 2: 'pre-2023 stock change'}
    rotation_names = {1: 'first', 2: 'later'}
    
    with open(filename, 'w') as f:
        # Header
        f.write('state_idx,age,carbon_price,timber_price,regime,rotation,'
                'R_do_nothing,R_harvest_replant,R_switch_permanent\n')
        
        # Data rows
        for s in range(state_space.N_states):
            age, i_pc, i_pt, regime, rotation = state_space.state_to_tuple[s]
            
            pc = pc_grid[i_pc]
            pt = pt_grid[i_pt]
            regime_str = regime_names[regime]
            rotation_str = rotation_names[rotation]
            
            f.write(f'{s},{age},{pc:.2f},{pt:.2f},{regime_str},{rotation_str},'
                    f'{R[s, 0]:.2f},{R[s, 1]:.2f},{R[s, 2]:.2f}\n')


# =============================================================================
# 7. Transition Matrix Construction
# =============================================================================

def build_transition_matrix(
    params: ModelParameters,
    state_space: StateSpace,
    price_data: Dict
) -> Tuple["sp.csr_matrix", np.ndarray, np.ndarray]:
    """
    Construct sparse transition matrix in state-action-pairs form for QuantEcon.
    
    No forced harvest - age just increments (capped at A_max for state space).
    
    Returns:
        Q_sa: Sparse CSR matrix of shape (L, N_states) where L = N_states * N_ACTIONS
        s_indices: Array of shape (L,) giving the state index for each SA row
        a_indices: Array of shape (L,) giving the action index for each SA row
    """
    import scipy.sparse as sp

    n = state_space.N_states
    m = N_ACTIONS
    L = n * m
    
    Pc = price_data['Pc']
    Pt = price_data['Pt']

    # Enumerate all state-action pairs (product formulation) as SA pairs
    s_indices = np.repeat(np.arange(n, dtype=np.int32), m)
    a_indices = np.tile(np.arange(m, dtype=np.int8), n)

    # Each (state, action) row has exactly N_pc * N_pt non-zeros (independent price transitions)
    nnz_per_row = params.N_pc * params.N_pt
    nnz = L * nnz_per_row
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)

    # Fast arithmetic mapping consistent with build_state_space enumeration order:
    # for a in ages:
    #   for i_pc:
    #     for i_pt:
    #       for regime:
    #         for rotation in {1,2}:
    #           s += 1
    def _state_index(a: int, i_pc: int, i_pt: int, regime: int, rotation: int) -> int:
        return (
            (((a * params.N_pc + i_pc) * params.N_pt + i_pt) * params.N_regimes + regime)
            * params.N_rotations
            + (rotation - 1)
        )
    
    k = 0
    for s in range(n):
        a, i_pc, i_pt, regime, rotation = state_space.state_to_tuple[s]
        
        # Compute next state components for each action
        for action in range(m):
            row = s * m + action
            
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
                    # Already permanent or pre-2023 stock-change - same as do nothing
                    a_next = min(a + 1, params.A_max)
                    regime_next = regime  # Stay in current regime
                    rotation_next = rotation
            
            # Now loop over all possible next price states
            for j_pc in range(params.N_pc):
                for j_pt in range(params.N_pt):
                    # Joint price transition probability (independent)
                    p_price = Pc[i_pc, j_pc] * Pt[i_pt, j_pt]
                    
                    # Get next state index (fast arithmetic mapping)
                    s_next = _state_index(a_next, j_pc, j_pt, regime_next, rotation_next)

                    rows[k] = row
                    cols[k] = s_next
                    data[k] = p_price
                    k += 1

    if k != nnz:
        raise RuntimeError(f"Internal error building sparse Q: expected nnz={nnz}, got k={k}")

    Q_sa = sp.csr_matrix((data, (rows, cols)), shape=(L, n), dtype=np.float32)

    # Quick validation: each row should sum to 1 (within float tolerance)
    row_sums = np.asarray(Q_sa.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=1e-5):
        warnings.warn("Sparse transition matrix rows don't sum to 1. Normalizing.")
        inv = np.reciprocal(np.maximum(row_sums, 1e-12)).astype(np.float32)
        Q_sa = sp.diags(inv, offsets=0, format="csr") @ Q_sa

    return Q_sa, s_indices, a_indices


# =============================================================================
# 8. Solve DP Model
# =============================================================================

def solve_model(
    R: np.ndarray,
    Q,
    beta: float,
    method: str = 'policy_iteration',
    s_indices: Optional[np.ndarray] = None,
    a_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve using QuantEcon's DiscreteDP (infinite horizon).
    
    Returns:
        V: Value function
        sigma: Optimal policy
    """
    # Use sparse state-action-pairs formulation if indices are provided.
    # This avoids allocating the dense (n, m, n) transition tensor.
    if s_indices is not None and a_indices is not None:
        # Convert (n, m) reward matrix to SA reward vector in the same ordering:
        # row = s * N_ACTIONS + a
        R_sa = np.asarray(R, dtype=np.float32).reshape(-1)
        ddp = DiscreteDP(R_sa, Q, beta, s_indices, a_indices)
    else:
        # Dense product formulation (only feasible for small problems)
        Q_transposed = np.transpose(Q, (1, 0, 2))
        ddp = DiscreteDP(R, Q_transposed, beta)

    # Respect requested method
    if method in ('value_iteration', 'vi'):
        results = ddp.solve(method='value_iteration')
    else:
        results = ddp.solve(method='policy_iteration')
    
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
        'harvest_by_age_perm_no_penalty': {},
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
    
    # Analyze for regime=2 (pre-2023 stock change)
    print("\nHarvest decisions (pre-2023 stock change regime, mid prices):")
    print("-" * 60)
    
    first_harvest_age_perm_no_penalty = None
    for a in range(params.N_a):
        state_tuple = (a, mid_pc, mid_pt, 2, 1)  # pre-2023 stock change, first rotation
        s = state_space.tuple_to_state[state_tuple]
        action = sigma[s]
        
        results['harvest_by_age_perm_no_penalty'][a] = action
        
        if action == ACTION_HARVEST_REPLANT and first_harvest_age_perm_no_penalty is None:
            first_harvest_age_perm_no_penalty = a
            action_name = ['Hold', 'Harvest', 'Switch'][action]
            print(f"  Age {a:2d}: {action_name} ← First harvest age")
    
    if first_harvest_age_perm_no_penalty is not None:
        print(f"\n  → Optimal harvest age (pre-2023 stock change): {first_harvest_age_perm_no_penalty} years")
    else:
        print("\n  → No harvest optimal in pre-2023 stock change regime at mid prices")
    
    return results


# Plotting functions have been moved to plot_results.py


def simulate_single_trajectory(
    params: ModelParameters,
    state_space: StateSpace,
    price_data: Dict,
    R: np.ndarray,
    Q,
    V: np.ndarray,
    sigma: np.ndarray,
    C_age: np.ndarray,
    n_years: int = 50,
    seed: Optional[int] = 42,
    carbon_prices: Optional[np.ndarray] = None,
    timber_prices: Optional[np.ndarray] = None,
    initial_age: int = 1,
    initial_regime: int = 0,
    initial_rotation: int = 1
) -> Dict:
    """
    Simulate a single trajectory of optimal policy execution with random prices.
    
    Tracks:
    - Prices (Carbon, Timber)
    - State (Age, Regime, Rotation)
    - Action Values (Q-factors) for Hold, Harvest, Switch
    - Optimal/Chosen Action
    - Realized Net Revenue (Immediate Reward)
    - Carbon Stock
    
    Returns:
        Dictionary containing time-series data for plotting.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Simulate continuous price paths if not provided
    # Use a new seed for prices if not provided, to ensure variety if called multiple times
    price_seed = seed if seed is not None else np.random.randint(0, 10000)
    
    if carbon_prices is None:
        carbon_prices = simulate_price_paths(
            params.pc_mean, params.pc_rho, params.pc_sigma,
            n_paths=1, n_periods=n_years + 1, seed=price_seed, 
            p0=params.pc_0
        ).flatten()
    
    if timber_prices is None:
        timber_prices = simulate_price_paths(
            params.pt_mean, params.pt_rho, params.pt_sigma,
            n_paths=1, n_periods=n_years + 1, seed=price_seed + 1
        ).flatten()
    
    pc_grid = price_data['pc_grid']
    pt_grid = price_data['pt_grid']
    
    # 2. Initialize storage
    data = {
        'years': np.arange(n_years + 1),
        'carbon_price': carbon_prices,
        'timber_price': timber_prices,
        'age': np.zeros(n_years + 1, dtype=int),
        'regime': np.zeros(n_years + 1, dtype=int),
        'rotation': np.zeros(n_years + 1, dtype=int),
        'action': np.zeros(n_years + 1, dtype=int),
        'q_hold': np.zeros(n_years + 1),
        'q_harvest': np.zeros(n_years + 1),
        'q_switch': np.zeros(n_years + 1),
        'net_revenue': np.zeros(n_years + 1),
        'carbon_stock': np.zeros(n_years + 1),
    }
    
    # 3. Initial state
    current_age = initial_age
    current_regime = initial_regime
    current_rotation = initial_rotation
    
    # Pre-compute expected continuation values vector E[V(s')] = sum_s' P(s'|s,a) V(s')
    # If Q is sparse SA-form (L x N_states), expected continuation is:
    #   expected_v = Q_sa[row] @ V, where row = s * N_ACTIONS + a.
    
    for t in range(n_years + 1):
        # Record state
        data['age'][t] = current_age
        data['regime'][t] = current_regime
        data['rotation'][t] = current_rotation
        data['carbon_stock'][t] = C_age[current_age]
        
        # Find nearest price grid indices
        i_pc = (np.abs(pc_grid - carbon_prices[t])).argmin()
        i_pt = (np.abs(pt_grid - timber_prices[t])).argmin()
        
        # Identify current state index
        state_tuple = (current_age, i_pc, i_pt, current_regime, current_rotation)
        s = state_space.tuple_to_state[state_tuple]
        
        # Compute Q-values (Action Values) for this state
        # Q(s, a) = R(s, a) + beta * sum(P(s'|s,a) * V(s'))
        q_values = np.zeros(3)
        for a in range(3):
            # Immediate reward
            r_sa = R[s, a]
            
            # Expected continuation value
            expected_v = None
            try:
                import scipy.sparse as sp
                if sp.issparse(Q):
                    row = s * N_ACTIONS + a
                    expected_v = float(Q[row].dot(V))
            except Exception:
                expected_v = None

            if expected_v is None:
                # Fallback for dense Q[action, s, :]
                expected_v = float(np.dot(Q[a, s, :], V))
            
            q_values[a] = r_sa + params.beta * expected_v
        
        data['q_hold'][t] = q_values[ACTION_DO_NOTHING]
        data['q_harvest'][t] = q_values[ACTION_HARVEST_REPLANT]
        data['q_switch'][t] = q_values[ACTION_SWITCH_PERMANENT]
        
        # Get optimal action
        # We could use argmax(q_values), but let's trust sigma[s]
        # They should be identical (modulo float precision tie-breaking)
        opt_action = sigma[s]
        data['action'][t] = opt_action
        
        # Record Net Revenue (Immediate Reward of chosen action)
        data['net_revenue'][t] = R[s, opt_action]
        
        # Update structural state for next period
        if opt_action == ACTION_DO_NOTHING:
            current_age = min(current_age + 1, params.A_max)
            # regime, rotation unchanged
        elif opt_action == ACTION_HARVEST_REPLANT:
            current_age = 0 # Replant -> age 0
            # regime unchanged
            if current_rotation == 1:
                current_rotation = 2
            # else stay 2
        elif opt_action == ACTION_SWITCH_PERMANENT:
            current_age = min(current_age + 1, params.A_max)
            if current_regime == 0:
                current_regime = 1 # Become permanent
            # rotation unchanged
        
    return data


# plot_simulation_trajectory and plot_value_function have been moved to plot_results.py


# =============================================================================
# 10. Sanity Checks
# =============================================================================
# Sanity checks have been moved to sanity_checks.py
# Import run_sanity_checks from that module if needed


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
  python harvest_timing_model.py --price-paths-only # Generate price paths and exit
  python harvest_timing_model.py --grid-size 20     # Use 20 price states for each price
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
        '--price-paths-only',
        action='store_true',
        help='Generate price paths plot and exit'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=7,
        help='Number of price states for both timber and carbon prices (default: 7)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='temp',
        help='Directory to save model results pickle file (default: temp)'
    )
    return parser.parse_args()


def save_model_parameters_to_txt(params: ModelParameters, directory: str):
    """
    Save model parameters to a text file with a timestamp in the specified directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_parameters_{timestamp}.txt"
    filepath = os.path.join('outputs',directory, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Model Parameters - Saved at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Get dictionary of parameters
        param_dict = asdict(params)
        
        # Sort keys for readability
        for key in sorted(param_dict.keys()):
            f.write(f"{key:35}: {param_dict[key]}\n")
            
    return filepath


def main(args=None, params=None):
    """
    Main function to run the harvest timing model.
    """
    if args is None:
        args = parse_args()
    
    # If only sanity checks requested, run them and exit
    if args.sanity_checks:
        from sanity_checks import run_sanity_checks
        print("=" * 70)
        print("REAL OPTIONS MODEL - SANITY CHECKS ONLY")
        print("=" * 70)
        run_sanity_checks()
        return None, None, None
    
    print("=" * 70)
    print("REAL OPTIONS MODEL FOR FOREST HARVEST TIMING")
    print("=" * 70)
    
    # Initialize parameters if not provided
    if params is None:
        params = ModelParameters(
            N_pt=args.grid_size,
            N_pc=args.grid_size
        )
    
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
    price_quality_factor = compute_price_quality_factor(params)
    
    print(f"  Carbon at age 16: {C_age[16]:.1f} tCO₂/ha")
    print(f"  Carbon at age 30: {C_age[30]:.1f} tCO₂/ha")
    print(f"  Volume at age 16: {V_age[16]:.1f} m³/ha (= {params.timber_per_tonne_carbon} × carbon)")
    print(f"  Volume at age 30: {V_age[30]:.1f} m³/ha")
    print(f"  Price quality factor at age 10: {price_quality_factor[10]:.2f}")
    print(f"  Price quality factor at age 25: {price_quality_factor[25]:.2f}")
    print(f"  Price quality factor at age 40: {price_quality_factor[40]:.2f}")
    print(f"  Averaging credits (ages 1-{params.carbon_credit_max_age}): {DeltaC_avg.sum():.1f} tCO₂/ha")
    print(f"  Permanent credits (all ages): continue indefinitely")
    
    # Build price grids
    print("\n--- Discretizing Price Processes ---")
    price_data = build_price_grids(params)
    
    print(f"  Carbon price range: ${price_data['pc_grid'][0]:.0f} - ${price_data['pc_grid'][-1]:.0f}")
    print(f"  Timber price range: ${price_data['pt_grid'][0]:.0f} - ${price_data['pt_grid'][-1]:.0f}")
    
    # Plot simulated price paths (always if --price-paths-only, otherwise if not --no-plots)
    if args.price_paths_only:
        print("\n--- Simulating Price Paths ---")
        try:
            os.makedirs('plots', exist_ok=True)
            # Simulate paths just for the plot (this function is now moved/removed, but we need the logic or just skip)
            # Actually, the user asked to remove plotting logic. 
            # If price-paths-only is requested, we should probably just exit or use the new script?
            # Let's assume we just save the pickle and let the other script handle it.
            pass
        except Exception as e:
            print(f"  Could not generate price paths plot: {e}")
    
    # Exit early if only price paths requested (and we're not doing them here anymore)
    if args.price_paths_only:
        print("\n" + "=" * 70)
        print("PRICE PATHS ONLY REQUESTED - PLEASE RUN plot_results.py")
        print("=" * 70)
        return None, None, None
    
    # Build state space
    print("\n--- Building State Space ---")
    state_space = build_state_space(params)
    # np.savetxt('state_space.csv', state_space, delimiter=',')
    
    print(f"  Total states: {state_space.N_states:,}")
    print(f"  Actions: {N_ACTIONS}")
    
    # Build reward matrix
    print("\n--- Building Reward Matrix ---")
    R = build_reward_matrix(params, state_space, price_data, V_age, C_age, DeltaC_avg, DeltaC_perm, price_quality_factor)
    
    # Save labeled reward matrix CSV
    save_reward_matrix_csv(R, state_space, price_data, 'reward_matrix.csv')
    print(f"  Saved labeled reward matrix to reward_matrix.csv")
    print(f"  Shape: {R.shape}")
    
    # Build transition matrix
    print("\n--- Building Transition Matrix ---")
    Q_sa, s_indices, a_indices = build_transition_matrix(params, state_space, price_data)
    print(f"  Shape (SA-form): {Q_sa.shape}")
    
    # Solve
    print("\n--- Solving DP (Infinite Horizon) ---")
    start_time = time.time()
    V, sigma = solve_model(R, Q_sa, params.beta, method='policy_iteration', s_indices=s_indices, a_indices=a_indices)
    end_time = time.time()
    print(f"  Time taken: {end_time - start_time:.2f} seconds")
    print("  ✓ Solution found")
    
    # Analyze policy
    results = analyze_policy(sigma, state_space, params, price_data)
    
    # Run sanity checks (optional, can be run separately via sanity_checks.py)
    # Uncomment the line below if you want sanity checks to run automatically
    # from sanity_checks import run_sanity_checks
    # run_sanity_checks(params)
    
    # Simulate single trajectory (analysis)
    print("\n--- Simulating Optimal Trajectory ---")
    sim_data = simulate_single_trajectory(
        params, state_space, price_data, R, Q_sa, V, sigma, C_age,
        n_years=50, seed=42
    )

    # Save results to pickle
    print("\n--- Saving Results ---")
    import pickle
    os.makedirs('outputs/'+ args.temp_dir, exist_ok=True)
    pickle_path = os.path.join('outputs', args.temp_dir, 'model_results.pkl')
    
    # Save parameters to text file
    params_txt_path = save_model_parameters_to_txt(params, args.temp_dir)
    print(f"  Parameters saved to {params_txt_path}")
    
    results_data = {
        'params': params,
        'state_space': state_space,
        'price_data': price_data,
        'V': V,
        'sigma': sigma,
        'sim_data': sim_data,
        # 'C_age': C_age # Not strictly needed if params and growth functions are available, but helpful
    }
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"  Results saved to {pickle_path}")
    print("  Run 'python plot_results.py' to generate plots.")
    
    print("\n" + "=" * 70)
    print("MODEL EXECUTION COMPLETE")
    print("=" * 70)
    
    return V, sigma, results


if __name__ == "__main__":
    main()
