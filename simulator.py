"""Path simulation engine supporting Bernoulli and Poisson execution models."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from params import ModelParams
from strategies import Quote, compute_fill_probability, constant_spread_quote


@dataclass
class SimulationState:
    """State of a single simulation path.
    
    Attributes:
        S: Current stock price
        inventory: Current inventory position
        cash: Current cash position
        pnl: Running profit and loss
        n_trades: Number of trades executed
        n_bid_fills: Number of bid fills
        n_ask_fills: Number of ask fills
    """
    S: float
    inventory: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    n_trades: int = 0
    n_bid_fills: int = 0
    n_ask_fills: int = 0
    
    def mark_to_market(self, S_current: float) -> float:
        """Compute current PnL including inventory mark-to-market."""
        return self.pnl + self.inventory * S_current


@dataclass
class SimulationResult:
    """Results from a simulation run.
    
    Attributes:
        paths: Array of price paths (n_paths, n_steps+1)
        inventories: Array of inventory paths (n_paths, n_steps+1)
        cash_paths: Array of cash paths (n_paths, n_steps+1)
        pnl_paths: Array of PnL paths (n_paths, n_steps+1)
        trade_counts: Array of trade counts per path
        final_pnl: Array of final PnL values
        final_inventory: Array of final inventory values
        params: Model parameters used
        execution_model: Execution model used
    """
    paths: np.ndarray
    inventories: np.ndarray
    cash_paths: np.ndarray
    pnl_paths: np.ndarray
    trade_counts: np.ndarray
    final_pnl: np.ndarray
    final_inventory: np.ndarray
    params: ModelParams
    execution_model: str
    
    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics of simulation results."""
        return {
            'mean_final_pnl': float(np.mean(self.final_pnl)),
            'std_final_pnl': float(np.std(self.final_pnl)),
            'mean_final_inventory': float(np.mean(self.final_inventory)),
            'std_final_inventory': float(np.std(self.final_inventory)),
            'mean_trades': float(np.mean(self.trade_counts)),
            'total_trades': int(np.sum(self.trade_counts)),
            'sharpe_ratio': float(np.mean(self.final_pnl) / (np.std(self.final_pnl) + 1e-10)),
        }


def _generate_price_paths(params: ModelParams, 
                          rng: np.random.Generator,
                          n_paths: int) -> np.ndarray:
    """Generate price paths using binomial model.
    
    Price updates: S += +/- sigma * sqrt(dt)
    
    Args:
        params: Model parameters
        rng: Random number generator
        n_paths: Number of paths to generate
        
    Returns:
        Price paths array of shape (n_paths, n_steps+1)
    """
    dt = params.dt
    sqrt_dt = params.sqrt_dt
    
    # Generate random signs for price moves (+1 or -1)
    moves = rng.choice([-1, 1], size=(n_paths, params.n_steps))
    
    # Compute price increments
    increments = params.sigma * sqrt_dt * moves
    
    # Build price paths
    paths = np.zeros((n_paths, params.n_steps + 1))
    paths[:, 0] = params.S0
    paths[:, 1:] = params.S0 + np.cumsum(increments, axis=1)
    
    return paths


def _simulate_path_bernoulli(S_path: np.ndarray,
                              params: ModelParams,
                              rng: np.random.Generator,
                              quote_fn: Callable = constant_spread_quote) -> SimulationState:
    """Simulate a single path using Bernoulli execution model.
    
    In Bernoulli model, fills occur with probability lambda = A * exp(-k * delta)
    at each time step.
    
    Args:
        S_path: Price path array
        params: Model parameters
        rng: Random number generator
        quote_fn: Quote strategy function
        
    Returns:
        Final simulation state
    """
    state = SimulationState(S=S_path[0])
    
    for t_idx in range(len(S_path) - 1):
        S = S_path[t_idx]
        t = t_idx * params.dt
        
        # Get quote from strategy
        quote = quote_fn(params, S, state.inventory, t)
        
        # Compute fill probabilities
        p_bid = compute_fill_probability(quote.delta_b, params)
        p_ask = compute_fill_probability(quote.delta_a, params)
        
        # Clamp probabilities to [0, 1]
        p_bid = min(max(p_bid, 0.0), 1.0)
        p_ask = min(max(p_ask, 0.0), 1.0)
        
        # Sample Bernoulli fills
        bid_fill = rng.random() < p_bid
        ask_fill = rng.random() < p_ask
        
        # Process fills
        if bid_fill:
            # Buy at bid: S - delta_b
            exec_price = S - quote.delta_b
            state.inventory += 1
            state.cash -= exec_price
            state.n_trades += 1
            state.n_bid_fills += 1
            
        if ask_fill:
            # Sell at ask: S + delta_a
            exec_price = S + quote.delta_a
            state.inventory -= 1
            state.cash += exec_price
            state.n_trades += 1
            state.n_ask_fills += 1
        
        # Update price
        state.S = S_path[t_idx + 1]
        
        # Update PnL (mark to market)
        state.pnl = state.cash + state.inventory * state.S
    
    return state


def _simulate_path_poisson(S_path: np.ndarray,
                            params: ModelParams,
                            rng: np.random.Generator,
                            quote_fn: Callable = constant_spread_quote) -> SimulationState:
    """Simulate a single path using Poisson execution model.
    
    In Poisson model, fills occur with intensity lambda = A * exp(-k * delta).
    Number of fills in dt is Poisson distributed with mean lambda * dt.
    
    Args:
        S_path: Price path array
        params: Model parameters
        rng: Random number generator
        quote_fn: Quote strategy function
        
    Returns:
        Final simulation state
    """
    state = SimulationState(S=S_path[0])
    dt = params.dt
    
    for t_idx in range(len(S_path) - 1):
        S = S_path[t_idx]
        t = t_idx * params.dt
        
        # Get quote from strategy
        quote = quote_fn(params, S, state.inventory, t)
        
        # Compute fill intensities
        lambda_bid = compute_fill_probability(quote.delta_b, params)
        lambda_ask = compute_fill_probability(quote.delta_a, params)
        
        # Expected number of fills in this time step
        expected_bid = lambda_bid * dt
        expected_ask = lambda_ask * dt
        
        # Sample Poisson number of fills
        n_bid_fills = rng.poisson(expected_bid)
        n_ask_fills = rng.poisson(expected_ask)
        
        # Process bid fills (buys)
        if n_bid_fills > 0:
            exec_price = S - quote.delta_b
            state.inventory += n_bid_fills
            state.cash -= n_bid_fills * exec_price
            state.n_trades += n_bid_fills
            state.n_bid_fills += n_bid_fills
        
        # Process ask fills (sells)
        if n_ask_fills > 0:
            exec_price = S + quote.delta_a
            state.inventory -= n_ask_fills
            state.cash += n_ask_fills * exec_price
            state.n_trades += n_ask_fills
            state.n_ask_fills += n_ask_fills
        
        # Update price
        state.S = S_path[t_idx + 1]
        
        # Update PnL (mark to market)
        state.pnl = state.cash + state.inventory * state.S
    
    return state


def simulate_path(params: ModelParams,
                  execution_model: str = 'bernoulli',
                  quote_fn: Optional[Callable] = None,
                  seed: Optional[int] = None,
                  use_common_random: bool = False) -> SimulationResult:
    """Simulate multiple paths with specified execution model.
    
    Args:
        params: Model parameters
        execution_model: 'bernoulli' or 'poisson'
        quote_fn: Quote strategy function (default: constant_spread_quote)
        seed: Random seed for reproducibility
        use_common_random: If True, use same random numbers for price paths
                          across different parameter sets (useful for comparison)
        
    Returns:
        SimulationResult with all paths and statistics
        
    Example:
        >>> params = ModelParams(S0=100, T=1.0, n_steps=252, sigma=0.2)
        >>> result = simulate_path(params, execution_model='bernoulli', n_paths=1000)
        >>> print(result.summary_stats())
    """
    if execution_model not in ('bernoulli', 'poisson'):
        raise ValueError(f"execution_model must be 'bernoulli' or 'poisson', "
                        f"got '{execution_model}'")
    
    if quote_fn is None:
        quote_fn = constant_spread_quote
    
    # Set up random number generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    n_paths = params.n_paths
    
    # Generate price paths
    price_paths = _generate_price_paths(params, rng, n_paths)
    
    # Pre-allocate result arrays
    inventories = np.zeros((n_paths, params.n_steps + 1))
    cash_paths = np.zeros((n_paths, params.n_steps + 1))
    pnl_paths = np.zeros((n_paths, params.n_steps + 1))
    trade_counts = np.zeros(n_paths, dtype=np.int32)
    
    # Select simulation function based on execution model
    simulate_fn = (_simulate_path_bernoulli if execution_model == 'bernoulli' 
                   else _simulate_path_poisson)
    
    # Simulate each path
    for i in range(n_paths):
        state = simulate_fn(price_paths[i], params, rng, quote_fn)
        
        # For simplicity, store final state replicated across time
        # (Could be extended to store full path history)
        inventories[i, :] = state.inventory
        cash_paths[i, :] = state.cash
        pnl_paths[i, :] = state.pnl
        trade_counts[i] = state.n_trades
    
    # Compute final values
    final_pnl = pnl_paths[:, -1].copy()
    final_inventory = inventories[:, -1].copy()
    
    return SimulationResult(
        paths=price_paths,
        inventories=inventories,
        cash_paths=cash_paths,
        pnl_paths=pnl_paths,
        trade_counts=trade_counts,
        final_pnl=final_pnl,
        final_inventory=final_inventory,
        params=params,
        execution_model=execution_model,
    )


# Optional numba-accelerated version for large-scale simulations
try:
    from numba import njit, prange
    
    @njit
    def _simulate_path_bernoulli_numba(S_path: np.ndarray,
                                        A: float, k: float,
                                        delta_max: float,
                                        dt: float) -> Tuple[float, float, float, int, int, int]:
        """Numba-accelerated Bernoulli simulation for a single path.
        
        Returns:
            Tuple of (final_inventory, final_cash, final_pnl, n_trades, n_bid_fills, n_ask_fills)
        """
        inventory = 0.0
        cash = 0.0
        n_trades = 0
        n_bid_fills = 0
        n_ask_fills = 0
        
        for t_idx in range(len(S_path) - 1):
            S = S_path[t_idx]
            
            # Constant spread quote (simplified for numba)
            delta = delta_max / 2
            
            # Fill probabilities
            p_bid = A * np.exp(-k * delta)
            p_ask = A * np.exp(-k * delta)
            p_bid = min(max(p_bid, 0.0), 1.0)
            p_ask = min(max(p_ask, 0.0), 1.0)
            
            # Bernoulli fills
            if np.random.random() < p_bid:
                exec_price = S - delta
                inventory += 1
                cash -= exec_price
                n_trades += 1
                n_bid_fills += 1
            
            if np.random.random() < p_ask:
                exec_price = S + delta
                inventory -= 1
                cash += exec_price
                n_trades += 1
                n_ask_fills += 1
        
        S_final = S_path[-1]
        pnl = cash + inventory * S_final
        
        return inventory, cash, pnl, n_trades, n_bid_fills, n_ask_fills
    
    @njit(parallel=True)
    def simulate_paths_numba(price_paths: np.ndarray,
                             A: float, k: float, delta_max: float, dt: float,
                             n_paths: int, n_steps: int):
        """Numba-accelerated batch simulation."""
        inventories = np.zeros(n_paths)
        cash_paths = np.zeros(n_paths)
        pnl_paths = np.zeros(n_paths)
        trade_counts = np.zeros(n_paths, dtype=np.int32)
        
        for i in prange(n_paths):
            inv, cash, pnl, trades, _, _ = _simulate_path_bernoulli_numba(
                price_paths[i], A, k, delta_max, dt
            )
            inventories[i] = inv
            cash_paths[i] = cash
            pnl_paths[i] = pnl
            trade_counts[i] = trades
        
        return inventories, cash_paths, pnl_paths, trade_counts
    
    NUMBA_AVAILABLE = True
    
except ImportError:
    NUMBA_AVAILABLE = False


def simulate_path_fast(params: ModelParams,
                       execution_model: str = 'bernoulli',
                       seed: Optional[int] = None) -> SimulationResult:
    """Fast simulation using numba acceleration if available.
    
    Falls back to pure numpy if numba is not installed.
    
    Args:
        params: Model parameters
        execution_model: 'bernoulli' or 'poisson'
        seed: Random seed
        
    Returns:
        SimulationResult
    """
    if NUMBA_AVAILABLE and execution_model == 'bernoulli':
        # Use numba-accelerated version
        rng = np.random.default_rng(seed)
        price_paths = _generate_price_paths(params, rng, params.n_paths)
        
        inventories, cash_paths, pnl_paths, trade_counts = simulate_paths_numba(
            price_paths, params.A, params.k, params.delta_max, params.dt,
            params.n_paths, params.n_steps
        )
        
        return SimulationResult(
            paths=price_paths,
            inventories=np.tile(inventories, (params.n_steps + 1, 1)).T,
            cash_paths=np.tile(cash_paths, (params.n_steps + 1, 1)).T,
            pnl_paths=np.tile(pnl_paths, (params.n_steps + 1, 1)).T,
            trade_counts=trade_counts,
            final_pnl=pnl_paths,
            final_inventory=inventories,
            params=params,
            execution_model=execution_model,
        )
    else:
        # Fall back to pure numpy
        return simulate_path(params, execution_model, seed=seed)
