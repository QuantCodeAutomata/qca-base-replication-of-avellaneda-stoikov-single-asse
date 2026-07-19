"""Quote strategies for the market making simulation."""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from params import ModelParams


@dataclass
class Quote:
    """Represents a bid-ask quote.
    
    Attributes:
        delta_b: Bid spread (distance from mid price)
        delta_a: Ask spread (distance from mid price)
    """
    delta_b: float
    delta_a: float
    
    @property
    def spread(self) -> float:
        """Total bid-ask spread."""
        return self.delta_b + self.delta_a
    
    @property
    def mid_offset(self) -> float:
        """Offset of quote mid from market mid."""
        return (self.delta_a - self.delta_b) / 2


def constant_spread_quote(params: ModelParams, 
                          S: float, 
                          inventory: float = 0.0,
                          t: float = 0.0) -> Quote:
    """Constant spread quoting strategy.
    
    Args:
        params: Model parameters
        S: Current mid price
        inventory: Current inventory position
        t: Current time
        
    Returns:
        Quote with constant spread on both sides
    """
    delta = params.delta_max / 2
    return Quote(delta_b=delta, delta_a=delta)


def avellaneda_stoikov_quote(params: ModelParams,
                             S: float,
                             inventory: float = 0.0,
                             t: float = 0.0) -> Quote:
    """Avellaneda-Stoikov optimal quoting strategy.
    
    Implements the classic market making model with inventory control.
    
    Args:
        params: Model parameters (kappa is risk aversion)
        S: Current mid price
        inventory: Current inventory position
        t: Current time
        
    Returns:
        Optimal bid-ask quote
    """
    kappa = params.kappa
    k = params.k
    sigma = params.sigma
    T = params.T
    
    # Reservation price adjustment based on inventory
    r = S - inventory * kappa * sigma**2 * (T - t)
    
    # Optimal spread
    delta_star = 1 / kappa - sigma**2 * (T - t) / 2
    delta_star = max(delta_star, 0.001)  # Ensure positive spread
    
    # Bid and ask quotes
    if inventory > 0:
        # Long inventory: widen ask, tighten bid
        delta_b = delta_star - inventory * kappa * sigma**2 * (T - t)
        delta_a = delta_star + inventory * kappa * sigma**2 * (T - t)
    elif inventory < 0:
        # Short inventory: tighten ask, widen bid
        delta_b = delta_star - inventory * kappa * sigma**2 * (T - t)
        delta_a = delta_star + inventory * kappa * sigma**2 * (T - t)
    else:
        delta_b = delta_star
        delta_a = delta_star
    
    # Clamp to maximum spread
    delta_b = min(max(delta_b, 0.001), params.delta_max)
    delta_a = min(max(delta_a, 0.001), params.delta_max)
    
    return Quote(delta_b=delta_b, delta_a=delta_a)


def inventory_skew_quote(params: ModelParams,
                         S: float,
                         inventory: float = 0.0,
                         t: float = 0.0,
                         skew_factor: float = 0.5) -> Quote:
    """Inventory-skewed quoting strategy.
    
    Simple strategy that skews quotes based on inventory position.
    
    Args:
        params: Model parameters
        S: Current mid price
        inventory: Current inventory position
        t: Current time
        skew_factor: How much to skew quotes based on inventory
        
    Returns:
        Inventory-skewed quote
    """
    base_delta = params.delta_max / 2
    
    # Skew based on inventory (normalized)
    inventory_signal = np.tanh(inventory * skew_factor)
    
    delta_b = base_delta * (1 + inventory_signal)
    delta_a = base_delta * (1 - inventory_signal)
    
    # Clamp to valid range
    delta_b = min(max(delta_b, 0.001), params.delta_max)
    delta_a = min(max(delta_a, 0.001), params.delta_max)
    
    return Quote(delta_b=delta_b, delta_a=delta_a)


def compute_fill_probability(delta: float, params: ModelParams) -> float:
    """Compute fill probability for a given spread.
    
    Uses exponential decay model: lambda = A * exp(-k * delta)
    
    Args:
        delta: Quote spread (distance from mid)
        params: Model parameters
        
    Returns:
        Fill probability (or intensity for Poisson model)
    """
    return params.A * np.exp(-params.k * delta)


def get_quote_strategy(strategy_name: str):
    """Get a quote strategy function by name.
    
    Args:
        strategy_name: Name of the strategy ('constant', 'avellaneda_stoikov', 
                       'inventory_skew')
                       
    Returns:
        Quote strategy function
    """
    strategies = {
        'constant': constant_spread_quote,
        'avellaneda_stoikov': avellaneda_stoikov_quote,
        'inventory_skew': inventory_skew_quote,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(strategies.keys())}")
    
    return strategies[strategy_name]
