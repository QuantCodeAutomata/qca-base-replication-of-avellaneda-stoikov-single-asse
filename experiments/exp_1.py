"""
Experiment 1: Base Replication - Monte Carlo Simulation
Comparing Inventory-based vs Symmetric Benchmark Strategies

Based on Avellaneda-Stoikov (2008) market-making model formulas.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationParams:
    """Simulation parameters as specified in the paper."""
    s0: float = 100.0      # Initial stock price
    T: float = 1.0         # Trading horizon (1 day)
    sigma: float = 2.0     # Volatility
    dt: float = 0.005      # Time step
    N: int = 200           # Number of time steps (T/dt)
    q0: int = 0            # Initial inventory
    X0: float = 0.0        # Initial cash
    A: float = 140.0       # Order arrival intensity parameter
    k: float = 1.5         # Order arrival decay parameter
    gamma_values: tuple = (0.01, 0.1, 0.5)  # Risk aversion parameters
    n_paths: int = 1000    # Number of Monte Carlo paths


def reservation_price(s: float, q: int, t: float, T: float, 
                      sigma: float, gamma: float, k: float) -> float:
    """
    Compute the reservation price per Avellaneda-Stoikov formula.
    
    r(s, t) = s - q * gamma * sigma^2 * (T - t)
    
    This is the price at which the market maker is indifferent to holding
    additional inventory, accounting for risk aversion.
    
    Args:
        s: Current stock price
        q: Current inventory
        t: Current time
        T: Terminal time
        sigma: Volatility
        gamma: Risk aversion parameter
        k: Order arrival decay parameter
    
    Returns:
        Reservation price
    """
    return s - q * gamma * sigma**2 * (T - t)


def inventory_deltas(gamma: float, k: float) -> float:
    """
    Compute the optimal inventory adjustment (delta) per paper formula.
    
    delta = (1/gamma) * ln(1 + gamma/k)
    
    This represents the optimal spread adjustment based on inventory risk.
    
    Args:
        gamma: Risk aversion parameter
        k: Order arrival decay parameter
    
    Returns:
        Optimal inventory delta
    """
    return (1.0 / gamma) * np.log(1.0 + gamma / k)


def symmetric_quotes(s: float, gamma: float, k: float) -> Tuple[float, float]:
    """
    Compute symmetric benchmark quotes using table-faithful spread.
    
    The spread is computed as: spread = (2/gamma) * ln(1 + gamma/k)
    Quotes are symmetric around the mid-price s.
    
    Args:
        s: Current stock price (mid-price)
        gamma: Risk aversion parameter
        k: Order arrival decay parameter
    
    Returns:
        Tuple of (bid_price, ask_price)
    """
    spread = (2.0 / gamma) * np.log(1.0 + gamma / k)
    bid = s - spread / 2.0
    ask = s + spread / 2.0
    return bid, ask


def simulate_price_path(params: SimulationParams, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate stock price paths using binomial Brownian approximation.
    
    As specified in the paper: S_{t+dt} = S_t +/- sigma*sqrt(dt) with equal probability.
    This is arithmetic Brownian motion, NOT geometric Brownian motion.
    
    Args:
        params: Simulation parameters
        n_paths: Number of paths to simulate
    
    Returns:
        Tuple of (price_paths, time_grid) where price_paths has shape (n_paths, params.N + 1)
    """
    dt = params.dt
    T = params.T
    N = params.N
    
    # Time grid
    t_grid = np.linspace(0, T, N + 1)
    
    # Simulate binomial increments: +/- sigma*sqrt(dt) with equal probability
    # Using arithmetic Brownian motion as specified in the paper
    increments = params.sigma * np.sqrt(dt) * (2 * np.random.binomial(1, 0.5, size=(n_paths, N)) - 1)
    
    # Price paths
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = params.s0
    
    for i in range(N):
        S[:, i + 1] = S[:, i] + increments[:, i]
    
    return S, t_grid


def run_inventory_strategy(S: np.ndarray, t_grid: np.ndarray, 
                           params: SimulationParams, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the inventory-based market making strategy.
    
    Uses reservation price and optimal deltas to set bid/ask quotes.
    Simulates fills as Bernoulli(lambda*dt) where lambda = A * exp(-k * |delta|).
    
    Args:
        S: Price paths of shape (n_paths, N+1)
        t_grid: Time grid
        params: Simulation parameters
        gamma: Risk aversion parameter
    
    Returns:
        Tuple of (terminal_pnl, terminal_inventory, cash_paths)
    """
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1
    dt = params.dt
    
    # Initialize state variables
    X = np.zeros(n_paths)  # Cash account
    q = np.full(n_paths, params.q0, dtype=int)  # Inventory
    
    # Track P&L over time
    pnl_paths = np.zeros((n_paths, N_plus_1))
    
    for i in range(N):
        t = t_grid[i]
        s = S[:, i]
        
        # Compute reservation price
        r = reservation_price(s, q, t, params.T, params.sigma, gamma, params.k)
        
        # Compute optimal delta
        delta = inventory_deltas(gamma, params.k)
        
        # Set bid and ask quotes around reservation price
        bid = r - delta
        ask = r + delta
        
        # Compute arrival intensities
        # lambda = A * exp(-k * |price_distance|)
        dist_bid = np.abs(bid - s)
        dist_ask = np.abs(ask - s)
        
        lambda_bid = params.A * np.exp(-params.k * dist_bid)
        lambda_ask = params.A * np.exp(-params.k * dist_ask)
        
        # Simulate fills as Bernoulli(lambda * dt)
        fill_bid = np.random.rand(n_paths) < (lambda_bid * dt)
        fill_ask = np.random.rand(n_paths) < (lambda_ask * dt)
        
        # Update inventory and cash
        # Bid fill: buy at bid price (inventory increases)
        X[fill_bid] -= bid[fill_bid]
        q[fill_bid] += 1
        
        # Ask fill: sell at ask price (inventory decreases)
        X[fill_ask] += ask[fill_ask]
        q[fill_ask] -= 1
        
        # Update P&L: cash + inventory * current_price
        pnl_paths[:, i + 1] = X + q * S[:, i + 1]
    
    # Terminal P&L includes liquidation at final price
    terminal_pnl = X + q * S[:, -1]
    terminal_inventory = q.copy()
    
    return terminal_pnl, terminal_inventory, pnl_paths


def run_symmetric_strategy(S: np.ndarray, t_grid: np.ndarray,
                           params: SimulationParams, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the symmetric benchmark strategy.
    
    Uses symmetric quotes around mid-price with table-faithful spread.
    Simulates fills as Bernoulli(lambda*dt).
    
    Args:
        S: Price paths of shape (n_paths, N+1)
        t_grid: Time grid
        params: Simulation parameters
        gamma: Risk aversion parameter
    
    Returns:
        Tuple of (terminal_pnl, terminal_inventory, cash_paths)
    """
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1
    dt = params.dt
    
    # Initialize state variables
    X = np.zeros(n_paths)  # Cash account
    q = np.full(n_paths, params.q0, dtype=int)  # Inventory
    
    # Track P&L over time
    pnl_paths = np.zeros((n_paths, N_plus_1))
    
    for i in range(N):
        t = t_grid[i]
        s = S[:, i]
        
        # Compute symmetric quotes
        bid, ask = symmetric_quotes(s, gamma, params.k)
        
        # Compute arrival intensities
        dist_bid = np.abs(bid - s)
        dist_ask = np.abs(ask - s)
        
        lambda_bid = params.A * np.exp(-params.k * dist_bid)
        lambda_ask = params.A * np.exp(-params.k * dist_ask)
        
        # Simulate fills as Bernoulli(lambda * dt)
        fill_bid = np.random.rand(n_paths) < (lambda_bid * dt)
        fill_ask = np.random.rand(n_paths) < (lambda_ask * dt)
        
        # Update inventory and cash
        X[fill_bid] -= bid[fill_bid]
        q[fill_bid] += 1
        
        X[fill_ask] += ask[fill_ask]
        q[fill_ask] -= 1
        
        # Update P&L
        pnl_paths[:, i + 1] = X + q * S[:, i + 1]
    
    # Terminal P&L
    terminal_pnl = X + q * S[:, -1]
    terminal_inventory = q.copy()
    
    return terminal_pnl, terminal_inventory, pnl_paths


def compute_summary_statistics(pnl: np.ndarray, inventory: np.ndarray) -> dict:
    """
    Compute summary statistics matching paper tables.
    
    Args:
        pnl: Terminal P&L array
        inventory: Terminal inventory array
    
    Returns:
        Dictionary of summary statistics
    """
    return {
        'mean_pnl': np.mean(pnl),
        'std_pnl': np.std(pnl),
        'median_pnl': np.median(pnl),
        'pnl_25th': np.percentile(pnl, 25),
        'pnl_75th': np.percentile(pnl, 75),
        'pnl_min': np.min(pnl),
        'pnl_max': np.max(pnl),
        'mean_inventory': np.mean(inventory),
        'std_inventory': np.std(inventory),
        'inventory_abs_mean': np.mean(np.abs(inventory)),
        'inventory_max_abs': np.max(np.abs(inventory)),
        'sharpe_ratio': np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
    }


def plot_histograms(inventory_pnl: np.ndarray, inventory_inv: np.ndarray,
                    symmetric_pnl: np.ndarray, symmetric_inv: np.ndarray,
                    gamma: float, save_path: str = None) -> None:
    """
    Generate histograms comparing strategies.
    
    Args:
        inventory_pnl: Terminal P&L from inventory strategy
        inventory_inv: Terminal inventory from inventory strategy
        symmetric_pnl: Terminal P&L from symmetric strategy
        symmetric_inv: Terminal inventory from symmetric strategy
        gamma: Risk aversion parameter
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L histograms
    bins_pnl = np.linspace(min(inventory_pnl.min(), symmetric_pnl.min()),
                           max(inventory_pnl.max(), symmetric_pnl.max()), 50)
    
    axes[0, 0].hist(inventory_pnl, bins=bins_pnl, alpha=0.6, label='Inventory-based', 
                    color='blue', edgecolor='black')
    axes[0, 0].hist(symmetric_pnl, bins=bins_pnl, alpha=0.6, label='Symmetric', 
                    color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Terminal P&L')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Terminal P&L Distribution (γ={gamma})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # P&L boxplot
    bp_data = axes[0, 1].boxplot([inventory_pnl, symmetric_pnl], positions=[1, 2], widths=0.6)
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Inventory', 'Symmetric'])
    axes[0, 1].set_ylabel('Terminal P&L')
    axes[0, 1].set_title(f'P&L Boxplot Comparison (γ={gamma})')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Inventory histograms
    inv_min = min(inventory_inv.min(), symmetric_inv.min())
    inv_max = max(inventory_inv.max(), symmetric_inv.max())
    n_bins = int(inv_max - inv_min) + 1
    bins_inv = np.arange(inv_min, inv_max + 2) - 0.5
    
    axes[1, 0].hist(inventory_inv, bins=bins_inv, alpha=0.6, label='Inventory-based',
                    color='blue', edgecolor='black')
    axes[1, 0].hist(symmetric_inv, bins=bins_inv, alpha=0.6, label='Symmetric',
                    color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Terminal Inventory')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Terminal Inventory Distribution (γ={gamma})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Inventory boxplot
    bp_inv = axes[1, 1].boxplot([inventory_inv, symmetric_inv], positions=[1, 2], widths=0.6)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Inventory', 'Symmetric'])
    axes[1, 1].set_ylabel('Terminal Inventory')
    axes[1, 1].set_title(f'Inventory Boxplot Comparison (γ={gamma})')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_summary_table(inventory_stats: dict, symmetric_stats: dict, gamma: float) -> None:
    """
    Print summary statistics table matching paper format.
    
    Args:
        inventory_stats: Statistics from inventory strategy
        symmetric_stats: Statistics from symmetric strategy
        gamma: Risk aversion parameter
    """
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS (γ = {gamma})")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'Inventory-based':>15} {'Symmetric':>15} {'Difference':>15}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Mean P&L', 'mean_pnl'),
        ('Std P&L', 'std_pnl'),
        ('Median P&L', 'median_pnl'),
        ('P&L 25th %ile', 'pnl_25th'),
        ('P&L 75th %ile', 'pnl_75th'),
        ('P&L Min', 'pnl_min'),
        ('P&L Max', 'pnl_max'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Mean |Inventory|', 'inventory_abs_mean'),
        ('Std Inventory', 'std_inventory'),
        ('Max |Inventory|', 'inventory_max_abs'),
    ]
    
    for label, key in metrics:
        inv_val = inventory_stats[key]
        sym_val = symmetric_stats[key]
        diff = inv_val - sym_val
        print(f"{label:<25} {inv_val:>15.4f} {sym_val:>15.4f} {diff:>15.4f}")
    
    print(f"{'='*80}")


def run_experiment(params: SimulationParams, seed: int = 42) -> dict:
    """
    Run the full Monte Carlo experiment for all gamma values.
    
    Args:
        params: Simulation parameters
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing all results
    """
    np.random.seed(seed)
    
    print("="*80)
    print("EXPERIMENT 1: BASE REPLICATION")
    print("Monte Carlo Simulation: Inventory-based vs Symmetric Benchmark")
    print("="*80)
    
    print(f"\nParameters:")
    print(f"  s0 = {params.s0}, T = {params.T}, sigma = {params.sigma}")
    print(f"  dt = {params.dt}, N = {params.N}")
    print(f"  q0 = {params.q0}, X0 = {params.X0}")
    print(f"  A = {params.A}, k = {params.k}")
    print(f"  gamma values = {params.gamma_values}")
    print(f"  Number of paths = {params.n_paths}")
    
    # Simulate price paths (same for all strategies)
    print("\nSimulating price paths...")
    S, t_grid = simulate_price_path(params, params.n_paths)
    
    results = {}
    
    for gamma in params.gamma_values:
        print(f"\n{'#'*80}")
        print(f"Running simulation for gamma = {gamma}")
        print(f"{'#'*80}")
        
        # Compute theoretical values
        delta = inventory_deltas(gamma, params.k)
        spread = (2.0 / gamma) * np.log(1.0 + gamma / params.k)
        
        print(f"\nTheoretical values:")
        print(f"  Optimal delta = {delta:.6f}")
        print(f"  Symmetric spread = {spread:.6f}")
        
        # Run inventory-based strategy
        print("Running inventory-based strategy...")
        inv_pnl, inv_inventory, inv_pnl_paths = run_inventory_strategy(
            S, t_grid, params, gamma
        )
        
        # Run symmetric benchmark strategy
        print("Running symmetric benchmark strategy...")
        sym_pnl, sym_inventory, sym_pnl_paths = run_symmetric_strategy(
            S, t_grid, params, gamma
        )
        
        # Compute statistics
        inv_stats = compute_summary_statistics(inv_pnl, inv_inventory)
        sym_stats = compute_summary_statistics(sym_pnl, sym_inventory)
        
        # Print summary table
        print_summary_table(inv_stats, sym_stats, gamma)
        
        # Generate plots
        plot_histograms(inv_pnl, inv_inventory, sym_pnl, sym_inventory,
                       gamma, save_path=f'experiments/results/gamma_{gamma}_histograms.png')
        
        # Store results
        results[gamma] = {
            'inventory_pnl': inv_pnl,
            'inventory_inventory': inv_inventory,
            'inventory_pnl_paths': inv_pnl_paths,
            'symmetric_pnl': sym_pnl,
            'symmetric_inventory': sym_inventory,
            'symmetric_pnl_paths': sym_pnl_paths,
            'inventory_stats': inv_stats,
            'symmetric_stats': sym_stats,
        }
    
    # Print overall comparison
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON ACROSS GAMMA VALUES")
    print(f"{'='*80}")
    
    print(f"\n{'Gamma':<10} {'Strategy':<20} {'Mean P&L':>12} {'Std P&L':>12} {'Sharpe':>10}")
    print(f"{'-'*80}")
    
    for gamma in params.gamma_values:
        inv_stats = results[gamma]['inventory_stats']
        sym_stats = results[gamma]['symmetric_stats']
        print(f"{gamma:<10.2f} {'Inventory':<20} {inv_stats['mean_pnl']:>12.4f} {inv_stats['std_pnl']:>12.4f} {inv_stats['sharpe_ratio']:>10.4f}")
        print(f"{gamma:<10.2f} {'Symmetric':<20} {sym_stats['mean_pnl']:>12.4f} {sym_stats['std_pnl']:>12.4f} {sym_stats['sharpe_ratio']:>10.4f}")
    
    print(f"{'='*80}")
    
    return results


def main():
    """Main entry point for the experiment."""
    # Create results directory
    import os
    os.makedirs('experiments/results', exist_ok=True)
    
    # Set up parameters as specified
    params = SimulationParams(
        s0=100.0,
        T=1.0,
        sigma=2.0,
        dt=0.005,
        N=200,
        q0=0,
        X0=0.0,
        A=140.0,
        k=1.5,
        gamma_values=(0.01, 0.1, 0.5),
        n_paths=1000
    )
    
    # Run experiment
    results = run_experiment(params, seed=42)
    
    print("\nExperiment completed successfully!")
    print("Results saved to experiments/results/")
    
    return results


if __name__ == "__main__":
    results = main()
