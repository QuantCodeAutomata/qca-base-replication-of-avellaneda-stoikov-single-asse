"""
Experiment 3: Execution Model Robustness - Bernoulli vs Poisson Fills

This experiment evaluates how the paper's discrete execution approximation affects results
by comparing Bernoulli per-step fills with exact Poisson arrival simulation.

Model 1 (Bernoulli): Fills occur with probability p = lambda*dt per step
Model 2 (Poisson): Number of fills ~ Poisson(lambda*dt), allowing multiple fills per step

Key question: Does the Bernoulli approximation materially distort results when lambda*dt > 1?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationParams:
    """Parameters matching the Avellaneda-Stoikov paper."""
    s0: float = 100.0      # Initial stock price
    T: float = 1.0         # Trading horizon
    sigma: float = 2.0     # Volatility
    dt: float = 0.005      # Time step
    N: int = 200           # Number of time steps
    q0: int = 0            # Initial inventory
    X0: float = 0.0        # Initial cash
    A: float = 140.0       # Order arrival intensity
    k: float = 1.5         # Order arrival decay
    gamma_values: tuple = (0.01, 0.1, 0.5)
    n_paths: int = 1000


def compute_inventory_deltas(gamma: float, k: float, sigma: float, tau: float, q: int) -> Tuple[float, float]:
    """
    Compute bid and ask distances using inventory-dependent formula.
    
    delta^a = (1/gamma)*ln(1+gamma/k) + ((1-2*q)/2)*gamma*sigma^2*tau
    delta^b = (1/gamma)*ln(1+gamma/k) + ((1+2*q)/2)*gamma*sigma^2*tau
    
    Args:
        gamma: Risk aversion
        k: Order arrival decay
        sigma: Volatility
        tau: Time to maturity
        q: Current inventory
        
    Returns:
        Tuple of (delta_a, delta_b)
    """
    base_delta = (1.0 / gamma) * np.log(1.0 + gamma / k)
    inventory_adjustment = gamma * sigma**2 * tau
    
    delta_a = base_delta + ((1 - 2*q) / 2) * inventory_adjustment
    delta_b = base_delta + ((1 + 2*q) / 2) * inventory_adjustment
    
    return delta_a, delta_b


def compute_symmetric_spread(gamma: float, k: float) -> float:
    """
    Compute symmetric benchmark spread (table-faithful).
    
    spread = (2/gamma) * ln(1 + gamma/k)
    """
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


def simulate_price_paths(params: SimulationParams, n_paths: int, seed: int = None) -> np.ndarray:
    """
    Simulate price paths using binomial Brownian approximation.
    
    S_{t+dt} = S_t +/- sigma*sqrt(dt) with equal probability
    
    Args:
        params: Simulation parameters
        n_paths: Number of paths
        seed: Random seed
        
    Returns:
        Price paths array of shape (n_paths, N+1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = params.dt
    sqrt_dt = np.sqrt(dt)
    
    # Generate random signs (+1 or -1)
    moves = np.random.choice([-1, 1], size=(n_paths, params.N))
    
    # Build price paths
    S = np.zeros((n_paths, params.N + 1))
    S[:, 0] = params.s0
    
    for i in range(params.N):
        S[:, i + 1] = S[:, i] + params.sigma * sqrt_dt * moves[:, i]
    
    return S


def run_bernoulli_strategy(S: np.ndarray, params: SimulationParams, 
                           gamma: float, strategy: str = 'inventory') -> Dict:
    """
    Run strategy with Bernoulli execution model.
    
    At each step, fills occur with probability p = lambda*dt where lambda = A*exp(-k*delta).
    
    Args:
        S: Price paths
        params: Simulation parameters
        gamma: Risk aversion
        strategy: 'inventory' or 'symmetric'
        
    Returns:
        Dictionary with results and diagnostics
    """
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1
    dt = params.dt
    
    # State variables
    X = np.zeros(n_paths)
    q = np.full(n_paths, params.q0, dtype=int)
    
    # Diagnostics
    lambda_dt_max = 0.0
    lambda_dt_exceeds_1 = 0
    total_steps = n_paths * N
    
    for i in range(N):
        t = i * dt
        tau = params.T - t
        s = S[:, i]
        
        # Compute quotes based on strategy
        if strategy == 'inventory':
            # Inventory-dependent deltas
            delta_a, delta_b = compute_inventory_deltas(gamma, params.k, params.sigma, tau, q)
            bid = s - delta_b
            ask = s + delta_a
        else:  # symmetric
            spread = compute_symmetric_spread(gamma, params.k)
            bid = s - spread / 2
            ask = s + spread / 2
            delta_a = delta_b = spread / 2
        
        # Compute intensities
        lambda_bid = params.A * np.exp(-params.k * np.abs(bid - s))
        lambda_ask = params.A * np.exp(-params.k * np.abs(ask - s))
        
        # Track diagnostics
        current_lambda_dt = np.maximum(lambda_bid * dt, lambda_ask * dt)
        lambda_dt_max = max(lambda_dt_max, np.max(current_lambda_dt))
        lambda_dt_exceeds_1 += np.sum(current_lambda_dt > 1)
        
        # Bernoulli fills
        fill_bid = np.random.rand(n_paths) < (lambda_bid * dt)
        fill_ask = np.random.rand(n_paths) < (lambda_ask * dt)
        
        # Update state
        X[fill_bid] -= bid[fill_bid]
        q[fill_bid] += 1
        
        X[fill_ask] += ask[fill_ask]
        q[fill_ask] -= 1
    
    # Terminal P&L
    terminal_pnl = X + q * S[:, -1]
    terminal_inventory = q.copy()
    
    return {
        'terminal_pnl': terminal_pnl,
        'terminal_inventory': terminal_inventory,
        'lambda_dt_max': lambda_dt_max,
        'lambda_dt_exceeds_1_count': lambda_dt_exceeds_1,
        'lambda_dt_exceeds_1_pct': lambda_dt_exceeds_1 / total_steps * 100,
        'strategy': strategy,
        'execution_model': 'bernoulli',
    }


def run_poisson_strategy(S: np.ndarray, params: SimulationParams,
                         gamma: float, strategy: str = 'inventory') -> Dict:
    """
    Run strategy with exact Poisson execution model.
    
    Number of fills per step ~ Poisson(lambda*dt), allowing multiple fills.
    
    Args:
        S: Price paths
        params: Simulation parameters
        gamma: Risk aversion
        strategy: 'inventory' or 'symmetric'
        
    Returns:
        Dictionary with results and diagnostics
    """
    n_paths, N_plus_1 = S.shape
    N = N_plus_1 - 1
    dt = params.dt
    
    # State variables
    X = np.zeros(n_paths)
    q = np.full(n_paths, params.q0, dtype=int)
    
    # Diagnostics
    lambda_dt_max = 0.0
    total_steps = n_paths * N
    fills_per_step_bid = []
    fills_per_step_ask = []
    multiple_fills_count = 0
    
    for i in range(N):
        t = i * dt
        tau = params.T - t
        s = S[:, i]
        
        # Compute quotes
        if strategy == 'inventory':
            delta_a, delta_b = compute_inventory_deltas(gamma, params.k, params.sigma, tau, q)
            bid = s - delta_b
            ask = s + delta_a
        else:  # symmetric
            spread = compute_symmetric_spread(gamma, params.k)
            bid = s - spread / 2
            ask = s + spread / 2
            delta_a = delta_b = spread / 2
        
        # Compute intensities
        lambda_bid = params.A * np.exp(-params.k * np.abs(bid - s))
        lambda_ask = params.A * np.exp(-params.k * np.abs(ask - s))
        
        # Track diagnostics
        current_lambda_dt = np.maximum(lambda_bid * dt, lambda_ask * dt)
        lambda_dt_max = max(lambda_dt_max, np.max(current_lambda_dt))
        
        # Poisson fills (can have multiple fills per step)
        n_bid_fills = np.random.poisson(lambda_bid * dt)
        n_ask_fills = np.random.poisson(lambda_ask * dt)
        
        # Track fill distribution
        fills_per_step_bid.extend(n_bid_fills)
        fills_per_step_ask.extend(n_ask_fills)
        multiple_fills_count += np.sum((n_bid_fills > 1) | (n_ask_fills > 1))
        
        # Update state (vectorized)
        for path_idx in range(n_paths):
            if n_bid_fills[path_idx] > 0:
                q[path_idx] += n_bid_fills[path_idx]
                X[path_idx] -= n_bid_fills[path_idx] * bid[path_idx]
            
            if n_ask_fills[path_idx] > 0:
                q[path_idx] -= n_ask_fills[path_idx]
                X[path_idx] += n_ask_fills[path_idx] * ask[path_idx]
    
    # Terminal P&L
    terminal_pnl = X + q * S[:, -1]
    terminal_inventory = q.copy()
    
    # Compute fill statistics
    fills_per_step_bid = np.array(fills_per_step_bid)
    fills_per_step_ask = np.array(fills_per_step_ask)
    
    return {
        'terminal_pnl': terminal_pnl,
        'terminal_inventory': terminal_inventory,
        'lambda_dt_max': lambda_dt_max,
        'mean_fills_per_step_bid': np.mean(fills_per_step_bid),
        'mean_fills_per_step_ask': np.mean(fills_per_step_ask),
        'max_fills_single_step': max(np.max(fills_per_step_bid), np.max(fills_per_step_ask)),
        'multiple_fills_count': multiple_fills_count,
        'multiple_fills_pct': multiple_fills_count / total_steps * 100,
        'strategy': strategy,
        'execution_model': 'poisson',
    }


def compute_statistics(results: Dict) -> Dict[str, float]:
    """Compute summary statistics from simulation results."""
    pnl = results['terminal_pnl']
    inv = results['terminal_inventory']
    
    return {
        'mean_pnl': float(np.mean(pnl)),
        'std_pnl': float(np.std(pnl)),
        'median_pnl': float(np.median(pnl)),
        'pnl_25th': float(np.percentile(pnl, 25)),
        'pnl_75th': float(np.percentile(pnl, 75)),
        'mean_inventory': float(np.mean(inv)),
        'std_inventory': float(np.std(inv)),
        'max_abs_inventory': float(np.max(np.abs(inv))),
    }


def print_comparison_table(bernoulli_results: Dict, poisson_results: Dict, 
                          gamma: float, strategy: str) -> None:
    """Print comparison table between Bernoulli and Poisson models."""
    print(f"\n{'='*90}")
    print(f"EXECUTION MODEL COMPARISON (γ={gamma}, {strategy.capitalize()} Strategy)")
    print(f"{'='*90}")
    
    bern_stats = compute_statistics(bernoulli_results)
    pois_stats = compute_statistics(poisson_results)
    
    print(f"\n{'Metric':<30} {'Bernoulli':>15} {'Poisson':>15} {'Difference':>15}")
    print(f"{'-'*90}")
    
    metrics = [
        ('Mean P&L', 'mean_pnl'),
        ('Std P&L', 'std_pnl'),
        ('Median P&L', 'median_pnl'),
        ('Mean Inventory', 'mean_inventory'),
        ('Std Inventory', 'std_inventory'),
        ('Max |Inventory|', 'max_abs_inventory'),
    ]
    
    for label, key in metrics:
        bern_val = bern_stats[key]
        pois_val = pois_stats[key]
        diff = bern_val - pois_val
        print(f"{label:<30} {bern_val:>15.4f} {pois_val:>15.4f} {diff:>15.4f}")
    
    # Execution diagnostics
    print(f"\n{'Execution Diagnostics':<30}")
    print(f"{'-'*90}")
    print(f"  Max lambda*dt:           {bernoulli_results['lambda_dt_max']:>15.6f} {poisson_results['lambda_dt_max']:>15.6f}")
    print(f"  Lambda*dt > 1 count:     {bernoulli_results['lambda_dt_exceeds_1_count']:>15d} {'N/A':>15}")
    print(f"  Lambda*dt > 1 %:         {bernoulli_results['lambda_dt_exceeds_1_pct']:>15.4f} {'N/A':>15}")
    print(f"  Multiple fills count:    {'N/A':>15} {poisson_results['multiple_fills_count']:>15d}")
    print(f"  Multiple fills %:        {'N/A':>15} {poisson_results['multiple_fills_pct']:>15.4f}")
    print(f"  Mean fills/step (bid):   {'N/A':>15} {poisson_results['mean_fills_per_step_bid']:>15.6f}")
    print(f"  Mean fills/step (ask):   {'N/A':>15} {poisson_results['mean_fills_per_step_ask']:>15.6f}")
    print(f"  Max fills in one step:   {'N/A':>15} {poisson_results['max_fills_single_step']:>15d}")


def plot_execution_comparison(bernoulli_results: Dict, poisson_results: Dict,
                             gamma: float, strategy: str, save_path: str = None) -> None:
    """Plot comparison of execution models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L histograms
    bins_pnl = 50
    axes[0, 0].hist(bernoulli_results['terminal_pnl'], bins=bins_pnl, alpha=0.6, 
                    label='Bernoulli', color='blue', edgecolor='black')
    axes[0, 0].hist(poisson_results['terminal_pnl'], bins=bins_pnl, alpha=0.6,
                    label='Poisson', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Terminal P&L')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Terminal P&L Distribution (γ={gamma}, {strategy})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Inventory histograms
    inv_min = min(bernoulli_results['terminal_inventory'].min(), 
                  poisson_results['terminal_inventory'].min())
    inv_max = max(bernoulli_results['terminal_inventory'].max(),
                  poisson_results['terminal_inventory'].max())
    bins_inv = np.arange(inv_min, inv_max + 2) - 0.5
    
    axes[0, 1].hist(bernoulli_results['terminal_inventory'], bins=bins_inv, alpha=0.6,
                    label='Bernoulli', color='blue', edgecolor='black')
    axes[0, 1].hist(poisson_results['terminal_inventory'], bins=bins_inv, alpha=0.6,
                    label='Poisson', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Terminal Inventory')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Terminal Inventory Distribution (γ={gamma}, {strategy})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # P&L boxplot
    bp_data = axes[1, 0].boxplot([bernoulli_results['terminal_pnl'], 
                                   poisson_results['terminal_pnl']], 
                                  positions=[1, 2], widths=0.6)
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(['Bernoulli', 'Poisson'])
    axes[1, 0].set_ylabel('Terminal P&L')
    axes[1, 0].set_title(f'P&L Boxplot (γ={gamma}, {strategy})')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Inventory boxplot
    bp_inv = axes[1, 1].boxplot([bernoulli_results['terminal_inventory'],
                                  poisson_results['terminal_inventory']],
                                 positions=[1, 2], widths=0.6)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Bernoulli', 'Poisson'])
    axes[1, 1].set_ylabel('Terminal Inventory')
    axes[1, 1].set_title(f'Inventory Boxplot (γ={gamma}, {strategy})')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def run_experiment(params: SimulationParams, seed: int = 42) -> pd.DataFrame:
    """
    Run full experiment comparing Bernoulli vs Poisson execution models.
    
    Args:
        params: Simulation parameters
        seed: Random seed
        
    Returns:
        DataFrame with all results
    """
    np.random.seed(seed)
    
    print("="*90)
    print("EXPERIMENT 3: EXECUTION MODEL ROBUSTNESS")
    print("Bernoulli Approximation vs Exact Poisson Fills")
    print("="*90)
    
    print(f"\nParameters:")
    print(f"  s0 = {params.s0}, T = {params.T}, sigma = {params.sigma}")
    print(f"  dt = {params.dt}, N = {params.N}")
    print(f"  A = {params.A}, k = {params.k}")
    print(f"  gamma values = {params.gamma_values}")
    print(f"  Number of paths = {params.n_paths}")
    
    # Simulate common price paths
    print("\nSimulating common price paths...")
    S = simulate_price_paths(params, params.n_paths, seed=seed)
    
    all_results = []
    
    for gamma in params.gamma_values:
        print(f"\n{'#'*90}")
        print(f"Gamma = {gamma}")
        print(f"{'#'*90}")
        
        for strategy in ['inventory', 'symmetric']:
            print(f"\nRunning {strategy} strategy...")
            
            # Bernoulli model
            print(f"  Bernoulli execution...")
            bern_results = run_bernoulli_strategy(S, params, gamma, strategy)
            
            # Poisson model (use same price paths but different execution randomness)
            print(f"  Poisson execution...")
            pois_results = run_poisson_strategy(S, params, gamma, strategy)
            
            # Print comparison
            print_comparison_table(bern_results, pois_results, gamma, strategy)
            
            # Plot
            plot_execution_comparison(
                bern_results, pois_results, gamma, strategy,
                save_path=f'experiments/results/gamma_{gamma}_{strategy}_execution.png'
            )
            
            # Store results
            bern_stats = compute_statistics(bern_results)
            pois_stats = compute_statistics(pois_results)
            
            all_results.append({
                'gamma': gamma,
                'strategy': strategy,
                'execution_model': 'bernoulli',
                **bern_stats,
                'lambda_dt_max': bern_results['lambda_dt_max'],
                'lambda_dt_exceeds_1_pct': bern_results['lambda_dt_exceeds_1_pct'],
            })
            
            all_results.append({
                'gamma': gamma,
                'strategy': strategy,
                'execution_model': 'poisson',
                **pois_stats,
                'lambda_dt_max': pois_results['lambda_dt_max'],
                'multiple_fills_pct': pois_results['multiple_fills_pct'],
                'max_fills_single_step': pois_results['max_fills_single_step'],
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print overall summary
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY: EXECUTION MODEL COMPARISON")
    print(f"{'='*90}")
    
    print(f"\n{'Gamma':<10} {'Strategy':<15} {'Model':<12} {'Mean P&L':>12} {'Std P&L':>12} {'Std Inv':>12}")
    print(f"{'-'*90}")
    
    for gamma in params.gamma_values:
        for strategy in ['inventory', 'symmetric']:
            for model in ['bernoulli', 'poisson']:
                row = results_df[(results_df['gamma'] == gamma) & 
                                (results_df['strategy'] == strategy) &
                                (results_df['execution_model'] == model)].iloc[0]
                print(f"{gamma:<10.2f} {strategy:<15} {model:<12} {row['mean_pnl']:>12.4f} {row['std_pnl']:>12.4f} {row['std_inventory']:>12.4f}")
    
    print(f"{'='*90}")
    
    # Save results
    results_df.to_csv('experiments/results/exp_3_execution_model_results.csv', index=False)
    print(f"\nResults saved to experiments/results/exp_3_execution_model_results.csv")
    
    return results_df


def main():
    """Main entry point."""
    import os
    os.makedirs('experiments/results', exist_ok=True)
    
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
    
    results = run_experiment(params, seed=42)
    
    print("\nExperiment 3 completed successfully!")
    
    return results


if __name__ == "__main__":
    results = main()
