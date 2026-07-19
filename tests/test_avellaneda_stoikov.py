"""
Comprehensive test suite for Avellaneda-Stoikov market making replication.

Tests cover:
1. Core formula validation (exp_4)
2. Strategy implementation correctness
3. Simulation engine correctness
4. Edge cases and boundary conditions
5. Mathematical properties
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp_1 import (
    reservation_price, inventory_deltas, symmetric_quotes,
    simulate_price_path, run_inventory_strategy, run_symmetric_strategy,
    SimulationParams
)
from experiments.exp_3 import (
    compute_inventory_deltas, compute_symmetric_spread,
    run_bernoulli_strategy, run_poisson_strategy
)


class TestReservationPrice:
    """Test reservation price formula implementation."""
    
    def test_reservation_price_zero_inventory(self):
        """When q=0, reservation price should equal mid-price."""
        s = 100.0
        q = 0
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        assert np.isclose(r, s, rtol=1e-10), f"Expected {s}, got {r}"
    
    def test_reservation_price_terminal_time(self):
        """When t approaches T, reservation price should approach mid-price."""
        s = 100.0
        q = 5
        t = 0.999
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        # Should be very close to s
        assert np.isclose(r, s, atol=0.01), f"Expected close to {s}, got {r}"
    
    def test_reservation_price_positive_inventory(self):
        """Positive inventory should lower reservation price."""
        s = 100.0
        q = 5
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        expected_adjustment = q * gamma * sigma**2 * (T - t)
        expected_r = s - expected_adjustment
        
        assert np.isclose(r, expected_r, rtol=1e-10)
        assert r < s, "Reservation price should be below mid-price for long inventory"
    
    def test_reservation_price_negative_inventory(self):
        """Negative inventory should raise reservation price."""
        s = 100.0
        q = -5
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        expected_adjustment = abs(q) * gamma * sigma**2 * (T - t)
        expected_r = s + expected_adjustment
        
        assert np.isclose(r, expected_r, rtol=1e-10)
        assert r > s, "Reservation price should be above mid-price for short inventory"
    
    def test_reservation_price_gamma_limit(self):
        """As gamma approaches 0, inventory effect should vanish."""
        s = 100.0
        q = 10
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma_small = 0.001
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma_small, k)
        # Should be very close to s
        assert np.isclose(r, s, atol=0.1), f"Expected close to {s}, got {r}"
    
    def test_reservation_price_linearity_in_q(self):
        """Reservation price adjustment should be linear in inventory."""
        s = 100.0
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        q1 = 5
        q2 = 10
        
        r1 = reservation_price(s, q1, t, T, sigma, gamma, k)
        r2 = reservation_price(s, q2, t, T, sigma, gamma, k)
        
        # Adjustment should scale linearly
        adjustment1 = s - r1
        adjustment2 = s - r2
        
        assert np.isclose(adjustment2 / adjustment1, q2 / q1, rtol=1e-10)


class TestInventoryDeltas:
    """Test optimal delta computation."""
    
    def test_inventory_deltas_positive(self):
        """Delta should be positive for reasonable parameters."""
        gamma = 0.1
        k = 1.5
        
        delta = inventory_deltas(gamma, k)
        assert delta > 0, f"Delta should be positive, got {delta}"
    
    def test_inventory_deltas_gamma_limit(self):
        """As gamma approaches 0, delta should approach 1/k."""
        gamma = 0.001
        k = 1.5
        
        delta = inventory_deltas(gamma, k)
        # Using Taylor expansion: (1/gamma)*ln(1+gamma/k) ≈ 1/k for small gamma
        expected = 1.0 / k
        
        assert np.isclose(delta, expected, rtol=0.01), f"Expected close to {expected}, got {delta}"
    
    def test_inventory_deltas_monotonic_in_gamma(self):
        """Delta should decrease as gamma increases (for fixed k)."""
        k = 1.5
        
        gamma1 = 0.01
        gamma2 = 0.1
        gamma3 = 0.5
        
        delta1 = inventory_deltas(gamma1, k)
        delta2 = inventory_deltas(gamma2, k)
        delta3 = inventory_deltas(gamma3, k)
        
        assert delta1 > delta2 > delta3, "Delta should decrease with gamma"


class TestSymmetricQuotes:
    """Test symmetric benchmark quotes."""
    
    def test_symmetric_quotes_center(self):
        """Symmetric quotes should be centered around mid-price."""
        s = 100.0
        gamma = 0.1
        k = 1.5
        
        bid, ask = symmetric_quotes(s, gamma, k)
        center = (bid + ask) / 2
        
        assert np.isclose(center, s, rtol=1e-10), f"Center should be {s}, got {center}"
    
    def test_symmetric_quotes_spread(self):
        """Spread should match (2/gamma)*ln(1+gamma/k)."""
        s = 100.0
        gamma = 0.1
        k = 1.5
        
        bid, ask = symmetric_quotes(s, gamma, k)
        spread = ask - bid
        expected_spread = (2.0 / gamma) * np.log(1.0 + gamma / k)
        
        assert np.isclose(spread, expected_spread, rtol=1e-10)
    
    def test_symmetric_quotes_positive_spread(self):
        """Spread should always be positive."""
        s = 100.0
        
        for gamma in [0.01, 0.1, 0.5]:
            for k in [0.5, 1.0, 1.5]:
                bid, ask = symmetric_quotes(s, gamma, k)
                assert ask > bid, f"Ask should exceed bid for gamma={gamma}, k={k}"


class TestInventoryDeltasExp3:
    """Test inventory-dependent deltas from exp_3."""
    
    def test_deltas_sum_formula(self):
        """delta_a + delta_b should equal gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)."""
        gamma = 0.1
        k = 1.5
        sigma = 2.0
        tau = 0.5
        q = 0
        
        delta_a, delta_b = compute_inventory_deltas(gamma, k, sigma, tau, q)
        spread = delta_a + delta_b
        
        expected_spread = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
        
        assert np.isclose(spread, expected_spread, rtol=1e-10)
    
    def test_deltas_inventory_symmetry(self):
        """Deltas should be symmetric around q=0."""
        gamma = 0.1
        k = 1.5
        sigma = 2.0
        tau = 0.5
        
        q = 5
        delta_a_pos, delta_b_pos = compute_inventory_deltas(gamma, k, sigma, tau, q)
        
        q = -5
        delta_a_neg, delta_b_neg = compute_inventory_deltas(gamma, k, sigma, tau, q)
        
        # Should have: delta_a(q) = delta_b(-q) and delta_b(q) = delta_a(-q)
        assert np.isclose(delta_a_pos, delta_b_neg, rtol=1e-10)
        assert np.isclose(delta_b_pos, delta_a_neg, rtol=1e-10)
    
    def test_deltas_inventory_effect_direction(self):
        """Long inventory should narrow ask, widen bid."""
        gamma = 0.1
        k = 1.5
        sigma = 2.0
        tau = 0.5
        
        q = 0
        delta_a_0, delta_b_0 = compute_inventory_deltas(gamma, k, sigma, tau, q)
        
        q = 5
        delta_a_pos, delta_b_pos = compute_inventory_deltas(gamma, k, sigma, tau, q)
        
        # Long inventory: ask distance decreases, bid distance increases
        assert delta_a_pos < delta_a_0, "Ask distance should decrease for long inventory"
        assert delta_b_pos > delta_b_0, "Bid distance should increase for long inventory"


class TestSymmetricSpread:
    """Test symmetric spread computation."""
    
    def test_spread_values_match_paper(self):
        """Spread values should approximately match paper tables."""
        k = 1.5
        
        # Paper table values (approximate)
        expected_spreads = {
            0.01: 1.33,
            0.1: 1.29,
            0.5: 1.15,
        }
        
        for gamma, expected in expected_spreads.items():
            spread = compute_symmetric_spread(gamma, k)
            # Allow some tolerance due to potential parameter differences
            assert np.isclose(spread, expected, rtol=0.1), \
                f"For gamma={gamma}, expected ~{expected}, got {spread}"


class TestPricePathSimulation:
    """Test price path simulation."""
    
    def test_price_path_initial_value(self):
        """Price paths should start at s0."""
        params = SimulationParams(s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200)
        S, t_grid = simulate_price_path(params, n_paths=10)
        
        assert np.all(S[:, 0] == params.s0), "All paths should start at s0"
    
    def test_price_path_length(self):
        """Price paths should have correct length."""
        params = SimulationParams(s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200)
        S, t_grid = simulate_price_path(params, n_paths=10)
        
        assert S.shape[1] == params.N + 1, f"Expected {params.N + 1} time steps, got {S.shape[1]}"
        assert len(t_grid) == params.N + 1, f"Expected {params.N + 1} time points, got {len(t_grid)}"
    
    def test_price_path_variance(self):
        """Price path variance should scale with sigma^2 * t."""
        params = SimulationParams(s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200)
        n_paths = 10000  # Large sample for statistical test
        
        S, t_grid = simulate_price_path(params, n_paths=n_paths)
        
        # At final time, variance should be approximately sigma^2 * T
        final_prices = S[:, -1]
        empirical_variance = np.var(final_prices)
        expected_variance = params.sigma**2 * params.T
        
        # Allow 20% tolerance due to Monte Carlo noise
        assert np.isclose(empirical_variance, expected_variance, rtol=0.2), \
            f"Expected variance ~{expected_variance}, got {empirical_variance}"


class TestStrategyComparison:
    """Test inventory vs symmetric strategy comparison."""
    
    def test_inventory_reduces_variance(self):
        """Inventory strategy should have lower inventory variance than symmetric."""
        params = SimulationParams(
            s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200,
            A=140.0, k=1.5, n_paths=1000
        )
        
        gamma = 0.1
        S, t_grid = simulate_price_path(params, params.n_paths)
        
        inv_pnl, inv_inv, _ = run_inventory_strategy(S, t_grid, params, gamma)
        sym_pnl, sym_inv, _ = run_symmetric_strategy(S, t_grid, params, gamma)
        
        inv_std = np.std(inv_inv)
        sym_std = np.std(sym_inv)
        
        assert inv_std < sym_std, \
            f"Inventory strategy should have lower inventory std: {inv_std} vs {sym_std}"
    
    def test_gamma_reduces_inventory_risk(self):
        """Higher gamma should reduce inventory risk."""
        params = SimulationParams(
            s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200,
            A=140.0, k=1.5, n_paths=1000
        )
        
        S, t_grid = simulate_price_path(params, params.n_paths)
        
        gamma_low = 0.01
        gamma_high = 0.5
        
        _, inv_low, _ = run_inventory_strategy(S, t_grid, params, gamma_low)
        _, inv_high, _ = run_inventory_strategy(S, t_grid, params, gamma_high)
        
        std_low = np.std(inv_low)
        std_high = np.std(inv_high)
        
        assert std_high < std_low, \
            f"Higher gamma should reduce inventory std: {std_high} vs {std_low}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_price_path(self):
        """Should handle minimal price path."""
        params = SimulationParams(s0=100.0, T=1.0, sigma=2.0, dt=1.0, N=1)
        S, t_grid = simulate_price_path(params, n_paths=10)
        
        assert S.shape[1] == 2, "Should have initial and final price"
    
    def test_single_path(self):
        """Should handle single Monte Carlo path."""
        params = SimulationParams(s0=100.0, T=1.0, sigma=2.0, dt=0.005, N=200, n_paths=1)
        S, t_grid = simulate_price_path(params, n_paths=1)
        
        assert S.shape[0] == 1, "Should have single path"
    
    def test_extreme_gamma(self):
        """Should handle extreme gamma values."""
        s = 100.0
        q = 5
        t = 0.5
        T = 1.0
        sigma = 2.0
        k = 1.5
        
        # Very small gamma
        gamma_small = 1e-6
        r_small = reservation_price(s, q, t, T, sigma, gamma_small, k)
        assert np.isfinite(r_small), "Should handle very small gamma"
        
        # Large gamma
        gamma_large = 10.0
        r_large = reservation_price(s, q, t, T, sigma, gamma_large, k)
        assert np.isfinite(r_large), "Should handle large gamma"
    
    def test_zero_time_to_maturity(self):
        """Should handle tau approaching zero."""
        s = 100.0
        q = 5
        t = 1.0
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        assert np.isclose(r, s, atol=1e-10), "At T, reservation price should equal mid-price"


class TestMathematicalProperties:
    """Test mathematical properties and identities."""
    
    def test_spread_identity(self):
        """Verify spread identity: delta_a + delta_b = gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)."""
        gamma_values = [0.01, 0.1, 0.5]
        sigma = 2.0
        tau_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        q = 0
        k = 1.5
        
        for gamma in gamma_values:
            for tau in tau_values:
                delta_a, delta_b = compute_inventory_deltas(gamma, k, sigma, tau, q)
                spread = delta_a + delta_b
                expected = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
                
                assert np.isclose(spread, expected, rtol=1e-10), \
                    f"Spread identity failed for gamma={gamma}, tau={tau}"
    
    def test_center_equals_reservation_price(self):
        """Quote center should equal reservation price."""
        s = 100.0
        q = 5
        t = 0.5
        T = 1.0
        sigma = 2.0
        gamma = 0.1
        k = 1.5
        
        tau = T - t
        delta_a, delta_b = compute_inventory_deltas(gamma, k, sigma, tau, q)
        
        # Quotes around mid-price
        bid = s - delta_b
        ask = s + delta_a
        center = (bid + ask) / 2
        
        # Reservation price
        r = reservation_price(s, q, t, T, sigma, gamma, k)
        
        # Center should equal reservation price
        assert np.isclose(center, r, rtol=1e-10), \
            f"Center {center} should equal reservation price {r}"


def run_all_tests():
    """Run all tests and return results."""
    import subprocess
    
    # Run pytest
    result = subprocess.run(
        ['pytest', __file__, '-v', '--tb=short'],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
