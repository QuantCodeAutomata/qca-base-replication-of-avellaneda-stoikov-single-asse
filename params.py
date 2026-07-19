"""Model parameters for the simulation engine."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelParams:
    """Parameters for the market making simulation model.
    
    Attributes:
        S0: Initial stock price
        T: Time horizon (in years)
        n_steps: Number of time steps for simulation
        sigma: Volatility parameter
        A: Fill probability amplitude parameter
        k: Fill probability decay parameter
        kappa: Risk aversion parameter
        delta_max: Maximum quote spread
        execution_model: Execution model ('bernoulli' or 'poisson')
        n_paths: Number of simulation paths
        seed: Random seed for reproducibility (None for random)
        use_common_random: Whether to use common random numbers
    """
    S0: float = 100.0
    T: float = 1.0
    n_steps: int = 252
    sigma: float = 0.2
    A: float = 0.1
    k: float = 0.5
    kappa: float = 0.1
    delta_max: float = 0.1
    execution_model: str = 'bernoulli'
    n_paths: int = 1000
    seed: Optional[int] = None
    use_common_random: bool = False
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.S0 <= 0:
            raise ValueError("S0 must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.A <= 0:
            raise ValueError("A must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.delta_max <= 0:
            raise ValueError("delta_max must be positive")
        if self.execution_model not in ('bernoulli', 'poisson'):
            raise ValueError("execution_model must be 'bernoulli' or 'poisson'")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
    
    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.n_steps
    
    @property
    def sqrt_dt(self) -> float:
        """Square root of time step size."""
        return self.dt ** 0.5
