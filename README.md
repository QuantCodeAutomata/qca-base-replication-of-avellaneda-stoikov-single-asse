# Avellaneda-Stoikov Market Making Replication

A comprehensive Python implementation replicating the Avellaneda-Stoikov (2008) market-making model with four distinct experiments.

## Overview

This repository implements a faithful replication of the seminal Avellaneda-Stoikov paper on high-frequency market making. The implementation includes:

- **Inventory-based quoting strategy** with reservation price centering
- **Symmetric benchmark strategy** for comparison
- **Monte Carlo simulation** framework with 1000+ paths
- **Four comprehensive experiments** validating different aspects of the model

## Repository Structure

```
.
├── src/                        # Core implementation
│   ├── params.py              # Model parameters and dataclasses
│   ├── strategies.py          # Quote formulas and strategies
│   ├── simulator.py           # Path simulation engine
│   └── metrics.py             # Aggregation and statistics
│
├── experiments/                # Experiment scripts
│   ├── exp_1.py               # Base replication (Section 3.3)
│   ├── exp_2.py               # Spread convention ambiguity
│   ├── exp_3.py               # Execution model comparison
│   ├── exp_4.py               # Formula validation
│   └── results/               # Experiment outputs
│
├── tests/                      # Test suite
│   └── test_avellaneda_stoikov.py
│
├── results/                    # Final results
│   ├── RESULTS.md             # Comprehensive results summary
│   └── *.png, *.csv           # Figures and data
│
├── data/                       # Data directory (simulated data)
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.26.4
- pandas >= 2.2.2
- scipy >= 1.14.1
- matplotlib
- seaborn
- pytest

## Usage

### Run All Experiments

```bash
# Experiment 1: Base replication
python experiments/exp_1.py

# Experiment 2: Ambiguity resolution
python experiments/exp_2.py

# Experiment 3: Execution model comparison
python experiments/exp_3.py

# Experiment 4: Formula validation
python experiments/exp_4.py
```

### Run Tests

```bash
pytest tests/test_avellaneda_stoikov.py -v
```

## Model Parameters

Default parameters matching the Avellaneda-Stoikov paper:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Initial price | s₀ | 100 | Starting mid-price |
| Horizon | T | 1 | Trading horizon (days) |
| Volatility | σ | 2 | Price volatility |
| Time step | dt | 0.005 | Simulation step size |
| Steps | N | 200 | Number of time steps |
| Intensity scale | A | 140 | Order arrival rate scale |
| Intensity decay | k | 1.5 | Order arrival rate decay |
| Risk aversion | γ | {0.01, 0.1, 0.5} | Dealer risk aversion |

## Key Formulas

### Reservation Price
```
r(s, q, t) = s - q · γ · σ² · (T - t)
```

### Optimal Quote Distances
```
δ^a = (1/γ) · ln(1 + γ/k) + ((1 - 2q)/2) · γ · σ² · (T - t)
δ^b = (1/γ) · ln(1 + γ/k) + ((1 + 2q)/2) · γ · σ² · (T - t)
```

### Bid and Ask Quotes
```
p^a = s + δ^a  (ask price)
p^b = s - δ^b  (bid price)
```

### Execution Intensity
```
λ(δ) = A · exp(-k · δ)
```

## Experiments Summary

### Experiment 1: Base Replication
- Compares inventory-based vs symmetric strategies
- 1000 Monte Carlo paths per γ value
- Validates risk reduction hypothesis

**Key Result:** Inventory strategy reduces P&L variance by 30-60% and inventory variance by 40-80%.

### Experiment 2: Ambiguity Resolution
- Tests formula-faithful vs table-faithful spread conventions
- Resolves paper's internal inconsistency

**Key Result:** Table-faithful (constant spread) matches paper tables better.

### Experiment 3: Execution Model
- Compares Bernoulli approximation vs exact Poisson fills
- Tests robustness to execution modeling choices

**Key Result:** Bernoulli approximation valid when λ·dt < 1; qualitative results robust.

### Experiment 4: Formula Validation
- Verifies analytical properties of optimal quotes
- Tests reservation price identity and spread formulas

**Key Result:** All formulas validated; implementation mathematically correct.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for comprehensive results including:

- Summary statistics tables
- Comparison with paper values
- Diagnostic plots
- Histograms of terminal P&L
- Execution model comparisons

## Reproducibility

- Random seed: 42 (fixed for all experiments)
- Python version: 3.12.13
- All dependencies pinned in requirements.txt

## License

MIT License

## Citation

```bibtex
@article{avellaneda2008high,
  title={High-frequency trading in a limit order book},
  author={Avellaneda, Marco and Stoikov, Sasha},
  journal={Quantitative Finance},
  volume={8},
  number={3},
  pages={217--224},
  year={2008},
  publisher={Taylor \& Francis}
}
```

## Contributing

Contributions welcome! Please ensure all tests pass before submitting changes.

```bash
pytest tests/ -v
```
