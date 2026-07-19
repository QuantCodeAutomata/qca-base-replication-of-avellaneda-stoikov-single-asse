# Experiment 4: Deterministic Validation of Quote Formulas

## Summary

This experiment validates the Avellaneda-Stoikov quote formulas through deterministic testing across a comprehensive parameter grid.

## Test Grid

- **s (mid-price)**: {95, 100, 105}
- **q (inventory)**: {-10, -5, -1, 0, 1, 5, 10}
- **t (time)**: {0, 0.25, 0.5, 0.75, 0.995}
- **gamma (risk aversion)**: {0.01, 0.1, 0.5}

**Total test cases**: 3 × 7 × 5 × 3 = 315

## Validation Results

### Test 1: Reservation Price Formula
**Formula**: `r = s - q * γ * σ² * τ` where `τ = T - t`

**Result**: ✅ PASSED (315/315 tests)

The reservation price formula correctly adjusts the mid-price based on:
- Inventory position (q)
- Risk aversion (γ)
- Volatility (σ)
- Time remaining (τ)

### Test 2: Quote Center Equals Reservation Price
**Formula**: `center = (bid + ask) / 2 = r`

**Result**: ✅ PASSED (315/315 tests)

The quote center is always equal to the reservation price, confirming that the market maker centers quotes around their risk-adjusted valuation.

### Test 3: Spread Formula
**Formula**: `spread = δ_a + δ_b = (2/γ) * ln(1 + γ/k) + γ * σ² * τ`

**Result**: ✅ PASSED (315/315 tests)

The spread calculation matches the theoretical formula and is:
- Independent of inventory (q)
- Independent of mid-price (s)
- Dependent on time remaining (τ)
- Dependent on risk aversion (γ)

### Test 4: Inventory Effects on Quotes
**Result**: ✅ VERIFIED

Key observations:
- When q > 0 (long inventory): reservation price decreases
- When q < 0 (short inventory): reservation price increases
- Spread remains constant for fixed t and γ
- Bid and ask prices shift symmetrically around reservation price

## Validated Formulas

### 1. Reservation Price
```
r = s - q * γ * σ² * τ
```
where τ = T - t (time remaining)

**Interpretation**:
- When q > 0 (long inventory): r < s (lower reservation price to encourage selling)
- When q < 0 (short inventory): r > s (higher reservation price to encourage buying)
- Effect increases with γ, σ², and τ

### 2. Quote Center
```
center = (bid + ask) / 2 = r
```
The quote center equals the reservation price.

### 3. Half-Spread
```
δ = (1/γ) * ln(1 + γ/k) + (γ * σ² * τ) / 2
```
Distance from reservation price to bid/ask quotes.

### 4. Total Spread
```
spread = δ_a + δ_b = 2 * δ
spread = (2/γ) * ln(1 + γ/k) + γ * σ² * τ
```

**Interpretation**:
- Spread decreases as t → T (tau → 0)
- Higher γ increases spread (more risk aversion)
- Spread is independent of inventory q

### 5. Bid and Ask Prices
```
bid = r - δ
ask = r + δ
```

## Diagnostic Plots

### 1. exp_4_validation.png
Four-panel figure showing:
- **Top-left**: Quote Center vs Inventory for different γ values
- **Top-right**: Bid/Ask Distances vs Inventory
- **Bottom-left**: Spread vs Time (showing time decay)
- **Bottom-right**: 3D Reservation Price Surface

### 2. exp_4_prices_spread.png
Two-panel figure showing:
- **Left**: Bid, Ask, and Reservation Prices vs Inventory
- **Right**: Spread vs Inventory (demonstrating independence)

### 3. exp_4_spread_decay.png
Single panel showing:
- Time decay of spread for different risk aversion levels (γ)

## Key Findings

1. **All formulas are mathematically consistent** - The implementation correctly implements the Avellaneda-Stoikov model.

2. **Inventory management works as expected** - The reservation price adjusts linearly with inventory to manage risk.

3. **Spread behavior is correct** - Spread decreases over time and increases with risk aversion, but is independent of inventory.

4. **Quote symmetry** - Bid and ask quotes are symmetric around the reservation price.

## Model Parameters Used

- σ (volatility) = 0.01
- T (time horizon) = 1.0
- k (order arrival intensity) = 1.0
- γ (risk aversion) = {0.01, 0.1, 0.5}

## Conclusion

All quote formulas have been successfully validated. The Avellaneda-Stoikov model implementation is correct and ready for use in further experiments.
