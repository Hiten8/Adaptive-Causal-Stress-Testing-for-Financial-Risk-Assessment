# Presentation samples

This folder contains isolated, synthetic scenarios for demonstrating the dashboard.

`market_stress_graph_change.json` describes a market-stress scenario with three intentional causal changes:

- `GDP -> Liquidity` is removed.
- `CreditSpread -> Volatility` is added.
- The strength of `Inflation -> CreditSpread` rises from `0.81` to `0.94`.

The dashboard can load this scenario from its sidebar. It is presentation-only and does not overwrite anything under `data/` or `results/`.
