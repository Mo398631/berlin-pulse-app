# Berlin Pulse

A Streamlit web app accompanying the research paper:

> **Berlin Pulse: A Case Study of AI, Urban Mobility, and Energy-Aware Transport Policy**
> Available on SSRN: [ssrn.com/abstract=6974299](https://ssrn.com/abstract=6974299)

The app simulates carbon-optimal overnight charging for Berlin's electric bus
fleet using real SMARD grid data (2025), and exposes the paper's analysis
through six interactive tabs.

All figures are illustrative simulations with synthetic demand — no operational
system exists.

## Tabs

1. **Depot Optimizer** — Compares naive chronological charging (Strategy A)
   against carbon-optimal merit-order charging (Strategy B), in both an *oracle*
   (perfect-foresight) and a *deployable* (fixed day-ahead ranking) mode.
2. **Scenario Explorer** — Interactive version of the paper's Section 6
   congestion scenarios, sweeping the passenger participation rate ρ.
3. **Deployability Gap** — Visualises Appendix B (Table B.2): how much of the
   perfect-foresight saving survives once the operator is restricted to a fixed,
   day-ahead-only ranking, split by channel.
4. **Robustness (Monte Carlo)** — Appendix D, Pillar One: a 10,000-draw
   emission-factor Monte Carlo (fixed seed) on the carbon saving, re-running
   Strategy A vs B over the 364 complete nights.
5. **Unified Model** — A conceptual illustration of Appendix A, the
   social-planner optimization: welfare W splits into a separable passenger
   (congestion) block and bus (energy) block.
6. **Network Prototype** — Map view of the Section 6 redirection prototype:
   spreads the network-wide peak-shift across Berlin's named arterial corridors
   so you can see *where* the relief lands, with Before/After toggle and
   per-corridor metrics.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

The test suite covers the simulation engines behind each tab:

- `test_engine.py` — Depot Optimizer engine
- `test_scenario.py` — Scenario Explorer engine
- `test_deployability.py` — Deployability Gap (gate) results
- `test_montecarlo.py` — Monte Carlo emission-factor engine
- `test_unified.py` — Unified Model welfare/optimization
- `test_prototype.py` — Network Prototype redirection + reconciliation

Run them all with:

```bash
python -m pytest
```

## Data attribution

Street data © OpenStreetMap contributors (ODbL); transit data © VBB
Verkehrsverbund Berlin-Brandenburg GmbH (CC BY).

## License

MIT — see [LICENSE](LICENSE).
