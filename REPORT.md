# Engineering Systems Design Report: Predict-then-optimize for multi-SKU inventory

This file is the **Markdown twin** of **`report.html`** / **`outputs/report.html`**. Read it here if you do not want to open HTML. Numbers in §6 and §7 match the **committed** `outputs/*.csv` at the time of writing; re-run the notebook and `python scripts/generate_report.py` to refresh HTML, JSON, and then update this file if you need parity after a new run.

**Document metadata (committed snapshot)**

| Field | Value |
|-------|--------|
| Data source | `uci_online_retail` (see `outputs/run_manifest.json`) |
| Experiment seed | `42` (`config/experiment.yaml`) |
| Panel | 48 SKUs × 53 weekly periods · test from period **39** |
| Service target τ | 95% aggregate fill on **planning** demand (`min_fill_rate`) |
| Artifacts | `outputs/kpi_comparison.csv`, `outputs/sensitivity_naive_forecast.csv`, `outputs/run_manifest.json`, `config/*.yaml` |

---

## 0. Executive summary

**Purpose.** Satisfy engineering-style documentation: problem, stakeholders, requirements with traceability, architecture, detailed design (data, forecasts, LP, simulation), results, sensitivity, risks, and reproducibility.

**Outcomes (this build, seed 42, UCI Online Retail).**

- Realized total cost is **lower** for naive (3,271,715) and ML (3,274,648) than for the perfect-foresight LP plan (3,328,858), about **−1.7%** vs oracle on both forecasts. The LP objective omits per-order fixed costs; the simulator adds them, so ranking by realized KPIs need not match the LP optimum.
- Realized fill rate can sit far below the nominal 95% LP service target because the aggregate fill constraint binds on **planning** demand (forecast or truth in the LP), not on realized demand.
- Stress multipliers on true demand move total cost and fill rate without re-fitting forecasts, illustrating sensitivity of a fixed plan under scale misspecification.

---

## 1. Problem definition, scope, and need

Retail and distribution systems must place replenishment orders before demand is realized. Decisions are informed by forecasts, yet performance is judged on outcomes under true demand. This report documents a decomposed system: forecasting, linear optimization with service and capacity constraints, and ex-post simulation for realized cost and fill.

The need is traceable evidence of how forecast-driven plans behave relative to a perfect-foresight benchmark, including sensitivity to demand scale, under explicit unit economics and capacity limits.

**System boundary**

- Single echelon: one planning pool with shared per-period order capacity (no multi-node network).
- Continuous order quantities in the LP; no lot sizing or lead-time delays beyond the period index.
- Point forecasts only; no joint demand distribution in the optimization layer.
- Implementation lives in `optimization_ml_hybrid_walkthrough.ipynb`; parameters in `config/*.yaml`.

---

## 2. Stakeholders and operational context

**Actors:** a planner chooses periodic replenishment subject to warehouse capacity. Customers draw true demand; the planner only sees history when fitting forecasts. Feedback is monetary: holding, variable ordering, fixed ordering (in simulation), and stockouts.

| Stakeholder / role | Need |
|--------------------|------|
| Planner / operations | Periodic replenishment under capacity; minimize holding, variable ordering, and shortage costs in the LP; evaluate with full cost stack in simulation. |
| Service policy owner | Aggregate fill target on planning demand at least τ (configurable). |
| Analysis and audit | Documented LP, open-source CBC solver, single-notebook traceability. |

---

## 3. Requirements, parameters, and traceability

### 3.1 Functional and non-functional requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| FR-1 | Multi-SKU, multi-period balance with nonnegative orders, inventory, and shortages. | Notebook `solve_inventory_lp` balance rows |
| FR-2 | Minimum aggregate fill on planning demand: total shortage ≤ (1−τ) × total planning demand, τ = 95%. | LP aggregate shortage constraint |
| FR-3 | Total order quantity per period ≤ 800 units (all SKUs). | `costs.yaml` · LP capacity per t |
| NFR-1 | No commercial solver. | PuLP + CBC |

### 3.2 Traceability (requirement → design artifact)

| Requirement | Design element in implementation |
|-------------|----------------------------------|
| FR-1 | `solve_inventory_lp`: per (t,s) balance I_end = I_prev + x − d_plan + u; u ≤ d_plan. |
| FR-2 | Single row: Σ u ≤ (1−τ) × Σ d_plan over the horizon. |
| FR-3 | Per period: Σ_s x[t,s] ≤ `order_capacity_per_period`. |
| NFR-1 | `pl.PULP_CBC_CMD` in notebook. |

### 3.3 Economic and capacity parameters (`config/costs.yaml`)

| Parameter | Value |
|-----------|-------|
| `holding_per_unit_period` | 0.25 |
| `ordering_variable_per_unit` | 1.0 |
| `ordering_fixed_per_order` | 15.0 |
| `stockout_penalty_per_unit` | 8.0 |
| `order_capacity_per_period` | 800 |
| `initial_inventory` | 5.0 |

---

## 4. System architecture and information flow

The system is decomposed into three subsystems: **prediction**, **optimization**, and **evaluation**. Information flows into the LP at decision time; truth enters the evaluation module (or the LP for the oracle benchmark).

| Subsystem | Function |
|-----------|----------|
| Prediction | Seasonal naive and lag-feature gradient boosting produce test-horizon planning demand matrices. |
| Optimization | Multi-SKU LP minimizes holding plus variable ordering plus shortage penalties with aggregate fill and per-period capacity. |
| Evaluation | Oracle uses truth in the LP; forecast scenarios use d_hat in the LP, then simulate realized cost and fill under true demand with fixed orders. |

**Information and decision flow**

```text
  Training-period demand history
           │
           ▼
  ┌────────────────────┐      planning demand d_plan
  │ Forecast subsystem │  (d_hat from naive or ML, or d_true for oracle)
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐      orders x[t,s] ≥ 0
  │  LP (CBC)          │      min Σ (h·I + c_var·x + c_short·u)
  │  min cost          │      s.t. balance, capacity, aggregate fill on d_plan
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐      true demand d_true (test horizon)
  │ Simulation / KPIs  │      realized cost (incl. fixed order cost), fill rate
  └────────────────────┘
```

---

## 5. Design description

### 5.1 Data pipeline

Real transaction data from the [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) repository are downloaded once to `data/raw/`, read with `openpyxl`, filtered to positive quantities, and aggregated to calendar-week units sold per `StockCode`. The top SKUs by total volume are retained, weeks are sorted and indexed 0…T−1, and the panel is expressed in long form then pivoted to wide. Missing SKU-week pairs are zero in the wide matrix.

**Synthetic alternative:** set `data_source: synthetic` in `config/experiment.yaml` for the Poisson simulator in the notebook.

### 5.2 Forecasting layer

Both methods produce a nonnegative matrix of planning demand on the test horizon.

- **Seasonal naive:** for each SKU, forecast at test period t uses the same seasonal lag (e.g. week t−52) when available in training; otherwise the closest prior lag.
- **ML (HGB):** `HistGradientBoostingRegressor` on lags 1, 2, 7, 14 and an encoded SKU index; trained on training periods only; recursive one-step predictions over the test window.

### 5.3 Optimization layer (linear program)

The LP is defined over the test horizon in the planning demand matrix. Let d_plan[t,s] be planning demand (forecast or truth).

| Element | Specification (implementation) |
|---------|-------------------------------|
| Decision variables | x[t,s] ≥ 0 orders; I_end[t,s] ≥ 0 end inventory; u[t,s] ≥ 0 shortage. |
| Objective | Minimize Σ (h·I_end + c_var·x + c_short·u) over periods and SKUs. |
| Inventory balance | I_end[t,s] = I_prev + x[t,s] − d_plan[t,s] + u[t,s]; first period uses `initial_inventory` per SKU. |
| Local shortage bound | u[t,s] ≤ d_plan[t,s] for each (t,s). |
| Order capacity | Σ_s x[t,s] ≤ 800 for each t. |
| Aggregate service | Sum of u ≤ (1−τ) × sum of d_plan with τ = 0.95 (minimum fill on **planning** demand). |

The aggregate constraint limits **planned** shortage relative to **planning** demand. It does not bound realized shortage against true demand, so realized fill in §6 can differ sharply from τ when d_plan and d_true disagree.

**Solver:** [PuLP](https://github.com/coin-or/pulp) with **CBC**. Fixed ordering is **not** in the LP (would need binaries).

### 5.4 Evaluation layer (simulation and metrics)

Given fixed orders from the LP, the simulator steps through periods with **true** demand and accumulates costs. Fixed ordering cost per positive order is included here but not in the LP objective.

| Metric | Definition |
|--------|------------|
| Realized total cost | Roll forward with true demand: holding, variable order cost, stockout penalty, plus **15.0** fixed cost per (t,s) when x[t,s] > 0. Same orders as the LP output. |
| Realized fill rate | (Total true demand − total unmet demand) / total true demand over the simulated horizon. |

### 5.5 Methods snapshot (config echo)

- Data: UCI Online Retail, top 48 SKUs by volume, 53 weekly periods (this run).
- Train fraction 75%; test from period 39 onward.
- LP service target: aggregate fill ≥ 95% on planning demand.
- Solver: PuLP + CBC; fixed ordering charged in simulation only.

**Predict-then-optimize caveat.** Plug-in forecasts optimize a statistical loss that need not align with ordering cost. A stronger forecast can lose to a naive one on realized inventory cost in finite samples.

---

## 6. Analysis, results, and interpretation

Orders come from the LP on **planning** demand; KPIs below are simulated against **true** test demand.

| Scenario | Realized total cost | Realized fill rate |
|----------|---------------------|---------------------|
| oracle_perfect_demand | 3,328,858.25 | 0.0270 |
| naive_seasonal | 3,271,715.00 | 0.0422 |
| ml_hgb_lags | 3,274,647.89 | 0.0400 |

**Interpretation.** Oracle realized cost exceeds naive and ML by about **+1.72%** and **+1.63%** respectively (naive and ML are cheaper than oracle on simulated total cost). This reflects **objective mismatch**: the LP minimizes linear holding, variable ordering, and shortage costs **without** fixed ordering charges, while the simulator adds fixed cost per placement. The oracle LP remains optimal for its stated objective on planning demand; it is not guaranteed to minimize the simulator’s objective. Realized fill rates stay well below τ = 95% because τ constrains the LP on planning demand, not on out-of-sample truth.

**Naive vs ML:** ML is slightly more expensive than naive on this split (~0.09% higher cost), consistent with decision loss differing from forecast accuracy.

**Figure:** `outputs/cost_by_scenario.png` bar chart of realized cost by scenario.

---

## 7. Sensitivity analysis

True demand is scaled by a multiplier while the order plan stays fixed from the naive-forecast LP at multiplier 1.0. Forecasts are not re-fit.

| Demand multiplier | Realized total cost | Realized fill rate |
|--------------------|---------------------|---------------------|
| 0.9 | 2,945,405.70 | 0.0434 |
| 1.0 | 3,271,715.00 | 0.0422 |
| 1.1 | 3,599,179.22 | 0.0410 |

---

## 8. Limitations and risk register

| Risk or limitation | Mitigation or extension |
|--------------------|-------------------------|
| LP omits binary fixed-order decisions; fixed costs appear only in simulation. | Compare KPIs under simulation consistently; consider MIP or fixed-cost proxies if decisions hinge on order counts. |
| Single temporal train or test split. | Rolling-origin evaluation; bootstrap by SKU where appropriate. |
| Recursive point forecasts; no uncertainty sets in the LP. | Scenario trees, robust optimization, or chance constraints. |
| Real data are aggregated and cleaned; not a live retailer feed. | Treat as illustrative; validate on domain data before operational use. |

---

## 9. Conclusions and recommendations

The pipeline separates forecasting, LP-based planning, and simulation-based evaluation. Under the committed parameters, realized cost and fill illustrate predict-then-optimize effects: planning service level does not translate directly to realized fill, and LP-optimal plans need not minimize simulated cost when the simulator adds terms absent from the LP.

**Recommendations.**

1. Align the optimization objective with the KPI simulator when fixed ordering drives behavior, or move fixed costs into the decision model (e.g., MIP).
2. Separate forecast model selection from decision quality: use decision-aware losses or cost-weighted validation when the goal is cost minimization.
3. Extend evaluation with multiple splits and, where data allow, hierarchical or stochastic demand models.

---

## 10. Reproducibility

1. `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt` (includes `openpyxl` for UCI Excel).
3. From the repo root, run **`optimization_ml_hybrid_walkthrough.ipynb`** top to bottom.
4. Run **`python scripts/generate_report.py`** to refresh `report.html`, `outputs/report.html`, and `outputs/run_summary.json`.

**Optional:** `make report` if you use `make`. For CI without the full notebook, `SKIP_NOTEBOOK_E2E=1` and `pytest` where applicable.

---

## Appendix A. Why not end-to-end reinforcement learning?

Reinforcement learning fits when a simulator is cheap and actions are stable. Here the emphasis is on modular assumptions (linear constraints, explicit costs) that a reviewer can question. RL remains a natural extension once dynamics and action space are agreed.

---

## Appendix B. File map

| Path | Role |
|------|------|
| `README.md` | Quick orientation. |
| **`REPORT.md`** (this file) | Same story as `report.html`, readable without downloading HTML. |
| **`report.html`** / **`outputs/report.html`** | Styled ESD report (figures embedded by path). |
| **`outputs/run_summary.json`** | Machine-readable KPIs and config echo. |
| `optimization_ml_hybrid_walkthrough.ipynb` | Only implementation source; writes `outputs/*.csv`, PNG, manifest. |

---

## Appendix C. References (orientation)

- Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*.
- PuLP and CBC documentation.
- Predict-then-optimize and decision-focused learning literature (forecast loss ≠ decision loss).

---

*Aligned with committed `outputs/` and `config/` for `uci_online_retail`. After re-running the pipeline, refresh HTML/JSON and edit §6–§7 tables here if you need Markdown to match new CSVs.*
