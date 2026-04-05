"""
Build the Engineering Systems Design (ESD) style HTML report and run_summary.json.

Outputs:
  - outputs/report.html (paths relative to outputs/)
  - report.html at repo root (figure path: outputs/cost_by_scenario.png)
  - outputs/run_summary.json

Usage (from repo root):
  python scripts/generate_report.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEMPLATE_DIR = ROOT / "reports" / "templates"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    exp = load_yaml(ROOT / "config" / "experiment.yaml")
    costs = load_yaml(ROOT / "config" / "costs.yaml")

    kpi_path = OUT / "kpi_comparison.csv"
    sens_path = OUT / "sensitivity_naive_forecast.csv"
    manifest_path = OUT / "run_manifest.json"

    if not kpi_path.is_file():
        raise FileNotFoundError(f"Missing {kpi_path}. Run the notebook or pipeline first.")

    kpi = pd.read_csv(kpi_path)
    sensitivity = pd.read_csv(sens_path) if sens_path.is_file() else pd.DataFrame()
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    kpi_rows = kpi.to_dict(orient="records")
    sens_rows = sensitivity.to_dict(orient="records") if len(sensitivity) else []

    seed = exp.get("random_seed", "")
    n_skus = manifest.get("n_skus", exp.get("n_skus"))
    n_periods = manifest.get("n_periods", exp.get("n_periods"))
    cutoff = manifest.get("cutoff", "")

    oracle_cost = kpi.loc[kpi["scenario"].str.contains("oracle", case=False), "total_cost"]
    oracle_cost = float(oracle_cost.iloc[0]) if len(oracle_cost) else 0.0
    naive_cost = float(kpi.loc[kpi["scenario"].str.contains("naive", case=False), "total_cost"].iloc[0])
    ml_cost = float(kpi.loc[kpi["scenario"].str.contains("ml", case=False), "total_cost"].iloc[0])

    gap_naive = (naive_cost - oracle_cost) / oracle_cost * 100 if oracle_cost else 0.0
    gap_ml = (ml_cost - oracle_cost) / oracle_cost * 100 if oracle_cost else 0.0

    data_source = str(manifest.get("data_source", exp.get("data_source", "synthetic")))

    if gap_naive >= 0 and gap_ml >= 0:
        cost_bullet = (
            f"Oracle planning with perfect test demand in the LP ({oracle_cost:,.0f} realized cost) "
            f"is below forecast-driven plans; naive and ML are about "
            f"{gap_naive:+.1f}% and {gap_ml:+.1f}% vs oracle on realized total cost."
        )
    elif gap_naive <= 0 and gap_ml <= 0:
        cost_bullet = (
            f"Realized total cost is lower for naive ({naive_cost:,.0f}) and ML ({ml_cost:,.0f}) than "
            f"for the perfect-foresight LP plan ({oracle_cost:,.0f}), about "
            f"{gap_naive:.1f}% and {gap_ml:.1f}% vs oracle. The LP objective omits per-order fixed "
            "costs; the simulator adds them, so ranking by realized KPIs need not match the LP optimum."
        )
    else:
        cost_bullet = (
            f"Oracle realized cost {oracle_cost:,.0f}; naive {naive_cost:,.0f} ({gap_naive:+.1f}% vs oracle); "
            f"ML {ml_cost:,.0f} ({gap_ml:+.1f}% vs oracle). Mixed ranking reflects objective mismatch "
            "between LP and simulation."
        )

    executive_bullets = [
        cost_bullet,
        (
            "Realized fill rate can sit far below the nominal LP service target because the aggregate "
            "fill constraint binds on planning demand (forecast or truth in the LP), not on realized demand."
        ),
        (
            "Stress multipliers on true demand move total cost and fill rate without re-fitting "
            "forecasts, illustrating sensitivity of a fixed plan under scale misspecification."
        ),
    ]

    system_framing = (
        "Actors: a planner chooses periodic replenishment orders subject to a warehouse capacity. "
        "Customers draw true demand; the planner only sees history when fitting forecasts. "
        "Feedback is monetary: holding, variable ordering, fixed ordering (in simulation), and stockouts."
    )

    system_rows = [
        {
            "layer": "Prediction",
            "description": (
                "Seasonal naive and lag-feature gradient boosting produce test-horizon planning demand matrices."
            ),
        },
        {
            "layer": "Optimization",
            "description": (
                "A multi-SKU LP minimizes holding plus variable ordering plus shortage penalties with "
                "aggregate fill and per-period capacity."
            ),
        },
        {
            "layer": "Evaluation",
            "description": (
                "Oracle uses truth in the LP; forecast scenarios use d_hat in the LP, then simulate realized "
                "cost and fill under true demand with fixed orders."
            ),
        },
    ]

    if data_source == "uci_online_retail":
        methods_lines = [
            (
                "Data: UCI Online Retail (UK transactions), aggregated to weekly units per StockCode; "
                f"top SKUs by volume, encoded as {n_skus} SKUs × {n_periods} weekly periods."
            ),
            f"Train fraction {exp['train_period_fraction']:.0%}; test horizon from period {cutoff} onward.",
            f"LP service target: minimum aggregate fill rate ≥ {exp['min_fill_rate']:.0%} on planning demand.",
            "Solver: PuLP with bundled CBC (no commercial licenses). Fixed ordering charged in simulation only.",
        ]
    else:
        methods_lines = [
            (
                f"Synthetic DGP: per-SKU log-normal level, sinusoidal seasonality, Poisson demand "
                f"({n_skus} SKUs, {n_periods} periods)."
            ),
            f"Train fraction {exp['train_period_fraction']:.0%}; test horizon from period {cutoff} onward.",
            f"LP service target: minimum aggregate fill rate ≥ {exp['min_fill_rate']:.0%} on planning demand.",
            "Solver: PuLP with bundled CBC (no commercial licenses). Fixed ordering charged in simulation only.",
        ]

    if data_source == "uci_online_retail":
        subtitle = (
            "Predict-then-optimize inventory pipeline · UCI Online Retail weekly panel · realized KPIs under true demand"
        )
    else:
        subtitle = (
            "Predict-then-optimize inventory pipeline · synthetic retail panel · realized KPIs under true demand"
        )

    pto_caveat = (
        "Plug-in forecasts optimize a statistical loss that need not align with ordering cost. "
        "A stronger forecast can lose to a naive one on realized inventory cost in finite samples."
    )

    figure_path = OUT / "cost_by_scenario.png"
    figure_rel_outputs = "cost_by_scenario.png" if figure_path.is_file() else None
    figure_rel_root = "outputs/cost_by_scenario.png" if figure_path.is_file() else None

    tau = float(exp["min_fill_rate"])
    cap = costs.get("order_capacity_per_period", "")

    data_source_label = (
        "UCI Online Retail (weekly aggregate)" if data_source == "uci_online_retail" else "Synthetic Poisson panel"
    )

    cost_rows = [{"parameter": k, "value": str(v)} for k, v in costs.items()]

    stakeholder_rows = [
        {
            "stakeholder": "Planner / operations",
            "need": "Periodic replenishment under warehouse capacity; minimize holding, variable ordering, and shortage costs in the LP; evaluate with full cost stack in simulation.",
        },
        {
            "stakeholder": "Service policy owner",
            "need": f"Aggregate fill target on planning demand at least {tau:.0%} (configurable τ).",
        },
        {
            "stakeholder": "Analysis and audit",
            "need": "Documented LP, open-source CBC solver, single-notebook implementation for traceability.",
        },
    ]

    requirement_rows = [
        {
            "id": "FR-1",
            "requirement": "Multi-SKU, multi-period inventory balance with nonnegative orders, inventory, and shortages.",
            "metric": "Notebook `solve_inventory_lp` balance rows",
        },
        {
            "id": "FR-2",
            "requirement": f"Minimum aggregate fill on planning demand: total shortage ≤ (1−τ) × total planning demand, τ = {tau:.0%}.",
            "metric": "LP aggregate shortage constraint",
        },
        {
            "id": "FR-3",
            "requirement": f"Total order quantity per period ≤ {cap} units (all SKUs combined).",
            "metric": "LP capacity constraint per t",
        },
        {
            "id": "NFR-1",
            "requirement": "No commercial solver dependency.",
            "metric": "PuLP + CBC",
        },
    ]

    traceability_rows = [
        {"req": "FR-1", "artifact": "`solve_inventory_lp`: per (t,s) balance I_end = I_prev + x − d_plan + u; u ≤ d_plan."},
        {"req": "FR-2", "artifact": f"Single row: Σ u ≤ (1−{tau:.2f}) × Σ d_plan over test horizon."},
        {"req": "FR-3", "artifact": "Per period: Σ_s x[t,s] ≤ order_capacity_per_period."},
        {"req": "NFR-1", "artifact": "`pl.PULP_CBC_CMD` in notebook."},
    ]

    report_title = "Predict-then-optimize for multi-SKU inventory"

    problem_statement = (
        "Retail and distribution systems must place replenishment orders before demand is realized. "
        "Decisions are informed by forecasts, yet performance is judged on outcomes under true demand. "
        "This report documents a decomposed system: forecasting, linear optimization with service and capacity "
        "constraints, and ex-post simulation for realized cost and fill."
    )
    need_statement = (
        "The need is traceable evidence of how forecast-driven plans behave relative to a perfect-foresight "
        "benchmark, including sensitivity to demand scale, under explicit unit economics and capacity limits."
    )

    scope_lines = [
        "Single echelon: one planning pool with a shared per-period order capacity (no multi-node network).",
        "Continuous order quantities in the LP; no lot sizing or lead-time delays beyond the period index.",
        "Point forecasts only; no joint demand distribution in the optimization layer.",
        "Implementation lives in `optimization_ml_hybrid_walkthrough.ipynb`; config in `config/*.yaml`.",
    ]

    exec_purpose = (
        "Satisfy engineering review-style documentation: problem, stakeholders, requirements with traceability, "
        "architecture, detailed design (data, forecasts, LP, simulation), results, sensitivity, risks, and reproducibility."
    )

    architecture_intro = (
        "The system is decomposed into three subsystems. Information flows one way into the LP at decision time; "
        "truth enters only in the evaluation module (or inside the LP for the oracle benchmark)."
    )

    information_flow_diagram = (
        "  Training-period demand history\n"
        "           │\n"
        "           ▼\n"
        "  ┌────────────────────┐      planning demand d_plan\n"
        "  │ Forecast subsystem │  (d_hat from naive or ML, or d_true for oracle)\n"
        "  └─────────┬──────────┘\n"
        "            │\n"
        "            ▼\n"
        "  ┌────────────────────┐      orders x[t,s] ≥ 0\n"
        "  │  LP (CBC)        │      min Σ (h·I + c_var·x + c_short·u)\n"
        "  │  min cost        │      s.t. balance, capacity, aggregate fill on d_plan\n"
        "  └─────────┬────────┘\n"
        "            │\n"
        "            ▼\n"
        "  ┌────────────────────┐      true demand d_true (test horizon)\n"
        "  │ Simulation / KPIs  │      realized cost (incl. fixed order cost), fill rate\n"
        "  └────────────────────┘"
    )

    if data_source == "uci_online_retail":
        data_pipeline_paragraph = (
            "Real transaction data from the UCI Online Retail repository are downloaded once to `data/raw/`, "
            "read with `openpyxl`, filtered to positive quantities, and aggregated to calendar-week units sold per "
            "`StockCode`. The top SKUs by total volume are retained, weeks are sorted and indexed 0…T−1, and the "
            "panel is expressed in long form then pivoted to wide for modeling. Missing SKU-week pairs are treated "
            "as zero demand in the wide matrix."
        )
    else:
        data_pipeline_paragraph = (
            "Synthetic demand is generated in the notebook via a documented stochastic process (heterogeneous "
            "levels, seasonality, Poisson sampling). The same long-to-wide pipeline applies."
        )

    forecast_detail_lines = [
        (
            "**Seasonal naive:** for each SKU, demand at test period t is forecast using the same seasonal lag "
            "(e.g. week t−52) when available in training; otherwise the closest prior lag."
        ),
        (
            "**ML (HGB):** `HistGradientBoostingRegressor` on features including lags 1, 2, 7, 14 and an encoded "
            "SKU index; trained on training periods only; recursive one-step predictions over the test window."
        ),
    ]

    lp_intro = (
        "The optimization layer is a linear program over the test horizon indices present in the planning demand "
        "matrix. Indices t and s refer to period and SKU; d_plan[t,s] is the planning demand (forecast or truth)."
    )

    lp_rows = [
        {
            "element": "Decision variables",
            "spec": (
                "x[t,s] ≥ 0 order quantity; I_end[t,s] ≥ 0 end-of-period inventory; u[t,s] ≥ 0 shortage "
                "(unmet demand in the LP accounting)."
            ),
        },
        {
            "element": "Objective",
            "spec": (
                "Minimize Σ_{t,s} ( h·I_end + c_var·x + c_short·u ) using `holding_per_unit_period`, "
                "`ordering_variable_per_unit`, `stockout_penalty_per_unit` from config."
            ),
        },
        {
            "element": "Inventory balance",
            "spec": (
                "I_end[t,s] = I_prev + x[t,s] − d_plan[t,s] + u[t,s]; first period uses `initial_inventory` per SKU."
            ),
        },
        {
            "element": "Local shortage bound",
            "spec": "u[t,s] ≤ d_plan[t,s] for each (t,s).",
        },
        {
            "element": "Order capacity",
            "spec": "Σ_s x[t,s] ≤ order_capacity_per_period for each t.",
        },
        {
            "element": "Aggregate service",
            "spec": (
                f"Sum over all (t,s) of u ≤ (1−τ) × sum of d_plan with τ = {tau:.2f} "
                "(minimum fill on planning demand)."
            ),
        },
    ]

    lp_aggregate_fill = (
        "The aggregate constraint limits total **planned** shortage relative to total **planning** demand. "
        "It does not bound realized shortage against true demand; hence realized fill rate in §6 can differ sharply "
        "from τ when d_plan and d_true disagree."
    )

    c_fix = costs.get("ordering_fixed_per_order", 0)

    simulation_metric_rows = [
        {
            "metric": "Realized total cost",
            "definition": (
                "Roll forward with true demand: end-of-period holding at rate h, variable cost c_var per unit "
                f"ordered, stockout penalty c_short per unmet unit, plus fixed cost {c_fix} per (t,s) when x[t,s] > 0. "
                "Same orders as the LP output."
            ),
        },
        {
            "metric": "Realized fill rate",
            "definition": (
                "(Total true demand − total unmet demand) / total true demand over the simulated horizon, "
                "with unmet demand computed period by period from inventory and orders."
            ),
        },
    ]

    oracle_fr = float(kpi.loc[kpi["scenario"].str.contains("oracle", case=False), "fill_rate"].iloc[0])
    naive_fr = float(kpi.loc[kpi["scenario"].str.contains("naive", case=False), "fill_rate"].iloc[0])
    ml_fr = float(kpi.loc[kpi["scenario"].str.contains("ml", case=False), "fill_rate"].iloc[0])

    results_interpretation = (
        f"Numerical results: oracle realized cost {oracle_cost:,.2f} (fill {oracle_fr:.4f}); "
        f"naive {naive_cost:,.2f} ({gap_naive:+.2f}% vs oracle, fill {naive_fr:.4f}); "
        f"ML {ml_cost:,.2f} ({gap_ml:+.2f}% vs oracle, fill {ml_fr:.4f}). "
    )
    if gap_naive <= 0 and gap_ml <= 0:
        results_interpretation += (
            "Both forecast-driven plans achieve lower **simulated** total cost than the oracle in this build. "
            "Interpret this as an objective mismatch: the LP minimizes linear holding, variable ordering, and "
            "shortage costs without fixed ordering charges, while the simulator adds fixed costs per placement. "
            "The oracle LP is still optimal for its stated LP objective on planning demand; it is not guaranteed "
            "to minimize the simulator’s objective."
        )
    elif gap_naive >= 0 and gap_ml >= 0:
        results_interpretation += (
            "Forecast-driven plans exceed oracle cost on realized KPIs in this build, which is the typical "
            "ordering when the simulation cost structure aligns more closely with the LP tradeoffs."
        )
    else:
        results_interpretation += (
            "Mixed ranking between naive and ML relative to oracle indicates scenario-specific interaction "
            "between forecast shape, capacity, and penalties."
        )
    results_interpretation += (
        f" Realized fill rates remain well below τ={tau:.0%} because τ constrains the LP on planning demand, "
        "not on out-of-sample truth."
    )

    sensitivity_intro = (
        "A stress test scales **true** demand by a multiplier while holding the order plan from the naive "
        "forecast LP at multiplier 1.0. This isolates exposure to demand level misspecification without re-solving "
        "the optimization or re-estimating forecasts."
    )

    limitations_intro = (
        "The following items are limitations of the current model and study design, stated as risks with "
        "mitigations or research extensions."
    )

    risk_rows = [
        {
            "risk": "LP omits binary fixed-order decisions; fixed costs appear only in simulation.",
            "mitigation": "Compare KPIs under simulation consistently; consider MIP or fixed-cost proxies if decisions hinge on order counts.",
        },
        {
            "risk": "Single temporal train or test split.",
            "mitigation": "Rolling-origin evaluation across multiple cutoffs; bootstrap by SKU where appropriate.",
        },
        {
            "risk": "Recursive point forecasts; no uncertainty sets in the LP.",
            "mitigation": "Scenario trees, robust optimization, or chance constraints as follow-on work.",
        },
        {
            "risk": "Real data are aggregated and cleaned; not a live retailer feed.",
            "mitigation": "Treat as illustrative; validate on domain-specific data before operational use.",
        },
    ]

    conclusions = (
        "The documented system separates forecasting, LP-based planning, and simulation-based evaluation. "
        "Under the committed parameters, realized cost and fill illustrate predict-then-optimize effects: "
        "planning service level does not translate directly to realized fill, and LP-optimal plans need not "
        "minimize simulated cost when the simulator adds cost terms absent from the LP."
    )

    recommendation_lines = [
        "Align the optimization objective with the KPI simulator when fixed ordering drives behavior, or move fixed costs into the decision model.",
        "Separate forecast model selection from decision quality: use decision-aware losses or cost-weighted validation when the goal is cost minimization.",
        "Extend evaluation with multiple splits and, where data allow, hierarchical or stochastic demand models.",
    ]

    reproducibility = (
        "Create a venv, <code>pip install -r requirements.txt</code>, run all cells in "
        "<code>optimization_ml_hybrid_walkthrough.ipynb</code>, then run this script. "
        "Figures and CSVs under <code>outputs/</code> must exist for a full regeneration."
    )

    ctx = {
        "subtitle": subtitle,
        "executive_bullets": executive_bullets,
        "system_framing": system_framing,
        "system_rows": system_rows,
        "methods_lines": methods_lines,
        "pto_caveat": pto_caveat,
        "kpi_rows": kpi_rows,
        "sensitivity_rows": sens_rows,
        "reproducibility": reproducibility,
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "seed": seed,
        "n_skus": n_skus,
        "n_periods": n_periods,
        "cutoff": cutoff,
        "document_label": "Engineering Systems Design Report",
        "report_title": report_title,
        "data_source_label": data_source_label,
        "cost_rows": cost_rows,
        "stakeholder_rows": stakeholder_rows,
        "requirement_rows": requirement_rows,
        "traceability_rows": traceability_rows,
        "problem_statement": problem_statement,
        "need_statement": need_statement,
        "scope_lines": scope_lines,
        "exec_purpose": exec_purpose,
        "architecture_intro": architecture_intro,
        "information_flow_diagram": information_flow_diagram,
        "data_pipeline_paragraph": data_pipeline_paragraph,
        "forecast_detail_lines": forecast_detail_lines,
        "lp_intro": lp_intro,
        "lp_rows": lp_rows,
        "lp_aggregate_fill": lp_aggregate_fill,
        "simulation_metric_rows": simulation_metric_rows,
        "results_interpretation": results_interpretation,
        "sensitivity_intro": sensitivity_intro,
        "limitations_intro": limitations_intro,
        "risk_rows": risk_rows,
        "conclusions": conclusions,
        "recommendation_lines": recommendation_lines,
    }

    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    html_outputs = template.render(**{**ctx, "figure_rel": figure_rel_outputs})
    html_root = template.render(**{**ctx, "figure_rel": figure_rel_root})

    out_report = OUT / "report.html"
    out_report.write_text(html_outputs, encoding="utf-8")
    print("Wrote", out_report)

    root_report = ROOT / "report.html"
    root_report.write_text(html_root, encoding="utf-8")
    print("Wrote", root_report, "(figure path: outputs/cost_by_scenario.png)")

    run_summary = {
        "project": "optimization-ml-hybrid",
        "generated_at": ctx["generated_at"],
        "experiment": exp,
        "costs_keys": list(costs.keys()),
        "kpi_comparison": kpi_rows,
        "sensitivity_naive_forecast": sens_rows,
        "manifest": manifest,
        "gaps_vs_oracle_pct": {
            "naive": round(gap_naive, 3),
            "ml": round(gap_ml, 3),
        },
    }
    (OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print("Wrote", OUT / "run_summary.json")


if __name__ == "__main__":
    main()
