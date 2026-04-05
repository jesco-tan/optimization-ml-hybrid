# optimization-ml-hybrid

Predict-then-optimize study: **forecasts feed a constrained inventory LP**, then orders are evaluated under **true** demand.

## Source of truth

**`optimization_ml_hybrid_walkthrough.ipynb`** at the repo root is the **only** place implementation logic lives. Edit and run that notebook. There is no parallel `src/` or `omh/` package to keep in sync.

Supporting pieces (not duplicate logic):

- `config/*.yaml`: costs and experiment knobs the notebook reads.
- `scripts/generate_report.py`: builds **`report.html`** (Engineering Systems Design style: requirements, traceability, LP design, risk register) and `outputs/run_summary.json` from CSVs the notebook writes.

## Documentation (pick your format)

| What | Who it is for |
|------|----------------|
| **[REPORT.md](REPORT.md)** | Same ESD content as `report.html` in Markdown (offline, no HTML needed). |
| **[report.html](report.html)** | **Single ESD-style report** (open in a browser): problem, stakeholders, requirements + traceability, architecture, detailed design, results, sensitivity, risk register. |
| **[outputs/report.html](outputs/report.html)** | Same report with image paths relative to `outputs/`. |
| **[outputs/run_summary.json](outputs/run_summary.json)** | Machine-readable KPIs. |
| **`optimization_ml_hybrid_walkthrough.ipynb`** | **Run this** for the full pipeline. |

After changing config or re-running the notebook:

```text
python scripts/generate_report.py
```

Commit updated `outputs/*` and `report.html` when you want a frozen snapshot for visitors.

## How to run (local)

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start Jupyter from the **repository root**, open **`optimization_ml_hybrid_walkthrough.ipynb`**, choose the `.venv` kernel, **Run All**.

Optional: refresh HTML after a successful run:

```text
python scripts/generate_report.py
```

## Tests

The suite **executes the walkthrough notebook** (slow, a few minutes). Skip when iterating:

```text
set SKIP_NOTEBOOK_E2E=1
python -m pytest tests -q
```

Full check:

```text
python -m pytest tests -q
```

## Why not end-to-end RL?

Reinforcement learning shines when a cheap, faithful simulator exists and exploration is safe. This repo optimizes for **auditability**: linear constraints, modular `predict → optimize → simulate`, and plots a reviewer can question. RL is a natural extension when the simulator is trusted.

## Optimization stack (no paid solvers)

The planning model is a **linear program** (PuLP + bundled CBC). Fixed ordering costs in simulation are documented in `REPORT.md`.

## Predict-then-optimize caveat

Realized fill and cost use **true** demand; the LP service rule binds on **forecast** demand. See `REPORT.md`.

## Runtime

Default config (`n_skus=48`, `n_periods=104`) is on the order of tens of seconds per full notebook run, depending on hardware.
