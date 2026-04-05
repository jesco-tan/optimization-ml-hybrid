.PHONY: test test-fast

# Full test: executes the walkthrough notebook (slow).
test:
	python -m pytest tests -q

# Quick: skip end-to-end notebook execution.
test-fast:
	set SKIP_NOTEBOOK_E2E=1 && python -m pytest tests -q

# HTML report: run Section 9 in optimization_ml_hybrid_walkthrough.ipynb (no separate script).
