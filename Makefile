.PHONY: test test-fast report

# Full test: executes the walkthrough notebook (slow).
test:
	python -m pytest tests -q

# Quick: skip end-to-end notebook execution.
test-fast:
	set SKIP_NOTEBOOK_E2E=1 && python -m pytest tests -q

report:
	python scripts/generate_report.py
