.PHONY: install test format clean

install:
	pip install -e ".[dev]"

test:
	pytest

format:
	@if command -v ruff >/dev/null 2>&1; then \
		echo "Running ruff --fix ..."; \
		ruff check --fix .; \
	else \
		echo "ruff not installed; skipping format"; \
	fi

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache *.egg-info
