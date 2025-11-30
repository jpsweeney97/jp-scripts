.PHONY: install test format clean lint

install:
	pip install -e ".[dev,ai]"

format:
	ruff format .
	ruff check --fix .

lint:
	ruff format --check .
	ruff check .
	mypy src

test: lint
	pytest --cov=src/jpscripts --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache *.egg-info
