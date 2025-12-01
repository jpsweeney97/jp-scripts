.PHONY: install test format clean lint docs

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

docs:
	pdoc --html --output-dir docs/api src/jpscripts --force
	@echo "API documentation generated in docs/api/"

docs-serve:
	pdoc src/jpscripts --http localhost:8080

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache *.egg-info docs/api
