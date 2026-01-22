.PHONY: test test-cov test-fast test-verbose test-notebooks test-notebooks-fast clean

# Run tests without coverage enforcement (for TDD/VS Code)
test:
	pytest tests/

# Run tests with coverage enforcement (for CI/CD)
test-cov:
	pytest tests/ --cov-fail-under=75

# Run tests without coverage reporting (fastest)
test-fast:
	pytest tests/ --no-cov

# Run tests with verbose output
test-verbose:
	pytest tests/ -vv

# Run specific test file
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=tests/path/to/test.py"; \
		exit 1; \
	fi
	pytest $(FILE)

# Clean coverage and cache files
clean:
	rm -rf .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Show coverage report
coverage-report:
	coverage report -m

# Generate HTML coverage report
coverage-html:
	coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Run notebook tests (all exploration notebooks)
test-notebooks:
	python scripts/test_notebooks.py

# Run notebook tests (skip slow notebooks 07, 09)
test-notebooks-fast:
	python scripts/test_notebooks.py --fast

# Run specific notebook tests by number
# Usage: make test-notebook NB=01  or  make test-notebook NB="01 02 03"
test-notebook:
	@if [ -z "$(NB)" ]; then \
		echo "Usage: make test-notebook NB=01  or  NB=\"01 02 03\""; \
		exit 1; \
	fi
	python scripts/test_notebooks.py $(NB)

# Run notebook tests via pytest (slower but integrated with test suite)
test-notebooks-pytest:
	pytest tests/test_notebooks.py -v --no-cov
