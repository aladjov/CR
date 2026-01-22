# Installation Guide

This guide will help you set up the Customer Retention Framework development environment.

## Prerequisites

- **Python 3.11+** installed on your system
- **uv** package manager installed

### Installing uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For other installation methods, visit: https://docs.astral.sh/uv/getting-started/installation/

## Quick Setup

### macOS/Linux

```bash
./install.sh
```

### Windows

```cmd
install.bat
```

The installation script will:
1. ✓ Check for uv installation
2. ✓ Create a virtual environment (`.venv`)
3. ✓ Activate the virtual environment
4. ✓ Install all dependencies using `uv sync`
5. ✓ Verify pytest installation
6. ✓ Install the package in editable mode
7. ✓ Run verification test

## Manual Setup

If you prefer to set up manually:

```bash
# Sync project (creates .venv and installs dependencies)
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate.bat  # Windows

# Verify installation
python -c "import customer_retention"
```

## Running Tests

### All Tests
```bash
python -m pytest tests/
```

### With Coverage Report
```bash
python -m pytest tests/ --cov=src/customer_retention --cov-report=term-missing
```

### Specific Test Module
```bash
# Type detector tests
python -m pytest tests/profiling/test_type_detector.py -v

# Quality checks tests
python -m pytest tests/profiling/test_quality_checks.py -v

# Feature quality gate tests
python -m pytest tests/validation/test_feature_quality_gate.py -v

# Integration tests
python -m pytest tests/integration/test_profiling_retail.py -v
```

### Test Statistics
- **Total Tests:** 3947
- **Coverage:** 83%+
- **Test Modules:** 100+

## Project Structure

```
CustomerRetention/
├── src/customer_retention/      # Core library
│   ├── analysis/                # Data analysis (auto_explorer, visualization)
│   ├── stages/                  # Pipeline stages (temporal, profiling, validation)
│   ├── generators/              # Code generation (notebook_generator)
│   ├── core/                    # Core abstractions (config, components)
│   └── feature_store/           # Temporal-aware feature store
├── exploration_notebooks/       # Interactive exploration notebooks
├── generated_pipelines/         # Auto-generated pipeline notebooks
├── experiments/                 # All outputs (gitignored)
│   ├── findings/                # Exploration findings (YAML)
│   ├── data/                    # Pipeline data (bronze/silver/gold)
│   ├── mlruns/                  # MLflow tracking
│   └── feature_store/           # Feast feature store
├── scripts/                     # Utility scripts
│   ├── databricks/              # Databricks deployment
│   ├── data/                    # Data generation/migration
│   └── notebooks/               # Notebook utilities
├── tests/                       # Test suite
│   ├── .coverage                # Coverage data
│   └── htmlcov/                 # Coverage HTML reports
├── install.sh                   # Unix installation script
├── install.bat                  # Windows installation script
└── pyproject.toml               # Project dependencies
```

## Troubleshooting

### Virtual Environment Issues

If you encounter virtual environment issues:

```bash
# Remove existing virtual environment
rm -rf .venv

# Re-run installation
./install.sh
```

### Import Errors

If you see `ModuleNotFoundError: No module named 'customer_retention'`:

```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall package in editable mode
uv pip install -e .
```

### Coverage Failures

If coverage is below 75%:

```bash
# Run full test suite (coverage is measured across all tests)
python -m pytest tests/ --cov=src/customer_retention
```

Individual test files may show lower coverage - this is expected.

## Development Workflow

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install pre-commit hooks (required):**
   ```bash
   pre-commit install --install-hooks
   pre-commit install --hook-type pre-push
   ```

3. **Make changes to code**

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Check coverage (must be >= 75%):**
   ```bash
   python -m pytest tests/ --cov=src/customer_retention --cov-report=html
   open tests/htmlcov/index.html
   ```

6. **Commit (pre-commit hooks run automatically):**
   ```bash
   git add .
   git commit -m "Your message"
   ```

7. **Push (coverage check runs automatically):**
   ```bash
   git push  # Coverage must be >= 75% to push
   ```

8. **Deactivate when done:**
   ```bash
   deactivate
   ```

## Pre-commit Hooks

This project enforces coding standards via pre-commit hooks:

| Hook | Stage | Purpose |
|------|-------|---------|
| ruff | pre-commit | Linting and formatting |
| pytest-coverage | pre-push | Enforces 75% minimum coverage |
| python-check | pre-commit | Syntax validation |

**Important:** The `--no-verify` flag should NOT be used. CI will reject PRs that bypass these checks.

## Additional Commands

### Update Dependencies
```bash
uv sync --upgrade
```

### Add New Dependency
```bash
uv add <package-name>
```

### Add Development Dependency
```bash
uv add --dev <package-name>
```

### Run Specific Test Pattern
```bash
python -m pytest tests/ -k "test_quality"
```

### Run Tests in Parallel
```bash
python -m pytest tests/ -n auto
```

## Support

For issues or questions:
- Check the test output for specific error messages
- Review the pytest documentation: https://docs.pytest.org/
- Review the uv documentation: https://docs.astral.sh/uv/
