# Contributing to Customer Retention Framework

Thank you for your interest in contributing! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the community and project
- Accept constructive criticism gracefully

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or personal attacks
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Git

### Quick Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/customer-retention.git
cd customer-retention

# Run the install script
./install.sh

# Or manually install
uv pip install -e ".[dev,ml]"

# Verify installation
pytest tests/ -v --tb=short
```

---

## Development Setup

### Virtual Environment

We use `uv` for fast, reliable dependency management:

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install all dependencies
uv pip install -e ".[dev,ml]"
```

### IDE Configuration

**VS Code** (recommended settings in `.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

**PyCharm**: Mark `src/` as Sources Root.

---

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Check existing [issues](https://github.com/yourusername/customer-retention/issues) to avoid duplicates
2. Collect relevant information:
   - Python version (`python --version`)
   - Package version (`pip show customer-retention`)
   - Operating system
   - Full error traceback
   - Minimal code to reproduce the issue

**Bug Report Template:**

```markdown
## Description
[Clear description of the bug]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [See error]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Package version: [e.g., 1.0.0]

## Additional Context
[Any other relevant information]
```

### Suggesting Features

We welcome feature suggestions! Please:

1. Check if the feature has already been requested
2. Explain the use case and business value
3. Describe the desired behavior
4. Consider implementation complexity

### Contributing Code

1. **Find an issue** to work on, or create one for discussion
2. **Comment on the issue** to let others know you're working on it
3. **Fork the repository** and create a feature branch
4. **Write code** following our standards (see below)
5. **Write tests** for your changes
6. **Submit a pull request**

---

## Code Standards

> **These standards are MANDATORY and enforced by pre-commit hooks and CI.**
> Pull requests that do not meet these standards will be automatically rejected.

### Coding Practices (REQUIRED)

The following practices from `docs/Coding_Practices.md` are **mandatory** for all contributions:

#### 1. Avoid Duplication
Before introducing any changes, look in the existing code for the best place to introduce changes/enhancements to avoid duplication and overlapping scope. **Ask questions instead of silently making architectural decisions.**

#### 2. No Comments - Use Descriptive Names
Do not use comments. Instead, use descriptive class, method, and field names that make the code self-documenting.

```python
# BAD - relies on comment
def calc(x, y):  # Calculate customer lifetime value
    return x * y * 12

# GOOD - self-documenting
def calculate_annual_customer_lifetime_value(monthly_revenue, retention_rate):
    return monthly_revenue * retention_rate * 12
```

#### 3. Single Responsibility Functions
Favor short, single-responsibility functions with no side effects that operate on the same abstraction level.

```python
# BAD - multiple responsibilities
def process_customer(customer):
    # validate
    if not customer.email:
        raise ValueError("Missing email")
    # transform
    customer.email = customer.email.lower()
    # save
    db.save(customer)
    # notify
    send_email(customer.email)

# GOOD - single responsibility each
def validate_customer(customer):
    if not customer.email:
        raise ValueError("Missing email")

def normalize_email(email):
    return email.lower()

def process_customer(customer):
    validate_customer(customer)
    customer.email = normalize_email(customer.email)
    save_customer(customer)
    notify_customer(customer)
```

#### 4. Public Methods First
Start from public methods/high-level functions first, then functions that are implementation details.

#### 5. Favor Inheritance Over Repetition
Design the structure of the code to avoid code repetition and allow inheritance to take care of variations.

#### 6. Compact Code
- No ceremonial assignments for variables used just once
- Do not spread method arguments on many lines unless they are more than 5
- Prefer compact code that does not spread across too many lines

```python
# BAD - ceremonial assignment
temp_value = calculate_score(data)
result = process(temp_value)

# GOOD - inline when used once
result = process(calculate_score(data))

# BAD - unnecessary line spreading
def calculate(
    x,
    y,
):
    return x + y

# GOOD - compact for few arguments
def calculate(x, y):
    return x + y
```

### Style Guide

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (enforced by pre-commit):

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `DataExplorer`, `RiskProfiler`)
- Functions/methods: `snake_case` (e.g., `calculate_roi`, `detect_drift`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- Private methods: `_leading_underscore`

### Type Hints (Required)

```python
def calculate_churn_probability(
    features: pd.DataFrame, model: BaseEstimator, threshold: float = 0.5
) -> np.ndarray:
    ...
```

### Project Structure

```
src/customer_retention/
├── module_name/
│   ├── __init__.py      # Public API exports
│   ├── core.py          # Main implementation
│   └── types.py         # Type definitions (if needed)
```

---

## Testing Requirements

> **Test-Driven Development (TDD) is REQUIRED.**
> Create tests first using the specification, then do the implementation.
> After completing all tasks, check against the specification again.

### Pre-commit Hooks (MANDATORY)

You **must** install pre-commit hooks before contributing:

```bash
# Install pre-commit hooks (required - do this once)
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
```

| Hook | Stage | What it does |
|------|-------|--------------|
| `ruff` | pre-commit | Linting and auto-fix |
| `ruff-format` | pre-commit | Code formatting |
| `python-check` | pre-commit | Syntax validation |
| `pytest-coverage` | **pre-push** | **Enforces 75% minimum coverage** |

### Coverage Requirements (ENFORCED)

| Requirement | Value | Enforcement |
|-------------|-------|-------------|
| **Minimum coverage** | **75%** | Pre-push hook + CI |
| Target coverage | 85%+ | Best practice |
| Current coverage | 84% | Maintained |

**This cannot be bypassed:**
- Pre-push hook blocks pushes below 75% coverage
- CI pipeline rejects PRs below 75% coverage
- Using `--no-verify` will NOT help - CI will still reject the PR

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (must be >= 75%)
pytest tests/ --cov=src/customer_retention --cov-report=term-missing

# Run with coverage enforcement (same as pre-push hook)
pytest tests/ --cov=src/customer_retention --cov-fail-under=75

# Run specific module tests
pytest tests/modeling/ -v

# Run single test file
pytest tests/modeling/test_baseline_trainer.py -v
```

### Writing Tests (TDD Required)

**Test-Driven Development Process:**
1. Write tests first based on the specification
2. Run tests (they should fail)
3. Implement the code
4. Run tests (they should pass)
5. Check against specification again

**Required Test Coverage:**
- **Edge cases** - empty inputs, boundary values, single elements
- **Antipatterns** - invalid inputs, type errors, missing data
- **Happy path** - normal expected behavior

**Test file structure:**
```python
"""Tests for customer_retention.modeling.baseline_trainer."""

import pytest
import numpy as np
import pandas as pd
from customer_retention.modeling import BaselineTrainer


class TestBaselineTrainerInit:
    """Tests for BaselineTrainer initialization."""

    def test_default_init(self):
        trainer = BaselineTrainer()
        assert trainer.random_state == 42

    def test_custom_random_state(self):
        trainer = BaselineTrainer(random_state=123)
        assert trainer.random_state == 123


class TestBaselineTrainerTrain:
    """Tests for BaselineTrainer.train() method."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.randn(100), 'f2': np.random.randn(100)})
        y = np.random.randint(0, 2, 100)
        return X, y

    # Happy path
    def test_train_returns_fitted_model(self, sample_data):
        X, y = sample_data
        model = BaselineTrainer().train(X, y, model_type='logistic')
        assert hasattr(model, 'predict')

    # Edge case - minimal data
    def test_train_with_minimal_data(self):
        X = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        y = [0, 1]
        model = BaselineTrainer().train(X, y, model_type='logistic')
        assert model is not None

    # Antipattern - invalid input
    def test_train_invalid_model_type_raises(self, sample_data):
        X, y = sample_data
        with pytest.raises(ValueError, match="Unknown model type"):
            BaselineTrainer().train(X, y, model_type='invalid')

    # Antipattern - empty input
    def test_train_empty_dataframe_raises(self):
        X = pd.DataFrame()
        y = []
        with pytest.raises(ValueError):
            BaselineTrainer().train(X, y, model_type='logistic')
```

### Test Categories

Use pytest markers for different test types:

```python
@pytest.mark.unit
def test_calculation():
    """Fast unit test."""
    ...

@pytest.mark.integration
def test_pipeline_end_to_end():
    """Integration test requiring multiple components."""
    ...

@pytest.mark.slow
def test_full_model_training():
    """Slow test (>10 seconds)."""
    ...
```

Run specific categories:
```bash
pytest -m "unit"           # Only unit tests
pytest -m "not slow"       # Skip slow tests
```

---

## Pull Request Process

### Mandatory Requirements (CANNOT BE BYPASSED)

> **All PRs must pass these automated checks. There are no exceptions.**

| Check | Requirement | Bypass Possible? |
|-------|-------------|------------------|
| Test Coverage | >= 75% | **NO** |
| All Tests Pass | 100% | **NO** |
| Ruff Lint | No errors | **NO** |
| Ruff Format | Formatted | **NO** |

Using `git commit --no-verify` or `git push --no-verify` will **NOT** help you bypass these requirements. The CI pipeline enforces all checks and will reject non-compliant PRs.

### Before Submitting

1. **Ensure pre-commit hooks are installed**:
   ```bash
   pre-commit install --install-hooks
   pre-commit install --hook-type pre-push
   ```

2. **Update your branch** with the latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

3. **Run the full test suite with coverage**:
   ```bash
   pytest tests/ --cov=src/customer_retention --cov-fail-under=75 -v
   ```

4. **Check code style**:
   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   ```

5. **Update documentation** if needed

### PR Template

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] Test coverage maintained or improved

## Checklist
- [ ] My code follows the project style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented hard-to-understand areas
- [ ] I have updated documentation as needed
- [ ] My changes generate no new warnings

## Related Issues
Closes #[issue number]
```

### Review Process

1. **Automated checks MUST pass** (non-negotiable):
   - All tests pass
   - Coverage >= 75%
   - Ruff lint passes
   - Ruff format passes
2. **Code review** by at least one maintainer
3. **Address feedback** with additional commits
4. **Squash and merge** when approved

> **Note:** PRs with failing automated checks cannot be merged, even with maintainer approval.
> Branch protection rules enforce this at the repository level.

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(modeling): add CatBoost support to BaselineTrainer

fix(monitoring): correct PSI calculation for categorical features

docs(readme): add case study section with ROI metrics

test(business): add fairness analyzer edge case tests
```

---

## Release Process

Releases are managed by maintainers following [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking API changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backward compatible

### Version Locations

Update version in:
- `pyproject.toml` (`version = "X.Y.Z"`)

---

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/yourusername/customer-retention/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/yourusername/customer-retention/issues)
- **Security issues**: Email maintainers directly (do not create public issues)

---

Thank you for contributing to Customer Retention Framework!
