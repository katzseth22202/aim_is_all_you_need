# AIM is All You Need

A basic Python project demonstrating proper project structure with src and tests directories, type checking with mypy, and comprehensive testing with pytest.

## Project Structure

```
aim_is_all_you_need/
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py            # Main application entry point
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_main.py       # Tests for main module
├── pyproject.toml         # Project configuration and dependencies
├── requirements.txt       # Development dependencies
├── Makefile              # Convenient commands for development

└── README.md            # This file
```

## Features

- **Python 3.10+**: Requires Python 3.10 or higher
- **Type Safety**: Full mypy type checking with strict settings
- **Testing**: Comprehensive pytest test suite with coverage reporting
- **Code Quality**: Black formatting, isort import sorting, and flake8 linting
- **Development Tools**: Makefile with convenient commands
- **Modern Python**: Uses pyproject.toml for modern Python packaging

## Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate balloon_math_env

# Install the package in development mode
make install-dev
```

### Option 2: Using pip

```bash
# Install the package with development dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
```

### 2. Run the Complete Workflow

```bash
# Run mypy, tests, and then the main script
make all
```

### 3. Individual Commands

```bash
# Type checking
make mypy

# Run tests
make test

# Run tests with coverage
make test-cov

# Run the main script
make run

# Format code
make format

# Check code formatting
make check-format

# Clean build artifacts
make clean
```

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

## Development

### Adding New Code

1. Add your code to the `src/` directory
2. Add corresponding tests to the `tests/` directory
3. Ensure type hints are properly added
4. Run `make all` to verify everything works

### Testing

The project includes:
- Unit tests for all functions
- Integration tests (marked with `@pytest.mark.slow`)
- Coverage reporting
- Type checking with mypy

### Code Quality

The project enforces:
- Black code formatting
- isort import sorting
- mypy type checking with strict settings

## Example Usage

```python
from src.main import greet, calculate_sum

# Greet someone
message = greet("Alice")
print(message)  # "Hello, Alice! Welcome to AIM is all you need!"

# Calculate sum of numbers
total = calculate_sum([1, 2, 3, 4, 5])
print(total)  # 15
```

## Running the Main Script

```bash
# Using make
make run

# Using python directly
python -m src.main

# Using the installed script
main
```

## Environment Management

### Conda Environment

The project includes an `environment.yml` file for reproducible conda environments:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate balloon_math_env

# Update the environment (if environment.yml changes)
conda env update -f environment.yml

# Export current environment (if you make changes)
conda env export > environment.yml
```

### Virtual Environment (Alternative)

If you prefer using venv instead of conda:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install pytest pytest-cov coverage
pip install -r requirements.txt
pip install -e ".[dev]"
```

**Note**: When using pip instead of conda, you'll need to manually install `pytest`, `pytest-cov`, and `coverage` as they are conda packages in the environment.yml.

## Configuration

### Mypy Configuration

The project uses strict mypy settings defined in `pyproject.toml`:
- Disallows untyped definitions
- Warns about return Any
- Enforces strict equality checks
- Tests are exempted from some strict rules

### Pytest Configuration

Configured in `pyproject.toml`:
- Test discovery in `tests/` directory
- Coverage reporting
- Custom markers for slow and integration tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `make all`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
