# Balloon Pulse Propulsion Calculations

This repository contains Python implementations of the orbital mechanics calculations described in the research paper "Aim Is All You Need" (located in the `paper/` subdirectory). The code provides computational tools for analyzing balloon-based propulsion systems for various space missions.

## Overview

The calculations in this repository support the theoretical framework presented in the "Aim Is All You Need" paper, which explores the use of balloon-based propulsion systems for:

- **Lunar Transfer Missions**: Optimizing trajectories from Earth to the Moon using balloon propulsion
- **Interplanetary Travel**: Analyzing balloon propulsion for missions to Jupiter, Saturn, and other bodies
- **Escape Velocity Calculations**: Computing optimal burn strategies for achieving escape velocities
- **Mass Ratio Optimization**: Finding the best payload-to-propulsion mass ratios for various scenarios

## Key Features

- **Orbital Mechanics**: Comprehensive calculations for Hohmann transfers, escape velocities, and orbital maneuvers
- **Balloon Propulsion Analysis**: Tools for analyzing balloon-based propulsion scenarios
- **Mass Ratio Optimization**: Functions to find optimal payload-to-propulsion mass ratios
- **Multi-body Systems**: Support for calculations involving Earth, Moon, Jupiter, Saturn, and other celestial bodies
- **Type Safety**: Full mypy type checking with strict settings and comprehensive type annotations

## Project Structure

```
aim_is_all_you_need/
├── src/                    # Source code
│   ├── __init__.py
│   ├── astro_constants.py  # Astronomical constants and parameters
│   ├── compute_utils.py    # Core orbital mechanics calculations
│   └── main.py            # Main application entry point
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_compute_utils.py # Tests for orbital mechanics functions
├── pyproject.toml         # Project configuration and dependencies
├── requirements.txt       # Development dependencies
├── Makefile              # Convenient commands for development
├── environment.yml       # Conda environment (auto-updated)
└── README.md            # This file
```

## Core Calculations

### Orbital Mechanics Functions

- `hohmann_transfer()`: Compute Hohmann transfer maneuvers between circular orbits
- `escape_velocity()`: Calculate escape velocities from celestial bodies
- `payload_mass_ratio()`: Determine optimal payload-to-balloon mass ratios
- `find_best_lunar_return()`: Optimize lunar return trajectories with maximum mass ratios

### Balloon Propulsion Scenarios

The `BalloonScenario` class provides predefined scenarios from the paper:
- Eccentric balloons for lunar transfer missions
- Retrograde balloon deceleration for Earth reentry
- Jupiter-to-Earth retrograde Hohmann transfers
- Saturn and Phoebe orbital maneuvers

### Advanced Features

- **Retrograde Calculations**: Support for retrograde orbital maneuvers
- **Multi-body Optimization**: Calculations involving multiple celestial bodies
- **Launch Capacity Analysis**: Time calculations for achieving target launch capacities

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

### Run Calculations

```bash
# Run the complete workflow including tests
make all

# Run specific calculations and generate output
make run
```

### Generate Calculation Output

To see the results of the orbital mechanics calculations, run:

```bash
make run
```

This will execute the main calculations and output the results to standard output, including:
- Balloon propulsion scenario analysis
- Optimal lunar return trajectories
- Mass ratio calculations for various missions
- Orbital mechanics computations

## Dependencies

The project uses several key libraries for orbital mechanics calculations:

- **poliastro**: For orbital mechanics and celestial body definitions
- **astropy**: For astronomical units and constants
- **numpy**: For numerical computations
- **pandas**: For data analysis and scenario tables

## Testing

The repository includes comprehensive tests for all orbital mechanics functions:

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `make all`
6. Submit a pull request

## Related Work

This code implements calculations from the research paper "Aim Is All You Need" (located in the `paper/` subdirectory), which presents a novel approach to space propulsion using balloon-based systems. The paper provides the theoretical framework, while this repository provides the computational tools to analyze and optimize these propulsion systems.

## License

This repository uses a dual licensing structure:

- **Code**: Licensed under Apache License 2.0 - see `LICENSE_OF_CODE_ONLY` for details
- **Paper**: All rights reserved - see `paper/LICENSE` for details

The source code and computational tools are open source and freely available for use, modification, and distribution. The research paper content is protected by copyright and requires explicit permission for reproduction or distribution.
