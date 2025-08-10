# PuffSat Pulse Propulsion Calculations

This repository contains Python implementations of the orbital mechanics calculations described in the research paper "Aim Is All You Need" ([DOI: 10.5281/zenodo.16741183](https://doi.org/10.5281/zenodo.16741183)) (also available in the `paper/` subdirectory). The code provides computational tools for analyzing PuffSat-based propulsion systems for various space missions.

## What are PuffSats?

**PuffSats**  PuffSats are compact satellite platforms that generate gas explosively or via rapid satomizations just prior to impact with a momentum pusher plate to push a second rockdet.

## Overview

The calculations in this repository support the theoretical framework presented in the "Aim Is All You Need" paper ([DOI: 10.5281/zenodo.16741183](https://doi.org/10.5281/zenodo.16741183)), which explores the use of PuffSat-based propulsion systems for:

- **Lunar Transfer Missions**: Optimizing trajectories from Earth to the Moon using PuffSat propulsion
- **Interplanetary Travel**: Analyzing PuffSat propulsion for missions to Jupiter, Saturn, and other bodies
- **Escape Velocity Calculations**: Computing optimal burn strategies for achieving escape velocities
- **Mass Ratio Optimization**: Finding the best payload-to-propulsion mass ratios for various scenarios

## Key Features

- **Orbital Mechanics**: Comprehensive calculations for Hohmann transfers, escape velocities, and orbital maneuvers
- **PuffSat Propulsion Analysis**: Tools for analyzing PuffSat-based propulsion scenarios
- **Mass Ratio Optimization**: Functions to find optimal payload-to-propulsion mass ratios
- **Multi-body Systems**: Support for calculations involving Earth, Moon, Jupiter, Saturn, and other celestial bodies
- **Type Safety**: Full mypy type checking with strict settings and comprehensive type annotations

## Project Structure

```
aim_is_all_you_need/
├── src/                    # Source code
│   ├── __init__.py
│   ├── astro_constants.py  # Astronomical constants and parameters
│   ├── orbit_utils.py      # Core orbital mechanics calculations
│   ├── propulsion.py       # Propulsion analysis functions
│   ├── scenario.py         # PuffSat scenario analysis
│   └── main.py            # Main application entry point
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_helpers.py     # Shared test utilities
│   ├── test_orbit_utils.py # Tests for orbital mechanics functions
│   ├── test_propulsion.py  # Tests for propulsion functions
│   └── test_scenario.py    # Tests for scenario analysis
├── paper/                  # Research paper
│   ├── Aim_Is_All_You_Need.pdf
│   ├── README.md
│   └── LICENSE
├── pyproject.toml         # Project configuration and dependencies
├── requirements.txt       # Development dependencies
├── Makefile              # Convenient commands for development
├── environment.yml       # Conda environment (auto-updated)
└── README.md            # This file
```

## Core Calculations

### Orbital Mechanics Functions (`orbit_utils.py`)

- `body_speed()`: Calculate orbital speed at given altitude
- `escape_velocity()`: Calculate escape velocities from celestial bodies
- `orbit_from_rp_ra()`: Create orbits from periapsis/apoapsis radii
- `velocity_at_distance()`: Calculate velocity at any point in orbit
- `get_orbital_velocity_at_radius()`: General velocity calculation for any radius

### Propulsion Analysis (`propulsion.py`)

- `hohmann_transfer()`: Compute Hohmann transfer maneuvers between circular orbits
- `payload_mass_ratio()`: Determine optimal payload-to-PuffSat mass ratios
- `burn_for_v_infinity()`: Calculate burns for hyperbolic trajectories
- `retrograde_jovian_hohmann_transfer()`: Specialized Jupiter-Earth transfer

### PuffSat Propulsion Scenarios (`scenario.py`)

The `PuffSatScenario` class provides predefined scenarios from the paper:
- Eccentric PuffSats for lunar transfer missions
- Retrograde PuffSat deceleration for Earth reentry
- Jupiter-to-Earth retrograde Hohmann transfers
- Saturn and Phoebe orbital maneuvers
- Solar periapsis maneuvers for high-velocity missions

### Advanced Features

- **Retrograde Calculations**: Support for retrograde orbital maneuvers using PuffSat propulsion
- **Multi-body Optimization**: Calculations involving multiple celestial bodies (Earth, Moon, Jupiter, Saturn, Sun)
- **Launch Capacity Analysis**: Time calculations for achieving target launch capacities through PuffSat systems
- **Nuclear Fusion Scenarios**: Analysis of high-velocity impact scenarios for fusion research
- **Parker Space Probe Trajectories**: Calculations for solar periapsis missions

## Quick Start

### Using Conda

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate puffsat_math_env
```

The conda environment includes all necessary dependencies including development tools.

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
- PuffSat propulsion scenario analysis
- Optimal lunar return trajectories
- Mass ratio calculations for various missions
- Orbital mechanics computations

## Dependencies

The project uses several key libraries for orbital mechanics and scientific computing:

- **poliastro**: For orbital mechanics and celestial body definitions
- **astropy**: For astronomical units and constants
- **numpy**: For numerical computations and array operations
- **pandas**: For data analysis and scenario tables
- **tabulate**: For formatted output display
- **mypy**: For static type checking
- **pytest**: For testing framework

## Testing

The repository includes comprehensive tests for all functions:

```bash
# Run all tests
make test

# Run type checking
make mypy

# Run code formatting checks
make check-format

# Run complete quality check (format, type check, tests)
make all
```

The test suite includes:
- **Unit tests** for all orbital mechanics functions
- **Integration tests** for propulsion calculations
- **Scenario tests** for PuffSat mission analysis
- **Helper utilities** for floating-point comparisons

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `make all`
6. Submit a pull request

### Raw LaTeX Assets

The raw LaTeX source files for the included research paper can be found in the separate repository:

**Repository:** [Balloon-Pulse-Propulsion](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)

**Citation:**
```bibtex
@misc{Katz_Balloon-Pulse-Propulsion_2025,
author = {Katz, Seth},
doi = {https://doi.org/10.5281/zenodo.16740748},
month = aug,
title = {{Balloon-Pulse-Propulsion}},
url = {https://github.com/katzseth22202/Balloon-Pulse-Propulsion},
year = {2025}
}
```

## Citation

If you use this software or the included research paper in your work, please cite it as follows:

**Software Citation:**
```bibtex
@software{Katz_aim_is_all_you_need_2025,
author = {Katz, Seth},
doi = {10.5281/zenodo.16718455},
month = aug,
title = {{aim\_is\_all\_you\_need}},
url = {https://github.com/katzseth22202/aim_is_all_you_need},
year = {2025}
}
```

**Paper Citation:**
```bibtex
@misc{katz_2025_16741183,
  author       = {Katz, Seth},
  title        = {Aim Is All You Need - A Speculative White Paper on
                   Externally Pulsed Propulsion
                  },
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16741183},
  url          = {https://doi.org/10.5281/zenodo.16741183},
}
```

## License

This repository uses a dual licensing structure:

- **Code**: Licensed under Apache License 2.0 - see `LICENSE_OF_CODE_ONLY` for details
- **Paper**: All rights reserved - see `paper/LICENSE` for details

The source code and computational tools are open source and freely available for use, modification, and distribution. The research paper content is protected by copyright and requires explicit permission for reproduction or distribution.
