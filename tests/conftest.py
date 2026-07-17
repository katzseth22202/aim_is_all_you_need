"""Shared fixtures for the Jovian-flyby / assist-chain test modules.

These three fixtures are used across test_retrograde_return_legs.py,
test_jovian_flyby.py, and test_assist_chain.py -- centralized here rather than
duplicated in each file. Module-scoped fixtures defined in conftest.py still
get one cached instance per requesting test module, so this changes nothing
about how often the (slow) searches run.
"""

import pytest
from astropy import units as u

from src.assist_chain import assist_chain_return
from src.jovian_flyby import powered_jovian_flyby_return
from src.retrograde_return_legs import _powered_flyby_params


@pytest.fixture(scope="module")
def flyby_params():
    """Float parameter block for the powered-flyby helpers."""
    return _powered_flyby_params()


@pytest.fixture(scope="module")
def flyby_optimum():
    """The end-to-end optimum, computed once for the module (seeded search)."""
    return powered_jovian_flyby_return()


@pytest.fixture(scope="module")
def chain_at_300(flyby_optimum):
    """The 300 m/s assist chain, computed once (the beam search is slow)."""
    chain = assist_chain_return(
        departure_burn=0.300 * u.km / u.s,
        target_collision_speed=flyby_optimum.collision_speed,
    )
    assert chain is not None
    return chain
