from astropy import units as u

# Semi-major axes of the planets (IAU 2015 values, in astronomical units)
MERCURY_A = 0.3871 * u.AU  # Mercury
VENUS_A = 0.7233 * u.AU  # Venus
EARTH_A = 1.0000 * u.AU  # Earth
MARS_A = 1.5237 * u.AU  # Mars
JUPITER_A = 5.2028 * u.AU  # Jupiter
SATURN_A = 9.5388 * u.AU  # Saturn
URANUS_A = 19.1914 * u.AU  # Uranus
NEPTUNE_A = 30.0611 * u.AU  # Neptune
PLUTO_A = 39.482 * u.AU  # Pluto (dwarf planet)

MOON_A = 384400 * u.km  # Moon (semi-major axis of orbit around Earth, in km)
LEO_ALTITUDE = 200 * u.km  # Low Earth Orbit altitude, in km

PARKER_PERIAPSIS = 6.9e6 * u.km  # lowest Parker periapsis

PHOEBE_A = 12952000 * u.km  # Phoebe (semi-major axis of orbit around Saturn, in km)
LOW_SATURN_ALTITUDE = 500 * u.km  # Low Saturn orbit altitude, in km
EFFECTIVE_DV_LUNAR = 3 * u.km / u.s  # Effective exhaust for lunar proposed in paper
REQUIRED_DV_LUNAR_TRANSFER_PROGRADE = (
    3 * u.km / u.s
)  # How much to get off the moon and into a prograde LEO transfer (varies depending on moon location)
REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE = (
    3.7 * u.km / u.s
)  # How much to get off the moon and into a retrograde LEO transfer (varies depending on moon location)
RETROGRADE_FRACTION = (
    1 / 4
)  # Mass fraction to send into retrograde orbit for collision in propulsion pulse chambers

TARGET_LAUNCH_CAPACITY_MULTIPLE = (
    1e6  # default of how much we want to multiply initial launch capacity by
)
LUNAR_MONTH = 27.3 * u.day  # average time of lunar orbit
PERIAPSIS_SOLAR_V = (
    200 * u.km / u.s
)  # we propose this velocity in the paper's section on using the Sun's Oberth  (a bit faster/closer than Parker Space Probe but similar)
PERIAPSIS_SOLAR_BURN = (
    50 * u.km / u.s
)  # we propose this speed increase in the paper's section on using the Sun's Oberth

STD_FUDGE_FACTOR: float = (
    0.8  # the fudge factor described in the paper for how elastic PuffSat collisions are
)
