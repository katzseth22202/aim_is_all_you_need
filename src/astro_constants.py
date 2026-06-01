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

# Ceres (dwarf planet semi-major axis) {source: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=ceres}
CERES_A = 2.7656 * u.AU
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
# Retrograde collision mass fraction, from the paper's Appendix D (Eq 12): the
# value that maximizes pulse-chamber momentum transfer 2*v_p*(sqrt(m) - m) is
# m = 1/4. The prograde fraction is the complementary 3/4 (m_rp + m_pp = 1).
RETROGRADE_FRACTION = 1 / 4

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

# --- Suborbital "200 km" rocket delta-v budget (paper Section 2.1) ---
# The paper claims a reusable suborbital rocket that merely reaches ~200 km
# altitude (it does NOT reach orbit on its own; PuffSat pulses do the rest) can
# have a propellant mass fraction under 60 percent. We back that with a ~2.5 km/s
# delta-v budget split into an ideal coast term plus gravity-drag losses:
#   - ideal impulsive speed to ballistically coast to 200 km is
#     sqrt(2 * g0 * h) = sqrt(2 * 9.80665 m/s^2 * 200 km) = 1981 m/s ~= 2.0 km/s
#   - gravity-drag losses during a finite high-thrust ascent ~= 0.5 km/s
# giving a total of 2.5 km/s (= 2500 m/s) to reach 200 km altitude.
SUBORBITAL_IDEAL_DV_TO_200KM = 2.0 * u.km / u.s  # ideal sqrt(2*g0*h) coast to 200 km
SUBORBITAL_GRAVITY_DRAG = 0.5 * u.km / u.s  # gravity-drag losses during ascent
SUBORBITAL_DV_TO_200KM = (
    SUBORBITAL_IDEAL_DV_TO_200KM + SUBORBITAL_GRAVITY_DRAG
)  # total delta-v to reach 200 km altitude (2.5 km/s)

# Methalox (LOX/CH4) engine sea-level specific impulse. Conservative versus real
# engines (e.g. Raptor ~327 s sea level / ~350 s vacuum); using the lower
# sea-level value makes the resulting propellant-fraction estimate an upper bound,
# which strengthens the paper's "under 60 percent" claim.
METHALOX_SEA_LEVEL_ISP = 310 * u.s
