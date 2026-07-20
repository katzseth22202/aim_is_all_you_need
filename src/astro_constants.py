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

# --- "Sorry, I Don't Need ISRU" solar-dive re-intercept (paper Appendix
# sec:earth_reintercept) ---
# The deep dive targets the 4 solar-radii periapsis of the 2005 Solar Probe
# design (mccomas2005solar), a count of Sun radii so astro_constants stays free
# of the boinor Sun body; callers multiply by Sun.R.
SOLAR_DIVE_PERIAPSIS_SOLAR_RADII = 4.0
# Shallow-dip periapsis of the two-impulse phasing loop: the projectile dips to
# ~0.50 AU, returns to 1 AU tangentially after one dip period (~0.65 yr), then a
# second colinear retrograde PuffSat boost drops it into the deep dive.
TWO_IMPULSE_DIP_PERIAPSIS = 0.50 * u.AU
# PuffSat boost applied at the 4 solar-radii dive periapsis. Added to the actual
# incoming ellipse speed, it leaves ~143 km/s of excess for the minimum-energy
# 1 AU dive and ~147 km/s for the resonant dive -- close to the main text's
# ~150 km/s Earth-crossing scale. This tones down an earlier formulation that
# scaled the Parker boost fraction (x1.25) up to the deeper dive, which left an
# aggressive ~233 km/s of excess.
SOLAR_DIVE_PERIAPSIS_BURN = 34.5 * u.km / u.s

# --- Powered Jovian flyby retrograde return (planned subsection under
#     sec:jupiter_only_growth; ADR 0002; CONTEXT.md "Jupiter powered-flyby
#     retrograde return") ---
# Floor on the Jovian flyby periapsis altitude (above the 1-bar level). Juno-class
# perijove: conservative against radiation/atmosphere, and nearly free -- the
# Oberth penalty versus a 200 km graze is only ~40 m/s of delta-v.
LOW_JUPITER_ALTITUDE = 4000 * u.km
# Hard cap on outbound (Earth->Jupiter) plus return (Jupiter->1 AU) heliocentric
# time of flight. Excludes extreme apoapsis-raise trajectories; the
# Hohmann-out/Hohmann-back baseline is ~5.5 yr, leaving ~1.5 yr of slack.
JUPITER_FLYBY_MAX_TOF = 7 * u.year
# Earth-closing-speed targets (km/s) for the v_b trade curve printed alongside
# the optimum: from plunge-like returns (~45) up past the retrograde-Hohmann
# closing speed (~69.3).
JUPITER_FLYBY_VB_TRADE_TARGETS = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0)

# --- Unpowered V/E/M assist chain to the retrograde return (companion to the
#     powered Jovian flyby above; ADR 0003-assist-chain-search; CONTEXT.md
#     "Unpowered assist chain") ---
# Minimum flyby altitude above the surface at the chain bodies (Venus, Earth,
# Mars). 300 km clears the Venusian and Martian atmospheres and matches the
# repo's LEO reference altitude scale.
LOW_ASSIST_FLYBY_ALTITUDE = 300 * u.km
# Hard cap on the whole trip: departure -> inner-planet flyby chain -> Jupiter
# -> first retrograde 1 AU crossing. Looser than the powered flyby's 7 yr
# because the chain spends time pumping v-infinity instead of propellant.
ASSIST_CHAIN_MAX_TRIP_TIME = 10 * u.year
# Cap on the number of inner-planet flybys before the Jovian leg.
ASSIST_CHAIN_MAX_FLYBYS = 5
# Deep-space maneuver budget reserved for planetary phasing. The chain model is
# phasing-free (each planet is wherever the trajectory needs it), so this
# reserve is charged as spent methalox in the delivered-fraction accounting --
# the headline mass numbers already carry the cost of making real ephemerides
# line up.
ASSIST_CHAIN_PHASING_BUDGET = 0.300 * u.km / u.s
# Departure-burn probes (km/s) for the minimum-burn scan, bracketing the
# analytic Venus-reach floor (~0.2794 km/s, venus_reach_departure_floor()) and
# the beam-search feasibility edge (~0.29 km/s at the production beam
# settings).
ASSIST_CHAIN_BURN_CANDIDATES = (0.250, 0.280, 0.290, 0.300)

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

# --- Apoapsis-raise Earth re-intercept (candidate sec:earth_reintercept option;
#     design doc apoapsis_raise_reintercept_design.md in the paper source repo) ---
# A projectile leaves Earth at 200 km (C3=0), an Oberth methalox burn raises its
# heliocentric aphelion to Q, a retrograde argon-SEP burn at apoapsis lowers
# perihelion, and it falls back to intercept Earth at 1 AU. The lowest-closing-
# speed member of the sec:earth_reintercept family: no solar dive, no gravity
# assist, no off-Earth boost node, only onboard propellant.
#
# Methalox vacuum Isp for the Oberth departure burn at 200 km (Leg 1). Higher than
# the 310 s sea-level value above because the departure burn fires in vacuum.
METHALOX_VACUUM_ISP = 380 * u.s
# Argon solar-electric (SEP) Isp for the retrograde apoapsis burn (Leg 2).
ARGON_SEP_ISP = 2000 * u.s
# Period of the bound Earth orbit the returning PuffSat's collision leaves the
# mass in, closing the growth loop: the collision pushes the mass to just under
# escape, it coasts to apoapsis and falls back to the 200 km periapsis, and the
# next cycle's Oberth burn fires there. Bound rather than escaping, so the mass
# is not lost; near-escape, so the collision extracts nearly all it can.
#
# The value is close to arbitrary. Periapsis speed runs 10.8610 km/s at 5 days to
# 10.9806 at 60 -- a 120 m/s spread on a v_rf of ~10.95 -- because the mass ratio
# 2f/ln((v_b - v_ri)/(v_b - v_rf)) is nearly blind to v_rf while v_b >> v_rf. It
# sets the coast that pads the cycle, not the growth. 20 days puts apoapsis at
# ~615,900 km, about 1.6 lunar distances.
PUFFSAT_CYCLE_ORBIT_PERIOD = 20 * u.day
# Retrograde SEP delta-v spent at apoapsis (Leg 2). Capped at the design's full
# 4 km/s SEP budget; the phasing-exact optimum spends it in full (raising Delta-v2
# does not shrink the phasing-locked orbit, only deepens the closing speed).
APOAPSIS_RAISE_SEP_DV = 4.0 * u.km / u.s
# Aphelion search bracket (AU) for the phasing-exact apoapsis root solve. The lone
# phasing-exact Earth intercept sits at Q ~ 2.26 AU; this brackets it while
# excluding the anti-phased (~180 deg miss) feature near Q ~ 1.65 AU. The design
# searched Q in [2.0, 3.0]; narrowed here to isolate the single root.
APOAPSIS_RAISE_APHELION_BRACKET = (2.1, 2.45)
# Nominal SEP burn duration for the finite-thrust check. A 60-90 day burn centered
# on apoapsis reproduces the impulsive kick to within ~1% in closing speed; apoapsis
# is the slowest part of the orbit, so the thrust stays near-tangential throughout.
APOAPSIS_RAISE_SEP_BURN_DURATION = 90 * u.day

# --- Near-escape dry-mass disposal: Earth reentry vs. lunar impact (paper
#     sec:coordinator_node_dry_mass_disposal) ---
# A spent package pushed just past Earth's escape velocity coasts out to the edge
# of Earth's gravitational reach, nearly stops (a near-parabolic orbit's speed
# there is essentially the local escape speed), and a small retrograde burn at
# that turnaround lowers its perigee for disposal. Two targets: graze the
# atmosphere and reenter, or fall only to lunar distance and strike the Moon. The
# Moon route lowers perigee less, so it costs less delta-v.
#
# Turnaround radius: Earth's sphere of influence is a_earth*(m_earth/m_sun)^(2/5)
# ~= 924,000 km; the paper rounds this to "roughly 900,000 km" for the near-zero
# turnaround speed (~0.94 km/s) it quotes, so we pin the same round value here.
EARTH_GRAVITATIONAL_REACH = 900000 * u.km
# Perigee radius that grazes the atmosphere for reentry. ~6500 km is Earth's
# ~6371 km radius plus a ~130 km reentry-interface altitude; the paper quotes
# "the atmosphere near 6500 km" as this target.
REENTRY_PERIGEE_RADIUS = 6500 * u.km
