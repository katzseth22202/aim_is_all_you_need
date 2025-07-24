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
