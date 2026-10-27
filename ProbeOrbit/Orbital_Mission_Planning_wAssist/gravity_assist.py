import numpy as np


# =========================================================
# Gravity assist (patched conic) helpers
#
# All inputs and outputs are in km, km/s and km^3/s^2, the
# same units as lambert.py.
#
# The functions are written for NumPy arrays so the sweep can
# test thousands of (departure, flyby, arrival) combinations
# at once. Scalars work too.
# =========================================================

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km

# J2000 obliquity of the ecliptic
OBLIQUITY = np.radians(23.43927944)


# =========================================================
# Flyby turn angle
# =========================================================

def turn_angle(v_inf_in, v_inf_out):
    """
    Angle between the incoming and outgoing v-infinity vectors (rad).

    v_inf_in, v_inf_out: (..., 3) arrays
    """

    v_in = np.linalg.norm(v_inf_in, axis=-1)
    v_out = np.linalg.norm(v_inf_out, axis=-1)

    cos_delta = np.sum(v_inf_in * v_inf_out, axis=-1) / (v_in * v_out)

    return np.arccos(np.clip(cos_delta, -1.0, 1.0))


def max_turn_angle(v_in, v_out, rp, mu):
    """
    Turn angle produced by a flyby with periapsis radius rp when the
    hyperbolic excess speed changes from v_in to v_out at periapsis.

    Each half of the flyby bends the path by asin(1/e):
        e_in  = 1 + rp v_in^2  / mu
        e_out = 1 + rp v_out^2 / mu

    For an unpowered flyby (v_in = v_out) this reduces to the usual
        delta = 2 asin(1 / (1 + rp v^2 / mu))
    """

    e_in = 1.0 + rp * v_in**2 / mu
    e_out = 1.0 + rp * v_out**2 / mu

    return np.arcsin(1.0 / e_in) + np.arcsin(1.0 / e_out)


# =========================================================
# Powered flyby
# =========================================================

def powered_flyby(v_inf_in, v_inf_out, mu, rp_min, rp_max, iterations=60):
    """
    Patched-conic flyby joining an incoming and an outgoing v-infinity.

    The spacecraft flies an incoming hyperbola (speed v_in) and an
    outgoing hyperbola (speed v_out) that share the same periapsis.
    If v_in != v_out a tangential burn at periapsis makes up the
    difference. A real unpowered gravity assist is the case where
    that burn is (close to) zero.

    Parameters
    ----------
    v_inf_in, v_inf_out : (..., 3) arrays [km/s], relative to the planet
    mu      : planet GM [km^3/s^2]
    rp_min  : smallest allowed periapsis radius [km] (planet radius + safety altitude)
    rp_max  : largest periapsis radius treated as a flyby [km] (e.g. sphere of influence)

    Returns
    -------
    rp       : periapsis radius [km] (nan where the turn needs rp < rp_min)
    delta_v  : periapsis burn [km/s] (nan where infeasible)
    delta    : turn angle [rad]
    feasible : bool array
    """

    v_in = np.linalg.norm(v_inf_in, axis=-1)
    v_out = np.linalg.norm(v_inf_out, axis=-1)

    delta = turn_angle(v_inf_in, v_inf_out)

    # The turn angle shrinks as rp grows. The largest turn possible
    # is at rp_min, so anything bigger than that cannot be flown.
    feasible = delta <= max_turn_angle(v_in, v_out, rp_min, mu)

    # -------------------------------------------------
    # Bisection on log(rp) between rp_min and rp_max
    #
    # Vectorised, so every combination is solved together.
    # -------------------------------------------------

    log_low = np.full(np.shape(delta), np.log(rp_min))
    log_high = np.full(np.shape(delta), np.log(rp_max))

    for _ in range(iterations):

        log_mid = 0.5 * (log_low + log_high)

        too_much_turn = max_turn_angle(v_in, v_out, np.exp(log_mid), mu) > delta

        # Too much turn means periapsis must move further out
        log_low = np.where(too_much_turn, log_mid, log_low)
        log_high = np.where(too_much_turn, log_high, log_mid)

    rp = np.exp(0.5 * (log_low + log_high))

    delta_v = np.abs(np.sqrt(v_out**2 + 2.0 * mu / rp) - np.sqrt(v_in**2 + 2.0 * mu / rp))

    rp = np.where(feasible, rp, np.nan)
    delta_v = np.where(feasible, delta_v, np.nan)

    return rp, delta_v, delta, feasible


# =========================================================
# Departure from a circular parking orbit
# =========================================================

def leo_burn(v_inf, altitude_km, mu=MU_EARTH, radius=R_EARTH):
    """
    Single tangential burn from a circular parking orbit onto the
    departure hyperbola.

    Returns (burn delta-v, speed after burn, circular speed) [km/s]
    """

    rp = radius + altitude_km

    v_circular = np.sqrt(mu / rp)
    v_periapsis = np.sqrt(v_inf**2 + 2.0 * mu / rp)

    return v_periapsis - v_circular, v_periapsis, v_circular


def ecliptic_to_equatorial(vector):
    """Rotate (..., 3) J2000 ecliptic vectors into J2000 equatorial."""

    c = np.cos(OBLIQUITY)
    s = np.sin(OBLIQUITY)

    vector = np.asarray(vector)

    x = vector[..., 0]
    y = vector[..., 1]
    z = vector[..., 2]

    return np.stack([x, c * y - s * z, s * y + c * z], axis=-1)


def right_ascension_declination(vector):
    """RA and Dec [deg] of (..., 3) equatorial vectors (the departure asymptote)."""

    vector = np.asarray(vector)
    norm = np.linalg.norm(vector, axis=-1)

    ra = np.degrees(np.arctan2(vector[..., 1], vector[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(vector[..., 2] / norm))

    return ra, dec


def leo_departure_state(v_inf_equatorial, altitude_km, inclination_deg, mu=MU_EARTH, radius=R_EARTH):
    """
    Example burn point for one departure: geocentric J2000 equatorial
    position, velocity before and after the LEO burn, and the burn vector.

    The parking orbit plane has to contain the departure asymptote, so
    its inclination must be at least |declination|. If the requested
    inclination is lower, |declination| is used instead.

    Returns position [km], velocity before [km/s], velocity after [km/s],
    burn vector [km/s], inclination used [deg]
    """

    v_inf_equatorial = np.asarray(v_inf_equatorial, dtype=float)

    v_inf = np.linalg.norm(v_inf_equatorial)
    S = v_inf_equatorial / v_inf

    _, dec = right_ascension_declination(S)
    inclination = max(inclination_deg, abs(dec) + 1e-9)

    rp = radius + altitude_km

    # True anomaly of the outgoing asymptote
    e = 1.0 + rp * v_inf**2 / mu
    theta = np.arccos(-1.0 / e)

    # Orbit normal for the ascending, prograde solution
    z_axis = np.array([0.0, 0.0, 1.0])
    A = z_axis - np.dot(z_axis, S) * S
    A /= np.linalg.norm(A)
    B = np.cross(S, A)

    cos_alpha = min(1.0, np.cos(np.radians(inclination)) / np.cos(np.radians(dec)))
    h = cos_alpha * A + np.sqrt(max(0.0, 1.0 - cos_alpha**2)) * B

    W = np.cross(h, S)

    # Periapsis (burn point) direction and flight direction there
    P = np.cos(theta) * S - np.sin(theta) * W
    Q = np.sin(theta) * S + np.cos(theta) * W

    _, v_periapsis, v_circular = leo_burn(v_inf, altitude_km, mu, radius)

    position = rp * P
    velocity_before = v_circular * Q
    velocity_after = v_periapsis * Q

    return position, velocity_before, velocity_after, velocity_after - velocity_before, inclination


if __name__ == "__main__":

    # Unpowered check: equal speeds, rp from the closed form must come back
    mu_jupiter = 1.26686534e8
    v = 10.0
    rp_true = 200000.0
    delta_true = 2.0 * np.arcsin(1.0 / (1.0 + rp_true * v**2 / mu_jupiter))

    v_in = np.array([v, 0.0, 0.0])
    v_out = v * np.array([np.cos(delta_true), np.sin(delta_true), 0.0])

    rp, dv, delta, ok = powered_flyby(v_in, v_out, mu_jupiter, 71492.0, 5.0e7)

    print("rp =", rp, "(expected", rp_true, ")")
    print("dv =", dv, "(expected 0)")
