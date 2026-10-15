"""Departure state + differential correction for the N-body simulation.

Works with the CelestialBody objects in orbiting_bodies.py (anything with
.name, .mu [m^3/s^2] and .get_state(t) -> heliocentric position [m], where t is
seconds after the simulation start date). SI units throughout, J2000 ecliptic.

Why this is needed
------------------
The trajectory search uses patched conics: Sun-only arcs between planet
*centres* and an instantaneous flyby. In a full N-body model the same
departure velocity does not reproduce the flyby, because
  1. the probe has to climb out of Earth's gravity well (start it at the
     burn point with the post-burn speed, not with the heliocentric velocity
     that applies far from Earth), and
  2. the other planets and Jupiter's own pull on the approach shift the
     arrival point by thousands of km, while the flyby needs ~100 km accuracy.
So the initial velocity has to be corrected (shooting / Newton's method) until
the probe passes through the intended point near Jupiter, and then until it
meets the comet.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

OBLIQUITY = math.radians(23.439291111)   # J2000 mean obliquity
MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0


def equ_to_ecl(v):
    c, s = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    x, y, z = v
    return np.array([x, c*y + s*z, -s*y + c*z])


def ecl_to_equ(v):
    c, s = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    x, y, z = v
    return np.array([x, c*y - s*z, s*y + c*z])


def leo_departure_state(r_earth, v_earth, vinf_ecl, altitude_m=400e3, inclination_deg=28.5):
    """Heliocentric state right after the LEO burn that leaves Earth with v_inf.

    The parking orbit is circular at altitude_m, inclined inclination_deg to
    Earth's equator, and its plane contains the outgoing asymptote. Returns
    (r, v) in J2000 ecliptic, metres and m/s, plus the burn magnitude.
    """
    S_equ = ecl_to_equ(np.asarray(vinf_ecl, float))
    vinf = np.linalg.norm(S_equ); S = S_equ/vinf
    dec = math.asin(S[2])
    inc = max(math.radians(inclination_deg), abs(dec) + 1e-9)
    rp = R_EARTH + altitude_m
    e = 1 + rp*vinf**2/MU_EARTH
    th = math.acos(-1/e)                       # asymptote true anomaly
    z = np.array([0., 0., 1.])
    A = z - (z @ S)*S; A /= np.linalg.norm(A)
    Bv = np.cross(S, A)
    ca = min(1., math.cos(inc)/math.cos(dec))
    h = ca*A + math.sqrt(max(0., 1 - ca*ca))*Bv
    W = np.cross(h, S)
    P = math.cos(th)*S - math.sin(th)*W        # burn point direction
    Q = math.sin(th)*S + math.cos(th)*W        # flight direction there
    vp = math.sqrt(vinf**2 + 2*MU_EARTH/rp)
    burn = vp - math.sqrt(MU_EARTH/rp)
    r = np.asarray(r_earth, float) + equ_to_ecl(rp*P)
    v = np.asarray(v_earth, float) + equ_to_ecl(vp*Q)
    return r, v, burn


class NBody:
    """Sun + bodies point-mass dynamics (same equations as Spacecraft.propogate_orbit)."""

    def __init__(self, bodies, mu_sun=1.32712440041279419e20, rtol=1e-12):
        self.bodies, self.mu_sun, self.rtol = bodies, mu_sun, rtol
        self.by_name = {b.name: b for b in bodies}

    def rhs(self, t, y):
        r = y[:3]
        a = -self.mu_sun*r/np.linalg.norm(r)**3
        for b in self.bodies:
            rb = b.get_state(t); d = rb - r
            a += b.mu*(d/np.linalg.norm(d)**3 - rb/np.linalg.norm(rb)**3)
        return np.concatenate([y[3:], a])

    def propagate(self, y0, t0, t1, dense=False):
        sol = solve_ivp(self.rhs, (t0, t1), np.asarray(y0, float), method='DOP853', rtol=self.rtol,
                        atol=[1e-3]*3 + [1e-8]*3, dense_output=dense)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol if dense else sol.y[:, -1]

    def closest_approach(self, sol, body, t0, t1, n=20001):
        b = self.by_name[body]
        ts = np.linspace(t0, t1, n)
        d = np.array([np.linalg.norm(sol.sol(t)[:3] - b.get_state(t)) for t in ts])
        k = int(np.argmin(d))
        res = minimize_scalar(lambda t: np.linalg.norm(sol.sol(t)[:3] - b.get_state(t)),
                              bounds=(ts[max(k-1, 0)], ts[min(k+1, n-1)]), method='bounded',
                              options=dict(xatol=1e-3))
        y = sol.sol(res.x)
        vb = b.get_velocity(res.x) if hasattr(b, 'get_velocity') else None
        return dict(t=float(res.x), distance_m=float(res.fun),
                    relative_speed_m_s=float(np.linalg.norm(y[3:] - vb)) if vb is not None else None)


def correct_velocity(nb, y0, t0, t1, target_r, tol_m=100.0, max_iter=20, step=1e-3, verbose=True):
    """Newton iteration on the velocity at t0 so that r(t1) = target_r.

    Use it from launch (t0 = launch) or for a trajectory-correction manoeuvre
    (t0 = some time after the flyby; the returned velocity minus the coasting
    velocity is the burn).

    3 unknowns (velocity components), 3 equations (position at t1).
    Finite-difference Jacobian with velocity perturbations of `step` m/s.
    """
    y = np.array(y0, float)
    for it in range(max_iter):
        r1 = nb.propagate(y, t0, t1)[:3]
        miss = r1 - target_r
        if verbose:
            print(f'    iteration {it}: miss {np.linalg.norm(miss)/1e3:,.3f} km')
        if np.linalg.norm(miss) < tol_m:
            return y, float(np.linalg.norm(miss))
        Jm = np.zeros((3, 3))
        for k in range(3):
            yk = y.copy(); yk[3 + k] += step
            Jm[:, k] = (nb.propagate(yk, t0, t1)[:3] - r1)/step
        dv = np.linalg.solve(Jm, -miss)
        # Damped Newton: halve the step until the miss actually shrinks.
        for _ in range(12):
            trial = y.copy(); trial[3:] += dv
            if np.linalg.norm(nb.propagate(trial, t0, t1)[:3] - target_r) < np.linalg.norm(miss):
                break
            dv /= 2
        y = trial
    raise RuntimeError(f'No convergence: miss {np.linalg.norm(miss)/1e3:.1f} km')


def jupiter_periapsis_vector(v_in_helio, v_out_helio, v_jupiter, rp_m):
    """Jupiter-centric periapsis position of the patched-conic flyby hyperbola."""
    vin = np.asarray(v_in_helio) - v_jupiter
    vout = np.asarray(v_out_helio) - v_jupiter
    P = vin/np.linalg.norm(vin) - vout/np.linalg.norm(vout)
    return rp_m*P/np.linalg.norm(P)


def jupiter_approach_target(r_jupiter_at_tc, v_in_helio, v_out_helio, v_jupiter, rp_m, mu_planet, dt_before_s):
    """Heliocentric point the probe should pass dt_before_s before periapsis.

    The patched-conic arcs end at Jupiter's CENTRE. Aiming the N-body probe at
    that point flies it into Jupiter, so instead aim at the incoming leg of the
    flyby hyperbola, a few days out, where the trajectory is still smooth.
    r_jupiter_at_tc is Jupiter's heliocentric position at t_peri - dt_before_s.
    """
    vin = np.asarray(v_in_helio) - v_jupiter
    vout = np.asarray(v_out_helio) - v_jupiter
    P = jupiter_periapsis_vector(v_in_helio, v_out_helio, v_jupiter, rp_m)
    h = np.cross(vin, vout); h /= np.linalg.norm(h)
    vinf = 0.5*(np.linalg.norm(vin) + np.linalg.norm(vout))
    vp = math.sqrt(vinf**2 + 2*mu_planet/rp_m)
    Q = np.cross(h, P/np.linalg.norm(P))
    f = lambda t, y: np.concatenate([y[3:], -mu_planet*y[:3]/np.linalg.norm(y[:3])**3])
    back = solve_ivp(f, (0, -dt_before_s), np.r_[P, vp*Q], method='DOP853', rtol=1e-12, atol=1e-6).y[:3, -1]
    return np.asarray(r_jupiter_at_tc) + back
