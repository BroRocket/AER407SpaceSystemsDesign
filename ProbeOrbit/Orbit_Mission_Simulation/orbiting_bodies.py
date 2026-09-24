
import numpy as np
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

class CelestialBody:
    def __init__(self, name :str, mu_self : float, ephemeris: Ephemeris,  jpl_horizons_id :str, start_date: datetime, end_date: datetime, vec_type: int = 1):
        self.name = name
        self.mu = mu_self #m^3/s^2
        self.id = jpl_horizons_id
        self.orbit = ephemeris
        self.vec_type = vec_type

        self.start_date = start_date
        self.end_date = end_date
        # this line is fucked to change lol, but just be carfeul you map the roight amoutn of time in each second or orbits of bodies and probe wont't match
        #lower line gives days for less compute time
        self.times_in_seconds = [((start_date + timedelta(minutes=i*30)) - start_date).total_seconds() for i in range(int((end_date - start_date).total_seconds()/(60*30)) + 1)]
        #self.times_in_seconds = [((start_date + timedelta(days=i)) - start_date).total_seconds() for i in range((end_date - start_date).days + 1)]


        states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'"), kwargs={"STEP_SIZE": "'30m'"}, metric=True)
        #states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'"), kwargs={"STEP_SIZE": "'1d'"}, metric=True)

        r_matrix = np.array([state["r"] for state in states])
        self.r = [r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2]]

        if vec_type == 2:
            v_matrix = np.array([state["v"] for state in states])
            self.v = [v_matrix[:, 0], v_matrix[:, 1], v_matrix[:, 2]]

            self.interpx = CubicHermiteSpline(self.times_in_seconds, self.r[0], self.v[0], extrapolate=False)
            self.interpy = CubicHermiteSpline(self.times_in_seconds, self.r[1], self.v[1], extrapolate=False)
            self.interpz = CubicHermiteSpline(self.times_in_seconds, self.r[2], self.v[2], extrapolate=False)

            self.interpvx = self.interpx.derivative()
            self.interpvy = self.interpy.derivative()
            self.interpvz = self.interpz.derivative()
        else:
            self.v = [np.array([]), np.array([]), np.array([])]

    def get_state(self, t: float):
        x = float(self.interpx(t))
        y = float(self.interpy(t))
        z = float(self.interpz(t))
        return np.array([x, y, z])

    def get_velocity(self, t):
        return np.array([float(self.interpvx(t)), float(self.interpvy(t)), float(self.interpvz(t))])


class Spacecraft:
    def __init__(self, name, init_position, init_velocity, acceleration = None):
        self.name = name

        self.r0 = np.array(init_position, dtype=float)
        self.v0 = np.array(init_velocity, dtype=float)
        self.r = []
        self.v = []
        self.accel = acceleration

    def propogate_orbit(self, t_start: float, t_final: float, mu_central: float, celestial_bodies : list[CelestialBody] = [], t_eval=None):

        def dynamics(t, state):
            r = state[:3]
            v = state[3:]

            r_mag = np.linalg.norm(r)

            a = (-mu_central * (r / r_mag**3))

            comet_vel = None
            for body in celestial_bodies:
                r_body = body.get_state(t)
                r_body_mag = np.linalg.norm(r_body)
                if body.name == "3I/ATLAS":
                    comet_vel = body.get_velocity(t)
                r_body_sc_mag = np.linalg.norm(r_body - r)
                a += body.mu * (((r_body - r) / (r_body_sc_mag**3)) - (r_body / r_body_mag**3))

            # add thurst
            if self.accel is not None and comet_vel is not None:
                v_rel = comet_vel - v
                mag = np.linalg.norm(v_rel)
                if mag > 0:
                    v_unit = v_rel/mag
                a += self.accel(t) * v_unit

            return np.concatenate([v, a])

        self.orbit_solutions = solve_ivp(dynamics, (t_start, t_final), np.concatenate([self.r0, self.v0]), method="DOP853", t_eval=t_eval, rtol=1e-11, atol=[1e-3, 1e-3, 1e-3, 1e-7, 1e-7, 1e-7])
        self.t = self.orbit_solutions.t
        self.r = self.orbit_solutions.y[:3]
        self.v = self.orbit_solutions.y[3:6]
        # boundaries = [t_start]

        # for burn_start, burn_end in burn_times:
        #     if t_start < burn_start < t_final:
        #         boundaries.append(burn_start)

        #     if t_start < burn_end < t_final:
        #         boundaries.append(burn_end)

        # boundaries.append(t_final)

        # boundaries = sorted(set(boundaries))


        # # --------------------------------------------------
        # # Integrate each segment
        # # --------------------------------------------------

        # state = np.concatenate([self.r0, self.v0])

        # all_t = []
        # all_r = []
        # all_v = []

        # for t0, tf in zip(boundaries[:-1], boundaries[1:]):

        #     sol = solve_ivp(
        #         dynamics,
        #         (t0, tf),
        #         state,
        #         method="DOP853",
        #         rtol=1e-11,
        #         atol=[1e-3, 1e-3, 1e-3, 1e-7, 1e-7, 1e-7]
        #     )

        #     # Save results
        #     if len(all_t) == 0:
        #         all_t.append(sol.t)
        #         all_r.append(sol.y[:3])
        #         all_v.append(sol.y[3:6])
        #     else:
        #         # Don't duplicate boundary point
        #         all_t.append(sol.t[1:])
        #         all_r.append(sol.y[:3, 1:])
        #         all_v.append(sol.y[3:6, 1:])

        #     # Final state becomes initial state of next segment
        #     state = sol.y[:, -1]

        # # --------------------------------------------------
        # # Store trajectory
        # # --------------------------------------------------

        # self.t = np.concatenate(all_t)
        # self.r = np.concatenate(all_r, axis=1)
        # self.v = np.concatenate(all_v, axis=1)

        # self.orbit_solutions = sol