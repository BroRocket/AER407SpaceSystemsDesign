
import numpy as np
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp

from Orbital_Mission_Planning.ephemeris import Ephemeris

class CelestialBody:
    def __init__(self, name :str, mu_self : float, ephemeris: Ephemeris,  jpl_horizons_id :str, start_date: datetime, end_date: datetime, vec_type: int = 1):
        self.name = name
        self.mu = mu_self #m^4/s^2
        self.id = jpl_horizons_id
        self.orbit = ephemeris
        self.vec_type = vec_type

        self.start_date = start_date
        self.end_date = end_date
        self.dates = [(start_date + timedelta(minutes=i)).strftime("%Y-%m-%d") for i in range(int((end_date - start_date).total_seconds()/60) + 1)]

        states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'"), kwargs={"STEP_SIZE": "'10m'", "VEC_TABLE": f"'{vec_type}'"}, metric=True)

        self.r = [state["r"] for state in states]
        if vec_type == 2:
            self.v = [state["v"] for state in states]

    def get_state(self, time: datetime):
        for i, date in enumerate(self.dates):
            if date == time:
                break

        return (self.r[i], self.v[i])


class Spacecraft:
    def __init__(self, name, init_position, init_velocity):
        self.name = name

        self.r0 = np.array(init_position, dtype=float)
        self.v0 = np.array(init_velocity, dtype=float)
        self.r = []
        self.v = []

    def initial_state(self):
        return np.concatenate([self.r0, self.v0])

    def propogate_orbit(self, t_start: float, t_final: float, mu_central: float, celestial_bodies : list[CelestialBody] = [], t_eval=None):

        def dynamics(t, state):
            r = state[:3]
            v = state[3:]

            r_mag = np.linalg.norm(r)

            a = (-mu_central * (r / r_mag**3))

            #for body in celestial_bodies:
            #    a += body.mu * (((body. r)/()) - ())

            # add thurst

            return np.concatenate([v, a])

        self.orbit_solutions = solve_ivp(dynamics, (t_start, t_final), self.initial_state(), method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-10)
        self.r = self.orbit_solutions.y[:3]
        self.v = self.orbit_solutions.y[3:6]


    