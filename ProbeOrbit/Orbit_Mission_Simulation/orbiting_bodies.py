
import numpy as np
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp

from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

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

        states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'"), kwargs={"STEP_SIZE": "'1d'"}, metric=True)

        # self.r = [[], [], []]
        # self.v = [[], [], []]
        # for state in states:
        #     self.r[0].append(state["r"][0])
        #     self.r[1].append(state["r"][1])
        #     self.r[2].append(state["r"][2])
        #     if vec_type == 2:
        #         self.v[0].append(state["v"][0])
        #         self.v[1].append(state["v"][1])
        #         self.v[2].append(state["v"][2])

        # self.r = [np.array(row) for row in self.r]
        # self.v = [np.array(row) for row in self.v]

        # Shape: (N, 3) where N is len(states)
        r_matrix = np.array([state["r"] for state in states])
        self.r = [r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2]]

        if vec_type == 2:
            v_matrix = np.array([state["v"] for state in states])
            self.v = [v_matrix[:, 0], v_matrix[:, 1], v_matrix[:, 2]]
        else:
            self.v = [np.array([]), np.array([]), np.array([])]

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

        self.orbit_solutions = solve_ivp(dynamics, (t_start, t_final), np.concatenate([self.r0, self.v0]), method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-10)
        self.r = self.orbit_solutions.y[:3]
        self.v = self.orbit_solutions.y[3:6]


class CelestialBody2:
    def __init__(self, name :str, mu_self : float, ephemeris: Ephemeris,  jpl_horizons_id :str, start_date: datetime):
        self.name = name
        self.mu = mu_self #m^4/s^2
        self.id = jpl_horizons_id
        self.orbit = ephemeris

        self.start_date = start_date

        states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), start_date.strftime("'%Y-%m-%d'"))

        self.r0 = states[0]["r"] 
        self.v0 = states[0]["v"]

    def get_state(self, time: float):
        for i, date in enumerate(self.dates):
            if date == time:
                break

        return (self.r[i], self.v[i])

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
    
            self.orbit_solutions = solve_ivp(dynamics, (t_start, t_final), np.concatenate([self.r0, self.v0]), method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-10)
            self.r = self.orbit_solutions.y[:3]
            self.v = self.orbit_solutions.y[3:6]