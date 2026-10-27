
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

def datetime_to_jd(date: datetime) -> float:
    """Julian date of a datetime (same time scale as the datetime, e.g. TDB if your csv is TDB)."""
    midnight = datetime(date.year, date.month, date.day)
    return date.toordinal() + 1721424.5 + (date - midnight).total_seconds() / 86400.0

class CelestialBody:
    def __init__(self, name :str, mu_self : float, start_date: datetime, end_date: datetime, ephemeris: Ephemeris|None = None, jpl_id: str|None = None, csv_file: str|None = None):
        self.name = name
        self.mu = mu_self #m^3/s^2
        self.start_date = start_date
        self.end_date = end_date

        if csv_file is not None:
            self.start_date_jdn =  datetime_to_jd(start_date)
            self.end_date_jdn = datetime_to_jd(end_date)

            headers = {"epoch": np.float64, "x": np.float64, "y": np.float64, "z": np.float64, "vx": np.float64, "vy": np.float64, "vz": np.float64}
            df = pd.read_csv(csv_file, dtype=headers)

            if df["epoch"][0] > self.start_date_jdn:
                raise ValueError(f"Start date is before earliest date in csv")
            elif df["epoch"].iloc[-1] < self.end_date_jdn:
                raise ValueError(f"End date is after last date in csv")

            epochs = df["epoch"].to_numpy()
            first_row = max(np.searchsorted(epochs, self.start_date_jdn, side="right") - 1, 0)
            last_row = min(np.searchsorted(epochs, self.end_date_jdn, side="left") + 1, len(epochs))
            df_filtered_date_window = df.iloc[first_row:last_row]

            self.times_in_seconds = (df_filtered_date_window["epoch"].to_numpy() - self.start_date_jdn) * 86400.0

            df_filtered_date_window = df[df["epoch"].between(self.start_date_jdn, self.end_date_jdn)]
            #Annoyingly csvs in km :(
            self.r = [df_filtered_date_window["x"].to_numpy()*1000, df_filtered_date_window["y"].to_numpy()*1000, df_filtered_date_window["z"].to_numpy()*1000]
            self.v = [df_filtered_date_window["vx"].to_numpy()*1000, df_filtered_date_window["vy"].to_numpy()*1000, df_filtered_date_window["vz"].to_numpy()*1000]

        elif ephemeris is not None and jpl_id is not None:
            self.orbit = ephemeris
            self.id = jpl_id

            self.times_in_seconds = [((start_date + timedelta(minutes=i*30)) - start_date).total_seconds() for i in range(int((end_date - start_date).total_seconds()/(60*30)) + 1)]

            states = self.orbit.get_state(self.id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'"), kwargs={"STEP_SIZE": "'30m'"}, metric=True)
            
            r_matrix = np.array([state["r"] for state in states])
            self.r = [r_matrix[:, 0], r_matrix[:, 1], r_matrix[:, 2]]
            
            v_matrix = np.array([state["v"] for state in states])
            self.v = [v_matrix[:, 0], v_matrix[:, 1], v_matrix[:, 2]]

        else:
            raise ValueError(f"No Ephemeris file or api object provided for celestial body: {self.name}")

        self.interpx = CubicHermiteSpline(self.times_in_seconds, self.r[0], self.v[0], extrapolate=False)
        self.interpy = CubicHermiteSpline(self.times_in_seconds, self.r[1], self.v[1], extrapolate=False)
        self.interpz = CubicHermiteSpline(self.times_in_seconds, self.r[2], self.v[2], extrapolate=False)

        self.interpvx = self.interpx.derivative()
        self.interpvy = self.interpy.derivative()
        self.interpvz = self.interpz.derivative()

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

        self.orbit_solutions = solve_ivp(dynamics, (t_start, t_final), np.concatenate([self.r0, self.v0]), method="DOP853", t_eval=t_eval, rtol=1e-9, atol=[1e-3, 1e-3, 1e-3, 1e-7, 1e-7, 1e-7])
        self.t = self.orbit_solutions.t
        self.r = self.orbit_solutions.y[:3]
        self.v = self.orbit_solutions.y[3:6]


