
import pandas as pd
import numpy as np
import datetime

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

AU = 149597870700  # m

orbits = Ephemeris()

CSV_FILE = "lambert_results_detailed.csv"


df = pd.read_csv(CSV_FILE, dtype={"departure_date": str, 
                                  "arrival_date": str,
                                  "time_of_flight_days": np.float64,
                                  "delta_v": np.float64,
                                  "dv_x": np.float64,
                                  "dv_y": np.float64,
                                  "dv_z": np.float64,
                                  "dv_unit_x": np.float64,
                                  "dv_unit_y": np.float64,
                                  "dv_unit_z": np.float64,
                                  "v_relative_comet": np.float64,
                                  "vrel_x": np.float64,
                                  "vrel_y": np.float64,
                                  "vrel_z": np.float64})

df['arrival_date_dt'] = pd.to_datetime(df['arrival_date'])

# Filter for delta_v < 2.3 km/s
filtered_df = df[df["v_relative_comet"] < 1.68]

# Find the row with the earliest arrival date
earliest_arrival = filtered_df.loc[
    filtered_df["arrival_date_dt"].idxmin()
]

print(earliest_arrival)

departurte_date = earliest_arrival["arrival_date_dt"] 
arrival_date =  earliest_arrival["arrival_date_dt"] + datetime.timedelta(days=10)

comet = CelestialBody("3I/ATLAS", 2202.52, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
print("Loaded Comet")
print(f"Distance from Sun (AU): {np.linalg.norm([comet.r[0][0], comet.r[1][0], comet.r[2][0]])/AU}")

filtered_df = df[df['arrival_date_dt'] < datetime.datetime(2028, 3, 21)]

optimum = filtered_df.loc[
    filtered_df["v_relative_comet"].idxmin()
]

departurte_date = datetime.datetime(2025, 10, 1)
arrival_date =  optimum["arrival_date_dt"]

comet = CelestialBody("3I/ATLAS", 2202.52, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
print("Loaded Comet")
print(f'Comet r: {[comet.r[0][-1], comet.r[1][-1], comet.r[2][-1]]}')
print(f"Distance from Sun (AU): {np.linalg.norm([comet.r[0][-1], comet.r[1][-1], comet.r[2][-1]])/AU}")


print(optimum)
