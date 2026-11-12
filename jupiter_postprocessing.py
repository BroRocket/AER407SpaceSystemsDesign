
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

AU = 149597870700  # m

orbits = Ephemeris()

CSV_FILE = os.path.join("Jupiter Gravity Assist Data", "jupiter_assist_results.csv")

df = pd.read_csv(CSV_FILE, dtype={"departure_date": str, 
                                  "jupiter_date": str,
                                  "arrival_date": str,
                                  "leg1_days": np.float64,
                                  "leg2_days": np.float64,
                                  "total_days": np.float64,
                                  "leo_burn_dv": np.float64,
                                  "c3": np.float64,
                                  "vinf_departure": np.float64,
                                  "vinf_departure_x": np.float64,
                                  "vinf_departure_y": np.float64,
                                  "vinf_departure_z": np.float64,
                                  "vinf_jupiter_in": np.float64,
                                  "vinf_jupiter_out": np.float64,
                                  "turn_angle_deg": np.float64,
                                  "periapsis_radius_km" : np.float64,
                                  "periapsis_altitude_km" : np.float64,
                                  "flyby_dv" : np.float64,
                                  "arrival_speed" : np.float64,
                                  "vrel_x" : np.float64,
                                  "vrel_y" : np.float64,
                                  "vrel_z" : np.float64,
                                  "comet_distance_au" : np.float64,
                                  "vinf_departure_eq_x" : np.float64,
                                  "vinf_departure_eq_y" : np.float64,
                                  "vinf_departure_eq_z" : np.float64,
                                  "asymptote_ra_deg" : np.float64,
                                  "asymptote_dec_deg" : np.float64,
                                  "onboard_dv" : np.float64
                                  })

df['departure_date_dt'] = pd.to_datetime(df['departure_date'])
df['jupiter_date_dt'] = pd.to_datetime(df['jupiter_date'])
df['arrival_date_dt'] = pd.to_datetime(df['arrival_date'])

earliest_arrival = df.loc[df["arrival_date_dt"].idxmin()]
closest_comet_distance = df.loc[df["comet_distance_au"].idxmin()]

min_arrival_dv = df.loc[df["arrival_speed"].idxmin()]
min_total_dv = df.loc[df["onboard_dv"].idxmin()]

filtered_df = df[df["onboard_dv"] < 1.0]
closest_comet_distance_filtered = filtered_df.loc[filtered_df["comet_distance_au"].idxmin()]

columns = ["departure_date", "jupiter_date", "arrival_date", "leo_burn_dv", "periapsis_altitude_km",
               "flyby_dv", "arrival_speed", "onboard_dv", "comet_distance_au"]

print(60*"=")
print("Earliest arrival date:")
print(earliest_arrival[columns])

print(60*"=")
print("Closest Arrival Distance:")
print(closest_comet_distance[columns])

print(60*"=")
print("Min Arrival DV:")
print(min_arrival_dv[columns])

print(60*"=")
print("Min Total DV:")
print(min_total_dv[columns])

print(60*"=")
print("Min Arrival under 1.05km/s delta-v:")
print(closest_comet_distance_filtered[columns])

print(60*"=")
print("Min Arrival under 1.05km/s delta-v (Full print out):")
print(closest_comet_distance_filtered)
print(f'Wanted Index {filtered_df["comet_distance_au"].idxmin()}')
print(f'Wanted Index Min DV {filtered_df["onboard_dv"].idxmin()}')
filtered_df.to_csv("Filetered_jupiter_flyby.csv")


#make plot here

plot_df = df.dropna(
    subset=["departure_date_dt", "arrival_date_dt", "onboard_dv"]
)

fig, ax = plt.subplots(figsize=(12, 8))

sc = ax.scatter(
    plot_df["departure_date_dt"],
    plot_df["arrival_date_dt"],
    c=plot_df["onboard_dv"],
    cmap="viridis",
    s=20,
    marker="s"
)

# Colour bar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Total Onboard Δv (km/s)")  # Adjust units if necessary

# Axis labels and title
ax.set_xlabel("Departure Date")
ax.set_ylabel("Arrival Date")
ax.set_title("Onboard Δv Heatmap: Departure vs Arrival Date")

# Format date axes
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate()

ax.grid(False)
plt.tight_layout()
plt.show()