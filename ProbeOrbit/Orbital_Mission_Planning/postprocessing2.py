
import pandas as pd
import numpy as np

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
filtered_df = df[df["v_relative_comet"] < 2.3]

# Find the row with the earliest arrival date
earliest_arrival = filtered_df.loc[
    filtered_df["arrival_date_dt"].idxmin()
]

print(earliest_arrival)
