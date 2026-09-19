
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "lambert_results_detailed.csv"

FILTERED_CSV = "lambert_sep_oct_to_dec_jan.csv"
BEST_CSV = "lambert_best_trajectories.csv"

# Departure window: September–October 2025
DEPARTURE_START = "2025-09-01"
DEPARTURE_END   = "2025-10-31"

# Arrival window: December 2025–January 2026
ARRIVAL_START = "2025-12-01"
ARRIVAL_END   = "2026-01-31"


# ============================================================
# LOAD AND FILTER
# ============================================================

df = pd.read_csv(INPUT_CSV)

# Convert dates to datetime before filtering
df["departure_date"] = pd.to_datetime(df["departure_date"])
df["arrival_date"] = pd.to_datetime(df["arrival_date"])

departure_start = pd.Timestamp(DEPARTURE_START)
departure_end = pd.Timestamp(DEPARTURE_END)

arrival_start = pd.Timestamp(ARRIVAL_START)
arrival_end = pd.Timestamp(ARRIVAL_END)

# Select rows within BOTH date windows
filtered_df = df[
    df["departure_date"].between(departure_start, departure_end)
    &
    df["arrival_date"].between(arrival_start, arrival_end)
].copy()

print(f"Total trajectories in original CSV: {len(df):,}")
print(f"Trajectories matching date ranges: {len(filtered_df):,}")

if filtered_df.empty:
    raise ValueError("No trajectories found in the specified date ranges.")


# ============================================================
# SAVE ALL MATCHING TRAJECTORIES
# ============================================================

filtered_df.to_csv(FILTERED_CSV, index=False)

print(f"Filtered results saved to: {FILTERED_CSV}")


# ============================================================
# FIND MINIMUM DEPARTURE DELTA-V
# ============================================================

best_dv = filtered_df.loc[
    filtered_df["delta_v"].idxmin()
]


# ============================================================
# FIND MINIMUM COMET-RELATIVE ARRIVAL VELOCITY
# ============================================================

best_relative_v = filtered_df.loc[
    filtered_df["v_relative_comet"].idxmin()
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_trajectory(row, title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    print(f"Departure date: {row['departure_date']}")
    print(f"Arrival date:   {row['arrival_date']}")

    print(
        f"Time of flight: "
        f"{row['time_of_flight_days']:.2f} days"
    )

    print(f"Departure delta-v: {row['delta_v']:.4f} km/s")
    print(
        f"Comet-relative arrival speed: "
        f"{row['v_relative_comet']:.4f} km/s"
    )


print_trajectory(
    best_dv,
    "MINIMUM DEPARTURE DELTA-V TRAJECTORY"
)

print_trajectory(
    best_relative_v,
    "MINIMUM COMET-RELATIVE ARRIVAL VELOCITY TRAJECTORY"
)


# ============================================================
# SAVE BOTH OPTIMUM TRAJECTORIES
# ============================================================

best_trajectories = pd.DataFrame([
    best_dv,
    best_relative_v
])

best_trajectories.insert(
    0,
    "optimization",
    [
        "minimum_departure_delta_v",
        "minimum_comet_relative_arrival_velocity"
    ]
)

best_trajectories.to_csv(BEST_CSV, index=False)

print(f"\nBest trajectories saved to: {BEST_CSV}")