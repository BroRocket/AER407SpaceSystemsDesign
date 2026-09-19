
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_FILE = "lambert_results_detailed.csv"


# def truncate_csv(filename, num_rows=400000):
#     df = pd.read_csv(filename)

#     print(f"Original number of rows: {len(df):,}")

#     if len(df) <= num_rows:
#         print("CSV already contains fewer than or equal to the requested number of rows.")
#         return

#     df = df.iloc[:num_rows]

#     df.to_csv(filename, index=False)

#     print(f"CSV truncated to: {len(df):,} rows")


# truncate_csv(CSV_FILE)
# exit()

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


df["departure_date"] = pd.to_datetime(df["departure_date"])
df["arrival_date"] = pd.to_datetime(df["arrival_date"])

delta_v = df["delta_v"].to_numpy()
relative_v = df["v_relative_comet"].to_numpy()

# Find minimums
min_delta_V_ind = np.argmin(delta_v)
min_relative_V_ind = np.argmin(relative_v)

min_dv_row = df.iloc[min_delta_V_ind]
min_rv_row = df.iloc[min_relative_V_ind]

# print("Minimum departure delta-V:")
# print(min_dv_row)
# print("\nMinimum arrival relative velocity:")
# print(min_rv_row)


DELTA_V_THRESHOLD = 40
RELATIVE_E_THRESHOLD = 8
# Extract possible solution trajectories:
candidates = []
min_dv = 40
min_dv_ind = 0
min_rv = 8
min_rv_ind = 0
min_total = 48
min_total_ind = 0
for i, (delta_v, relative_v) in enumerate(zip(df["delta_v"].to_list(), df["v_relative_comet"].to_list())):
    if delta_v <= DELTA_V_THRESHOLD and relative_v <= RELATIVE_E_THRESHOLD:
        candidates.append(df.iloc[i])
        if delta_v < min_dv:
            min_dv = delta_v
            min_dv_ind = i
        if relative_v < min_rv:
            min_rv = relative_v
            min_rv_ind = i
        if delta_v + relative_v <= min_total:
            min_total = delta_v + relative_v
            min_total_ind = i

# need to print minimum of delta v and relative speed
print(f"Number of candidates: {len(candidates)}")
print(min_dv)
print(df.iloc[min_dv_ind])
print(min_rv)
print(df.iloc[min_rv_ind])
print(min_total)
print(df.iloc[min_total_ind])




# =========================================================
# Create 2D grids
# =========================================================

departure_dates = np.sort(
    df["departure_date"].unique()
)

arrival_dates = np.sort(
    df["arrival_date"].unique()
)

delta_v_grid = df.pivot(
    index="arrival_date",
    columns="departure_date",
    values="delta_v"
)

delta_v_grid = delta_v_grid.reindex(
    index=arrival_dates,
    columns=departure_dates
)


# =========================================================
# Create mesh
# =========================================================

departure_mesh, arrival_mesh = np.meshgrid(
    departure_dates,
    arrival_dates
)

# =========================================================
# Delta-V porkchop plot
# =========================================================

fig, ax = plt.subplots(figsize=(12, 8))

Z = delta_v_grid.values

# ---------------------------------------------------------
# Choose colour scale
# ---------------------------------------------------------

# Minimum valid delta-V
vmin = np.nanmin(Z)

# Set the upper colour limit.
# Change this number depending on your data.
vmax = 40  # km/s

# Number of colour levels
levels = np.linspace(vmin, vmax, 100)


# ---------------------------------------------------------
# Filled contour
# ---------------------------------------------------------

contour = ax.contourf(
    departure_mesh,
    arrival_mesh,
    Z,
    levels=levels,
    extend="max"
)


# ---------------------------------------------------------
# Colourbar
# ---------------------------------------------------------

cbar = fig.colorbar(contour, ax=ax)

cbar.set_label(
    r"Departure $v_\infty$ (km/s)"
)


# ---------------------------------------------------------
# Contour lines
# ---------------------------------------------------------

# contour_lines = ax.contour(
#     departure_mesh,
#     arrival_mesh,
#     Z,
#     levels=np.arange(
#         np.ceil(vmin),
#         vmax + 1,
#         1.0
#     ),
#     linewidths=0.8
# )

# ax.clabel(
#     contour_lines,
#     inline=True,
#     fontsize=8,
#     fmt="%.0f"
# )


# ---------------------------------------------------------
# Minimum delta-V point and minimum relative
# ---------------------------------------------------------

ax.scatter(
    min_dv_row["departure_date"],
    min_dv_row["arrival_date"],
    marker="*",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum departure $v_\\infty$ = "
        f"{min_dv_row['delta_v']:.2f} km/s"
    )
)

ax.scatter(
    min_rv_row["departure_date"],
    min_rv_row["arrival_date"],
    marker="o",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum Relative Volcity = "
        f"{min_rv_row['v_relative_comet']:.2f} km/s"
    )
)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

ax.set_xlabel("Departure Date")
ax.set_ylabel("Arrival Date")

ax.set_title(
    "Earth → 3I/ATLAS Departure $v_\\infty$ Porkchop Plot"
)


# ---------------------------------------------------------
# Date formatting
# ---------------------------------------------------------

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.yaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.yaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend()

plt.tight_layout()
plt.show()


### PLot of relative arrival velcoity 

relative_v_grid = df.pivot(
    index="arrival_date",
    columns="departure_date",
    values="v_relative_comet"
)

relative_v_grid = relative_v_grid.reindex(
    index=arrival_dates,
    columns=departure_dates
)


# =========================================================
# Create mesh
# =========================================================

departure_mesh, arrival_mesh = np.meshgrid(
    departure_dates,
    arrival_dates
)

# =========================================================
# relative V porkchop plot
# =========================================================

fig, ax = plt.subplots(figsize=(12, 8))

Z = relative_v_grid.values

# ---------------------------------------------------------
# Choose colour scale
# ---------------------------------------------------------

# Minimum valid delta-V
vmin = np.nanmin(Z)

# Set the upper colour limit.
# Change this number depending on your data.
vmax = 12  # km/s

# Number of colour levels
levels = np.linspace(vmin, vmax, 100)


# ---------------------------------------------------------
# Filled contour
# ---------------------------------------------------------

contour = ax.contourf(
    departure_mesh,
    arrival_mesh,
    Z,
    levels=levels,
    extend="max"
)

# ---------------------------------------------------------
# Colourbar
# ---------------------------------------------------------

cbar = fig.colorbar(contour, ax=ax)

cbar.set_label(
    r"Relative Velocity Comet $v_R$ (km/s)"
)

# ---------------------------------------------------------
# Minimum delta-V point and minimum relative
# ---------------------------------------------------------

ax.scatter(
    min_dv_row["departure_date"],
    min_dv_row["arrival_date"],
    marker="*",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum departure $v_\\infty$ = "
        f"{min_dv_row['delta_v']:.2f} km/s"
    )
)

ax.scatter(
    min_rv_row["departure_date"],
    min_rv_row["arrival_date"],
    marker="o",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum Relative Volcity = "
        f"{min_rv_row['v_relative_comet']:.2f} km/s"
    )
)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

ax.set_xlabel("Departure Date")
ax.set_ylabel("Arrival Date")

ax.set_title(
    "Earth → 3I/ATLAS Departure Relative Comet Velocity Plot"
)

# ---------------------------------------------------------
# Date formatting
# ---------------------------------------------------------

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.yaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.yaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend()

plt.tight_layout()
plt.show()


### Joint velocity
df["total_dv"] = df["delta_v"] + df["v_relative_comet"]

### PLot of joint velocity

total_v_grid = df.pivot(
    index="arrival_date",
    columns="departure_date",
    values="total_dv"
)

total_v_grid = total_v_grid.reindex(
    index=arrival_dates,
    columns=departure_dates
)


# =========================================================
# Create mesh
# =========================================================

departure_mesh, arrival_mesh = np.meshgrid(
    departure_dates,
    arrival_dates
)

# =========================================================
# Delta-V porkchop plot
# =========================================================

fig, ax = plt.subplots(figsize=(12, 8))

Z = total_v_grid.values

# ---------------------------------------------------------
# Choose colour scale
# ---------------------------------------------------------

# Minimum valid delta-V
vmin = np.nanmin(Z)

# Set the upper colour limit.
# Change this number depending on your data.
vmax = 300  # km/s

# Number of colour levels
levels = np.linspace(vmin, vmax, 200)


# ---------------------------------------------------------
# Filled contour
# ---------------------------------------------------------

contour = ax.contourf(
    departure_mesh,
    arrival_mesh,
    Z,
    levels=levels,
    extend="max"
)

# ---------------------------------------------------------
# Colourbar
# ---------------------------------------------------------

cbar = fig.colorbar(contour, ax=ax)

cbar.set_label(
    r"Total Velocity $v_T$ (km/s)"
)

# ---------------------------------------------------------
# Minimum delta-V point and minimum relative
# ---------------------------------------------------------

ax.scatter(
    min_dv_row["departure_date"],
    min_dv_row["arrival_date"],
    marker="*",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum departure $v_\\infty$ = "
        f"{min_dv_row['delta_v']:.2f} km/s"
    )
)

ax.scatter(
    min_rv_row["departure_date"],
    min_rv_row["arrival_date"],
    marker="o",
    s=250,
    edgecolor="black",
    linewidth=1.0,
    label=(
        f"Minimum Relative Volcity = "
        f"{min_rv_row['v_relative_comet']:.2f} km/s"
    )
)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

ax.set_xlabel("Departure Date")
ax.set_ylabel("Arrival Date")

ax.set_title(
    "Earth → 3I/ATLAS Departure Relative Comet Velocity Plot"
)

# ---------------------------------------------------------
# Date formatting
# ---------------------------------------------------------

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.yaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.yaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend()

plt.tight_layout()
plt.show()