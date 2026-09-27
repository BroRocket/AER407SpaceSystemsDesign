from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_FOLDER = Path(
    r"C:\Users\boldi\Desktop\University\Semester 1\Capstone\antenna_analysis"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ============================================================
# REFERENCE DESIGN
# ============================================================

REFERENCE_POWER_W = 15.0
REFERENCE_DIAMETER_M = 2.5
REFERENCE_DISTANCE_AU = 30.0
REFERENCE_DATA_RATE_BPS = 1000.0

# ============================================================
# REQUIREMENTS
# ============================================================

MIN_DATA_RATE_BPS = 1000.0
REQUIREMENT_DISTANCE_AU = 30.0
MIN_ANTENNA_DIAMETER_M = 2.5

# ============================================================
# ANALYSIS RANGES
# ============================================================

# Start at the minimum required distance
DISTANCES_AU = np.linspace(
    REQUIREMENT_DISTANCE_AU,
    150,
    500
)

# Start at the minimum required antenna diameter
ANTENNA_DIAMETERS_M = np.linspace(
    MIN_ANTENNA_DIAMETER_M,
    5.0,
    500
)

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_data_rate(
    distance_au,
    antenna_diameter_m,
    transmit_power_w=REFERENCE_POWER_W
):
    """
    Calculate data rate relative to the reference design:

    15 W transmitter
    2.5 m antenna
    1 kbps at 30 AU
    """

    data_rate = (
        REFERENCE_DATA_RATE_BPS
        * (transmit_power_w / REFERENCE_POWER_W)
        * (antenna_diameter_m / REFERENCE_DIAMETER_M) ** 2
        * (REFERENCE_DISTANCE_AU / distance_au) ** 2
    )

    return data_rate


def calculate_required_power(
    distance_au,
    antenna_diameter_m,
    data_rate_bps
):
    """
    Calculate RF transmitter power required to achieve
    a desired data rate.
    """

    required_power = (
        REFERENCE_POWER_W
        * (data_rate_bps / REFERENCE_DATA_RATE_BPS)
        * (REFERENCE_DIAMETER_M / antenna_diameter_m) ** 2
        * (distance_au / REFERENCE_DISTANCE_AU) ** 2
    )

    return required_power


# ============================================================
# PLOT 1
# DATA RATE VS DISTANCE
# ============================================================

data_rate_vs_distance = calculate_data_rate(
    distance_au=DISTANCES_AU,
    antenna_diameter_m=REFERENCE_DIAMETER_M,
    transmit_power_w=REFERENCE_POWER_W
)

plt.figure(figsize=(10, 6))

plt.plot(
    DISTANCES_AU,
    data_rate_vs_distance,
    linewidth=2,
    label="2.5 m HGA, 15 W"
)

plt.axhline(
    MIN_DATA_RATE_BPS,
    linestyle="--",
    label="Minimum data rate = 1 kbps"
)

plt.scatter(
    [REFERENCE_DISTANCE_AU],
    [REFERENCE_DATA_RATE_BPS],
    s=80,
    zorder=5,
    label="Minimum requirement point"
)

plt.xlabel("Distance from Earth (AU)")
plt.ylabel("Data Rate (bits/s)")
plt.title("Downlink Data Rate vs. Distance")

plt.xlim(
    REQUIREMENT_DISTANCE_AU,
    DISTANCES_AU[-1]
)

plt.yscale("log")

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plot1_path = (
    OUTPUT_FOLDER
    / "data_rate_vs_distance.png"
)

plt.savefig(
    plot1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2
# DATA RATE VS ANTENNA DIAMETER
# ============================================================

data_rate_vs_diameter = calculate_data_rate(
    distance_au=REQUIREMENT_DISTANCE_AU,
    antenna_diameter_m=ANTENNA_DIAMETERS_M,
    transmit_power_w=REFERENCE_POWER_W
)

plt.figure(figsize=(10, 6))

plt.plot(
    ANTENNA_DIAMETERS_M,
    data_rate_vs_diameter,
    linewidth=2,
    label="15 W transmitter at 30 AU"
)

plt.axhline(
    MIN_DATA_RATE_BPS,
    linestyle="--",
    label="Minimum data rate = 1 kbps"
)

plt.scatter(
    [MIN_ANTENNA_DIAMETER_M],
    [MIN_DATA_RATE_BPS],
    s=80,
    zorder=5,
    label="Minimum requirement point"
)

plt.xlabel("HGA Diameter (m)")
plt.ylabel("Data Rate at 30 AU (bits/s)")
plt.title("Data Rate Sensitivity to HGA Diameter")

plt.xlim(
    MIN_ANTENNA_DIAMETER_M,
    ANTENNA_DIAMETERS_M[-1]
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plot2_path = (
    OUTPUT_FOLDER
    / "data_rate_vs_antenna_diameter.png"
)

plt.savefig(
    plot2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 3
# REQUIRED TRANSMITTER POWER VS ANTENNA DIAMETER
# ============================================================

required_power_vs_diameter = calculate_required_power(
    distance_au=REQUIREMENT_DISTANCE_AU,
    antenna_diameter_m=ANTENNA_DIAMETERS_M,
    data_rate_bps=MIN_DATA_RATE_BPS
)

plt.figure(figsize=(10, 6))

plt.plot(
    ANTENNA_DIAMETERS_M,
    required_power_vs_diameter,
    linewidth=2,
    label="Power required for 1 kbps at 30 AU"
)

plt.axhline(
    REFERENCE_POWER_W,
    linestyle="--",
    label="15 W reference transmitter"
)

plt.scatter(
    [MIN_ANTENNA_DIAMETER_M],
    [REFERENCE_POWER_W],
    s=80,
    zorder=5,
    label="Minimum requirement point"
)

plt.xlabel("HGA Diameter (m)")
plt.ylabel("Required RF Transmitter Power (W)")
plt.title("RF Power Sensitivity to HGA Diameter at 30 AU")

plt.xlim(
    MIN_ANTENNA_DIAMETER_M,
    ANTENNA_DIAMETERS_M[-1]
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plot3_path = (
    OUTPUT_FOLDER
    / "required_power_vs_antenna_diameter.png"
)

plt.savefig(
    plot3_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 4
# REQUIRED POWER VS DISTANCE FOR 1 KBPS
# ============================================================

required_power_vs_distance = calculate_required_power(
    distance_au=DISTANCES_AU,
    antenna_diameter_m=REFERENCE_DIAMETER_M,
    data_rate_bps=MIN_DATA_RATE_BPS
)

plt.figure(figsize=(10, 6))

plt.plot(
    DISTANCES_AU,
    required_power_vs_distance,
    linewidth=2,
    label="2.5 m HGA"
)

plt.axhline(
    REFERENCE_POWER_W,
    linestyle="--",
    label="15 W reference transmitter"
)

plt.scatter(
    [REFERENCE_DISTANCE_AU],
    [REFERENCE_POWER_W],
    s=80,
    zorder=5,
    label="Minimum requirement point"
)

plt.xlabel("Distance from Earth (AU)")
plt.ylabel("Required RF Transmitter Power (W)")
plt.title("RF Power Required to Maintain 1 kbps")

plt.xlim(
    REQUIREMENT_DISTANCE_AU,
    DISTANCES_AU[-1]
)

plt.yscale("log")

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plot4_path = (
    OUTPUT_FOLDER
    / "required_power_vs_distance.png"
)

plt.savefig(
    plot4_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("COMMUNICATION REQUIREMENT ANALYSIS")
print("==================================")

baseline_rate = calculate_data_rate(
    distance_au=REFERENCE_DISTANCE_AU,
    antenna_diameter_m=REFERENCE_DIAMETER_M,
    transmit_power_w=REFERENCE_POWER_W
)

print()
print("Reference design:")
print(f"HGA diameter: {REFERENCE_DIAMETER_M:.2f} m")
print(f"RF transmitter power: {REFERENCE_POWER_W:.2f} W")
print(f"Distance: {REFERENCE_DISTANCE_AU:.2f} AU")
print(f"Data rate: {baseline_rate:.2f} bits/s")

print()
print("Data rate vs. distance:")
print("-----------------------")

for distance in [30, 40, 50, 60, 80, 100, 120, 150]:

    rate = calculate_data_rate(
        distance_au=distance,
        antenna_diameter_m=REFERENCE_DIAMETER_M,
        transmit_power_w=REFERENCE_POWER_W
    )

    print(
        f"{distance:6.1f} AU | "
        f"{rate:10.2f} bits/s"
    )

print()
print("Antenna diameter sensitivity at 30 AU:")
print("--------------------------------------")

for diameter in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:

    rate = calculate_data_rate(
        distance_au=REQUIREMENT_DISTANCE_AU,
        antenna_diameter_m=diameter,
        transmit_power_w=REFERENCE_POWER_W
    )

    power = calculate_required_power(
        distance_au=REQUIREMENT_DISTANCE_AU,
        antenna_diameter_m=diameter,
        data_rate_bps=MIN_DATA_RATE_BPS
    )

    print(
        f"{diameter:.1f} m | "
        f"Data rate = {rate:8.1f} bits/s | "
        f"Power for 1 kbps = {power:6.2f} W"
    )

print()
print("Saved plots:")
print(plot1_path)
print(plot2_path)
print(plot3_path)
print(plot4_path)