from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(settings, milestones, output_folder, show_plot=True, log_scale=False):
    """Calculate and plot the average solar-array power generated each year."""

    years = settings["number_of_years"]
    samples = settings["samples_per_year"]

    milestone_times = np.array([row[0] for row in milestones], dtype=float)
    milestone_distances = np.array([row[1] for row in milestones], dtype=float)

    if milestone_times[0] != 0:
        raise ValueError("Milestones must start at year 0.")

    if milestone_times[-1] < years:
        raise ValueError("Milestones must cover the entire simulation.")

    # Time array
    time = np.linspace(0, years, years * samples + 1)

    # Distance from the Sun
    distance = np.interp(
        time,
        milestone_times,
        milestone_distances
    )

    # Solar incidence angle
    angle = settings["pointing_angle_deg"]

    if angle < 90:
        incidence = np.cos(np.deg2rad(angle))
    else:
        incidence = 0.0

    # ============================================================
    # SOLAR POWER AT 1 AU
    # ============================================================

    reference_power = (
        settings["solar_flux_1au_w_m2"]
        * settings["array_area_m2"]
        * settings["cell_efficiency"]
        * settings["packing_factor"]
        * settings["temperature_factor"]
        * settings["electrical_efficiency"]
        * settings["sunlight_fraction"]
        * settings["low_light_factor"]
        * incidence
    )

    # Solar-cell degradation
    retained_fraction = (
        1.0 - settings["annual_degradation_fraction"]
    ) ** time

    # Instantaneous solar-array power
    power_w = (
        reference_power
        * retained_fraction
        / distance**2
    )

    # ============================================================
    # AVERAGE POWER PER YEAR
    # ============================================================

    year = np.arange(1, years + 1)
    average_power_w = np.zeros(years)

    for i in range(years):
        mask = (time >= i) & (time <= i + 1)

        yearly_time = time[mask]
        yearly_power = power_w[mask]

        # Average power over the year
        average_power_w[i] = np.trapezoid(
            yearly_power,
            yearly_time
        )

    # ============================================================
    # PLOT
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        year,
        average_power_w,
        marker="o",
        markersize=3,
        linewidth=2
    )

    plt.xlabel("Elapsed Time (years)")
    plt.ylabel("Average Solar Power (W)")

    plt.title(
        f"Average Annual Solar-Array Power "
        f"({settings['array_area_m2']:.1f} m² Array)"
    )

    plt.xlim(1, years)

    if log_scale and np.all(average_power_w > 0):
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ============================================================
    # SAVE FIGURE
    # ============================================================

    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    image_path = output_folder / "average_annual_solar_power.png"

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved figure: {image_path}")

    print()
    print("AVERAGE ANNUAL SOLAR POWER")
    print("==========================")
    print(f"Solar-array area: {settings['array_area_m2']:.2f} m²")
    print(f"Power at 1 AU: {reference_power:.2f} W")
    print()

    for i in range(years):
        print(
            f"Year {year[i]:3d}: "
            f"{average_power_w[i]:.6f} W"
        )

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "year": year,
        "average_power_w": average_power_w,
        "image_path": image_path,
    }


if __name__ == "__main__":

    SETTINGS = {
        "number_of_years": 100,

        # Solar-array area
        "array_area_m2": 58.0,

        "cell_efficiency": 0.30,
        "packing_factor": 0.90,
        "temperature_factor": 0.88,
        "electrical_efficiency": 0.85,

        "pointing_angle_deg": 0.0,
        "sunlight_fraction": 1.0,
        "low_light_factor": 1.0,

        "annual_degradation_fraction": 0.0,

        "solar_flux_1au_w_m2": 1361.0,

        "samples_per_year": 1000,
    }

    OUTPUT_FOLDER = Path(
        r"C:\Users\boldi\Desktop\University\Semester 1\Capstone\solar_array"
    )

    SHOW_PLOT = True

    # I recommend True because power becomes extremely small
    # at large distances from the Sun.
    USE_LOG_SCALE = False

    MILESTONES = [
        (0, 1.0, "Launch"),
        (2, 1.33, "Rendezvous"),
        (4, 1.66, "Mars crossing"),
        (10, 5.1, "Jupiter crossing"),
        (25, 40, "Kuiper Belt entry"),
        (40, 80, "Pluto crossing / solar-system exit"),
        (50, 120, "Interstellar cruise"),
        (80, 1000, "Oort Cloud stragglers"),
        (160, 2000, "Inner Oort Cloud"),
        (8137, 100000, "Deep Oort Cloud"),
    ]

    results = run_simulation(
        settings=SETTINGS,
        milestones=MILESTONES,
        output_folder=OUTPUT_FOLDER,
        show_plot=SHOW_PLOT,
        log_scale=USE_LOG_SCALE,
    )