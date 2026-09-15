# 1004083 SPK ID for 3I/ATLAS

import csv
import os
import numpy as np
from datetime import datetime, timedelta

from ephemeris import Ephemeris
from lambert import lambert


# ---------------------------------------------------------
# Mission date ranges
# ---------------------------------------------------------

start_date_leaving = datetime(2025, 7, 1)
end_date_leaving = datetime(2026, 6, 30)

start_date_arriving = datetime(2026, 7, 1)
end_date_arriving = datetime(2028, 12, 31)


# Generates dates inclusively from start_date to end_date
leaving_date_list = [(start_date_leaving + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date_leaving - start_date_leaving).days + 1)]

arriving_date_list = [(start_date_arriving + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date_arriving - start_date_arriving).days + 1)]

# ---------------------------------------------------------
# Get ephemeris data
# ---------------------------------------------------------

orbits = Ephemeris()

earth_data = orbits.get_state("'399'", "'2025-07-01'", "'2026-06-30'")

comet_data = orbits.get_state("'DES=1004083'", "'2026-07-01'", "'2028-12-31'")

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

MU_SUN = 1.32712440018e11  # km^3/s^2


CSV_FILE = "lambert_results.csv"


# ---------------------------------------------------------
# Set up CSV file
# ---------------------------------------------------------

file_exists = os.path.exists(CSV_FILE)

if not file_exists:

    with open(CSV_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "departure_date",
            "arrival_date",
            "time_of_flight_days",
            "delta_v (km-s)",
            "v_relative_comet (km-s)"
        ])


# ---------------------------------------------------------
# Load already-completed cases
# ---------------------------------------------------------

completed_cases = set()

if file_exists:

    with open(CSV_FILE, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            completed_cases.add(
                (
                    row["departure_date"],
                    row["arrival_date"]
                )
            )


print(f"Previously completed cases: {len(completed_cases)}")


# ---------------------------------------------------------
# Run Lambert calculations
# ---------------------------------------------------------

total_cases = (
    len(leaving_date_list)
    * len(arriving_date_list)
)

completed = len(completed_cases)


with open(CSV_FILE, "a", newline="") as file:

    writer = csv.writer(file)

    for i, leaving_date in enumerate(leaving_date_list):

        earth_state = earth_data[i]

        r_earth = np.array(earth_state["r"])
        v_earth = np.array(earth_state["v"])


        for j, arriving_date in enumerate(arriving_date_list):

            # Skip cases that have already been calculated
            if (leaving_date, arriving_date) in completed_cases:
                continue

            comet_state = comet_data[j]

            r_comet = np.array(comet_state["r"])
            v_comet = np.array(comet_state["v"])


            # Time of flight
            departure = datetime.strptime(
                leaving_date,
                "%Y-%m-%d"
            )

            arrival = datetime.strptime(
                arriving_date,
                "%Y-%m-%d"
            )

            dt = (arrival - departure).total_seconds()


            # Lambert solution
            try:

                v_departure, v_arrival = lambert(
                    r_earth,
                    r_comet,
                    dt,
                    MU_SUN
                )

            except ValueError:

                # No valid Lambert solution
                continue


            # Departure delta-V
            delta_v = np.linalg.norm(
                v_departure - v_earth
            )


            # Relative velocity at comet
            v_relative_comet = np.linalg.norm(
                v_arrival - v_comet
            )


            # Write result immediately
            writer.writerow([
                leaving_date,
                arriving_date,
                dt / 86400.0,
                delta_v,
                v_relative_comet
            ])

            # Make sure it actually gets written to disk
            file.flush()


            completed += 1


            # Progress
            if completed % 100 == 0:

                percentage = (
                    completed / total_cases * 100
                )

                print(
                    f"Completed: {completed:,}/{total_cases:,} "
                    f"({percentage:.2f}%)"
                )