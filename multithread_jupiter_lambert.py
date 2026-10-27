# ---------------------------------------------------------
# Earth -> Jupiter gravity assist -> 3I/ATLAS Lambert sweep
# ---------------------------------------------------------
#
# How it works (patched conics, three stages):
#
#   Stage 1  Lambert sweep Earth -> Jupiter for every
#            (departure date, Jupiter date) pair.
#   Stage 2  Lambert sweep Jupiter -> 3I/ATLAS for every
#            (Jupiter date, arrival date) pair.
#   Stage 3  For every Jupiter date, join each stage 1 leg
#            to each stage 2 leg with a flyby: the incoming
#            and outgoing v-infinity must be joined by a
#            hyperbola with periapsis above the safety
#            altitude. If their speeds differ, a burn at
#            periapsis makes up the difference (a pure
#            gravity assist is the case where that is ~0).
#
# Stages 1 and 2 are the slow part. They run on the process
# pool and append to resumable CSV files, exactly like the
# old direct sweep. Stage 3 is vectorised NumPy and takes
# seconds to minutes, so it is simply re-run every time.
#
# Units:
#   CelestialBody stores metres, m/s and m^3/s^2.
#   lambert.py and gravity_assist.py work in km, km/s and
#   km^3/s^2. Everything is converted ONCE, when the daily
#   ephemeris arrays are built in main() (M_TO_KM and
#   M3_TO_KM3 below). After that every number is km based.
# ---------------------------------------------------------

# 1004083 SPK ID for 3I/ATLAS

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
from multiprocessing import shared_memory
import csv
import os

import numpy as np
import pandas as pd

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, datetime_to_jd
from ProbeOrbit.Orbital_Mission_Planning_wAssist.lambert import lambert
from ProbeOrbit.Orbital_Mission_Planning_wAssist.gravity_assist import (powered_flyby, leo_burn, leo_departure_state, ecliptic_to_equatorial, right_ascension_declination)


# =========================================================
# Configuration
# =========================================================

LEG1_CSV_FILE = os.path.join("Jupiter Gravity Assist Data", "leg1_earth_jupiter.csv")
LEG2_CSV_FILE = os.path.join("Jupiter Gravity Assist Data", "leg2_jupiter_comet.csv")
RESULTS_CSV_FILE = os.path.join("Jupiter Gravity Assist Data", "jupiter_assist_results.csv")

# Ephemeris CSVs, one per body.
# Columns: epoch (JD), x, y, z [km], vx, vy, vz [km/s]
# Heliocentric, all in the same frame (see CSV_FRAME).
EARTH_CSV = os.path.join("Ephemeris", "EARTH.csv")
JUPITER_CSV = os.path.join("Ephemeris", "JUPITER.csv")
COMET_CSV = os.path.join("Ephemeris", "3IATLAS.csv")

# "ecliptic" (Horizons default, J2000 ecliptic) or "equatorial" (ICRF).
# Only used to report the departure asymptote RA/Dec and LEO burn point.
CSV_FRAME = "ecliptic"

# Number of CPU processes to use.
# Leaving 2 cores free keeps Windows responsive.
NUM_PROCESSES = max(1, cpu_count() - 2)

# Number of Lambert cases given to each worker at once.
CHUNK_SIZE = 2000

# Number of chunks allowed to be running simultaneously.
# Keeping this relatively small avoids excessive memory usage.
MAX_PENDING_CHUNKS = NUM_PROCESSES * 3

# Sun gravitational parameter
MU_SUN = 1.32712440018e11  # km^3/s^2

# Body gravitational parameters in CelestialBody units (m^3/s^2)
MU_EARTH_SI = 3.986004418e14
MU_JUPITER_SI = 1.26686534e17
MU_COMET_SI = 2200

# Unit conversions from CelestialBody (SI) to lambert.py (km)
M_TO_KM = 1.0e-3
M3_TO_KM3 = 1.0e-9

# Jupiter
R_JUPITER = 71492.0  # km, equatorial radius (1 bar)
MIN_FLYBY_ALTITUDE = 5000.0  # km above the cloud tops

# Launch
LEO_ALTITUDE = 400.0  # km
PARKING_INCLINATION_DEG = 28.5  # only used for the example burn point of the best case

# Mission constraints
MAX_ARRIVAL_SPEED = 1.5 # km/s relative to the comet
MAX_COMET_DISTANCE_AU = 30.0  # rendezvous at or before this heliocentric distance
MAX_FLYBY_DELTA_V = 0.1 #0.1  # km/s, largest periapsis burn kept in the results (0 = pure gravity assist)

# Leg time of flight limits (days)
MIN_LEG_DAYS = 30
MAX_LEG1_DAYS = 1000

AU = 149597870.7  # km

# =========================================================
# Mission date ranges
#
# Step sizes let you do a quick coarse run first
# (e.g. 5 days) and then a fine run around the best region.
# =========================================================

start_date_leaving = datetime(2025, 6, 1)
end_date_leaving = datetime(2026, 6, 1)
step_days_leaving = 1

start_date_jupiter = datetime(2025, 8, 1)
end_date_jupiter = datetime(2027, 12, 31)
step_days_jupiter = 1

start_date_arriving = datetime(2025, 9, 1)
end_date_arriving = datetime(2028, 12, 31)
step_days_arriving = 1

# Window loaded from the ephemeris CSVs. Must cover every date above.
ephemeris_start = min(start_date_leaving, start_date_jupiter, start_date_arriving)
ephemeris_end = max(end_date_leaving, end_date_jupiter, end_date_arriving)


# =========================================================
# Generate dates
# =========================================================

def date_list(start, end, step_days):
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0, (end - start).days + 1, step_days)]


leaving_date_list = date_list(start_date_leaving, end_date_leaving, step_days_leaving)

jupiter_date_list = date_list(start_date_jupiter, end_date_jupiter, step_days_jupiter)

arriving_date_list = date_list(start_date_arriving, end_date_arriving, step_days_arriving)


# =========================================================
# Precompute time values
#
# Times are seconds since ephemeris_start, which is t = 0 for
# every CelestialBody, so they can go straight into
# CelestialBody.get_state(t).
# =========================================================

reference_date = ephemeris_start


def seconds_since_reference(dates):
    return np.array([(datetime.strptime(date, "%Y-%m-%d") - reference_date).total_seconds() for date in dates], dtype=np.float64)


leaving_times = seconds_since_reference(leaving_date_list)

jupiter_times = seconds_since_reference(jupiter_date_list)

arriving_times = seconds_since_reference(arriving_date_list)


# =========================================================
# Legs
#
# Each leg is a Lambert arc from one body to another.
# The names refer to the shared-memory arrays below.
# =========================================================

LEGS = {
    "leg1": ("earth", "jupiter"),
    "leg2": ("jupiter", "comet"),
}

LEG_CSV_HEADER = [
    "start_date",
    "end_date",
    "time_of_flight_days",

    # v-infinity leaving the start body (leg1: Earth departure, leg2: Jupiter outgoing)
    "vinf_start",
    "vinf_start_x",
    "vinf_start_y",
    "vinf_start_z",

    # v-infinity at the end body (leg1: Jupiter incoming, leg2: velocity relative to comet)
    "vinf_end",
    "vinf_end_x",
    "vinf_end_y",
    "vinf_end_z",
]


# =========================================================
# Worker globals
#
# Each worker attaches to the shared-memory arrays during
# initialization.
# =========================================================

shared_arrays = {}
shared_memory_blocks = {}
worker_times = {}


# =========================================================
# Worker initialization
# =========================================================

def initialize_worker(shared_array_info, times_data):
    """
    Attach each worker process to the shared ephemeris arrays.

    shared_array_info: {"earth_r": (shm_name, shape), ...}
    times_data:        {"earth": leaving_times, "jupiter": jupiter_times, "comet": arriving_times}

    The arrays are treated as read-only by the workers.
    """

    global shared_arrays
    global shared_memory_blocks
    global worker_times

    for label, (shm_name, shape) in shared_array_info.items():

        # Attach to shared memory
        shared_memory_blocks[label] = shared_memory.SharedMemory(name=shm_name)

        # Create NumPy view onto shared memory
        shared_arrays[label] = np.ndarray(shape, dtype=np.float64, buffer=shared_memory_blocks[label].buf)

    # Time arrays are tiny, so copying these into each worker
    # is not an issue.
    worker_times = times_data


# =========================================================
# Solve one chunk of Lambert cases
# =========================================================

def solve_chunk(leg_name, jobs):
    """
    Solve a chunk of Lambert cases for one leg.

    jobs contains tuples of:

        (start_index, end_index)

    The worker accesses the ephemeris through shared memory.
    Failed cases are returned with NaN values so they are
    still written to the CSV and not retried on resume.
    """
    start_body, end_body = LEGS[leg_name]

    r_start_all = shared_arrays[start_body + "_r"]
    v_start_all = shared_arrays[start_body + "_v"]
    r_end_all = shared_arrays[end_body + "_r"]
    v_end_all = shared_arrays[end_body + "_v"]

    start_times = worker_times[start_body]
    end_times = worker_times[end_body]

    results = []

    for start_index, end_index in jobs:

        # Retrieve states from shared memory
        r_start = r_start_all[start_index]
        v_start = v_start_all[start_index]

        r_end = r_end_all[end_index]
        v_end = v_end_all[end_index]

        # Time of flight
        dt = (end_times[end_index] - start_times[start_index])

        # -------------------------------------------------
        # Lambert
        # -------------------------------------------------
        try:

            v_departure, v_arrival = lambert(
                r_start,
                r_end,
                float(dt),
                MU_SUN,
            )

        except (ValueError, RuntimeError, FloatingPointError, ZeroDivisionError):
            # Invalid/no Lambert solution
            results.append((start_index, end_index, dt / 86400.0) + (np.nan,) * 8)
            continue

        # -------------------------------------------------
        # v-infinity at both ends
        # -------------------------------------------------

        vinf_start_vector = (v_departure - v_start)
        vinf_start = np.linalg.norm(vinf_start_vector)

        vinf_end_vector = (v_arrival - v_end)
        vinf_end = np.linalg.norm(vinf_end_vector)

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append((
            start_index,
            end_index,
            dt / 86400.0,

            vinf_start,
            vinf_start_vector[0],
            vinf_start_vector[1],
            vinf_start_vector[2],

            vinf_end,
            vinf_end_vector[0],
            vinf_end_vector[1],
            vinf_end_vector[2],
        ))

    return results


# =========================================================
# Create shared memory arrays
# =========================================================

def create_shared_array(array):

    """
    Creates a shared-memory copy of a NumPy array.

    Returns:
        shared_memory_object, numpy_array_view
    """

    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)

    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)

    shared_array[:] = array[:]

    return shm, shared_array


# =========================================================
# Ephemeris: CelestialBody -> daily km arrays
# =========================================================

def sample_body(body, times):
    """
    Sample a CelestialBody at the given times (seconds since
    ephemeris_start) and convert m, m/s -> km, km/s.
    """

    r = np.array([body.get_state(t) for t in times], dtype=np.float64) * M_TO_KM

    v = np.array([body.get_velocity(t) for t in times], dtype=np.float64) * M_TO_KM

    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise ValueError(f"{body.name}: ephemeris returned NaN, check the CSV covers every sweep date.")

    return r, v


# =========================================================
# Run one Lambert leg (resumable)
# =========================================================

def count_cases(start_times, end_times, min_days, max_days):
    tof_days = (end_times[None, :] - start_times[:, None]) / 86400.0
    return int(np.count_nonzero((tof_days >= min_days) & (tof_days <= max_days)))


def write_chunk_results(writer, chunk_results, start_dates, end_dates):
    for result in chunk_results:
        start_index, end_index = result[0], result[1]
        writer.writerow([start_dates[start_index], end_dates[end_index], *result[2:]])
    return len(chunk_results)


def run_leg(executor, leg_name, csv_file, start_dates, end_dates, start_times, end_times, min_days, max_days):

    start_body, end_body = LEGS[leg_name]

    print("=" * 60)
    print(f"{leg_name}: {start_body} -> {end_body}")
    print("=" * 60)

    total_cases = count_cases(start_times, end_times, min_days, max_days)
    print(f"Cases with {min_days} <= time of flight <= {max_days} days: {total_cases:,}")

    # =====================================================
    # Load previously completed cases
    # =====================================================

    completed_cases = set()

    file_exists = os.path.exists(csv_file)

    if file_exists:
        print(f"Loading completed cases from {csv_file}...")

        with open(csv_file, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:
                completed_cases.add(
                    (row["start_date"], row["end_date"])
                )

        print(f"Previously completed: {len(completed_cases):,}")

    else:
        print("No existing CSV found. Starting from zero.\n")

    # =====================================================
    # Open CSV
    # =====================================================

    csv_mode = "a" if file_exists else "w"

    with open(csv_file, csv_mode, newline="") as file:

        writer = csv.writer(file)

        # Write header for a new file
        if not file_exists:
            writer.writerow(LEG_CSV_HEADER)
            file.flush()

        # =================================================
        # Generate chunks without creating one giant
        # list of all jobs.
        # =================================================

        def generate_chunks():

            current_chunk = []

            for i, start_date in enumerate(start_dates):
                for j, end_date in enumerate(end_dates):

                    tof_days = (end_times[j] - start_times[i]) / 86400.0

                    # Skip already-completed cases
                    if (start_date, end_date) in completed_cases:
                        continue
                    elif tof_days < min_days or tof_days > max_days:
                        continue

                    current_chunk.append((i, j))

                    if len(current_chunk) >= CHUNK_SIZE:
                        yield current_chunk
                        current_chunk = []

            if current_chunk:
                yield current_chunk

        # =================================================
        # Submit chunks while limiting the number
        # simultaneously in memory.
        # =================================================

        pending = set()

        cases_completed = len(completed_cases)

        print("Starting Lambert calculations...\n")

        def collect(done):
            nonlocal cases_completed

            for completed_future in done:
                cases_completed += write_chunk_results(writer, completed_future.result(), start_dates, end_dates)

                # Flush after each completed chunk
                file.flush()

                percentage = (cases_completed / max(total_cases, 1) * 100)

                print(f"{leg_name} completed: {cases_completed:,}/{total_cases:,} ({percentage:.2f}%)")

        for chunk in generate_chunks():

            future = executor.submit(solve_chunk, leg_name, chunk)
            pending.add(future)

            # Keep only a limited number of chunks
            # waiting/running at once.
            if len(pending) >= MAX_PENDING_CHUNKS:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)

        # =================================================
        # Wait for remaining chunks
        # =================================================

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            collect(done)

    print(f"{leg_name} saved to: {csv_file}\n")


# =========================================================
# Stage 3: join the legs with a Jupiter flyby
# =========================================================

def match_flybys(mu_jupiter, jupiter_r_by_date, jupiter_v_by_date, comet_distance_by_date):
    """
    mu_jupiter:             km^3/s^2
    jupiter_r_by_date:      {"YYYY-MM-DD": Jupiter heliocentric position [km]}
    jupiter_v_by_date:      {"YYYY-MM-DD": Jupiter heliocentric velocity [km/s]}
    comet_distance_by_date: {"YYYY-MM-DD": comet heliocentric distance [AU]}
    """

    print("=" * 60)
    print("Stage 3: Jupiter flyby matching")
    print("=" * 60)

    leg1 = pd.read_csv(LEG1_CSV_FILE).dropna()
    leg2 = pd.read_csv(LEG2_CSV_FILE).dropna()

    # Only arrivals that already meet the comet constraints
    leg2 = leg2[(leg2["vinf_end"] <= MAX_ARRIVAL_SPEED) & leg2["end_date"].isin(comet_distance_by_date.keys())]
    print(f"Jupiter -> comet legs with arrival speed <= {MAX_ARRIVAL_SPEED} km/s: {len(leg2):,}")

    leg1_by_jupiter_date = dict(tuple(leg1.groupby("end_date")))

    rp_min = R_JUPITER + MIN_FLYBY_ALTITUDE

    vin_columns = ["vinf_end_x", "vinf_end_y", "vinf_end_z"]
    vout_columns = ["vinf_start_x", "vinf_start_y", "vinf_start_z"]

    result_frames = []

    for jupiter_date, leg2_group in leg2.groupby("start_date"):

        if jupiter_date not in leg1_by_jupiter_date:
            continue

        leg1_group = leg1_by_jupiter_date[jupiter_date]

        # (n1, 1, 3) and (1, n2, 3) so every pair is tested at once
        v_inf_in = leg1_group[vin_columns].to_numpy()[:, None, :]
        v_inf_out = leg2_group[vout_columns].to_numpy()[None, :, :]

        v_inf_in, v_inf_out = np.broadcast_arrays(v_inf_in, v_inf_out)
        leg1_index, leg2_index = np.meshgrid(np.arange(len(leg1_group)), np.arange(len(leg2_group)), indexing="ij")

        # -------------------------------------------------
        # Quick rejection: the periapsis burn is smallest at
        # the lowest allowed periapsis (Oberth effect), so if
        # even that is too big the pair can never work.
        # -------------------------------------------------

        v_in = np.linalg.norm(v_inf_in, axis=-1)
        v_out = np.linalg.norm(v_inf_out, axis=-1)
        smallest_burn = np.abs(np.sqrt(v_out**2 + 2.0 * mu_jupiter / rp_min) - np.sqrt(v_in**2 + 2.0 * mu_jupiter / rp_min))

        keep = smallest_burn <= MAX_FLYBY_DELTA_V

        if not np.any(keep):
            continue

        # Sphere of influence radius at this date
        r_jupiter = np.linalg.norm(jupiter_r_by_date[jupiter_date])
        rp_max = r_jupiter * (mu_jupiter / MU_SUN) ** 0.4

        rp, flyby_dv, delta, feasible = powered_flyby(v_inf_in[keep], v_inf_out[keep], mu_jupiter, rp_min, rp_max)

        good = feasible & (flyby_dv <= MAX_FLYBY_DELTA_V)

        if not np.any(good):
            continue

        rows_1 = leg1_group.iloc[leg1_index[keep][good]].reset_index(drop=True)
        rows_2 = leg2_group.iloc[leg2_index[keep][good]].reset_index(drop=True)

        # -------------------------------------------------
        # Velocities at Jupiter for the N-body simulation
        #
        # v-infinity is relative to Jupiter. Adding Jupiter's
        # velocity gives the heliocentric velocity arriving at
        # and leaving Jupiter (the patched-conic values that
        # main_wirth_orbit.py aims the flyby with).
        # -------------------------------------------------

        v_jupiter = jupiter_v_by_date[jupiter_date]

        vinf_in_vectors = rows_1[vin_columns].to_numpy()
        vinf_out_vectors = rows_2[vout_columns].to_numpy()

        v_into_jupiter = vinf_in_vectors + v_jupiter
        v_out_of_jupiter = vinf_out_vectors + v_jupiter

        frame = pd.DataFrame({
            "departure_date": rows_1["start_date"],
            "jupiter_date": jupiter_date,
            "arrival_date": rows_2["end_date"],

            "leg1_days": rows_1["time_of_flight_days"],
            "leg2_days": rows_2["time_of_flight_days"],
            "total_days": rows_1["time_of_flight_days"] + rows_2["time_of_flight_days"],

            # Earth departure v-infinity (CSV frame)
            "vinf_departure": rows_1["vinf_start"],
            "vinf_departure_x": rows_1["vinf_start_x"],
            "vinf_departure_y": rows_1["vinf_start_y"],
            "vinf_departure_z": rows_1["vinf_start_z"],

            # Jupiter flyby
            "vinf_jupiter_in": rows_1["vinf_end"],
            "vinf_jupiter_out": rows_2["vinf_start"],
            "turn_angle_deg": np.degrees(delta[good]),
            "periapsis_radius_km": rp[good],
            "periapsis_altitude_km": rp[good] - R_JUPITER,
            "flyby_dv": flyby_dv[good],

            # Jupiter-relative v-infinity vectors (CSV frame, km/s)
            "vinf_jupiter_in_x": vinf_in_vectors[:, 0],
            "vinf_jupiter_in_y": vinf_in_vectors[:, 1],
            "vinf_jupiter_in_z": vinf_in_vectors[:, 2],
            "vinf_jupiter_out_x": vinf_out_vectors[:, 0],
            "vinf_jupiter_out_y": vinf_out_vectors[:, 1],
            "vinf_jupiter_out_z": vinf_out_vectors[:, 2],

            # Jupiter's heliocentric velocity at 00:00 on the flyby date (km/s)
            "jupiter_vx": v_jupiter[0],
            "jupiter_vy": v_jupiter[1],
            "jupiter_vz": v_jupiter[2],

            # Heliocentric velocity arriving at / leaving Jupiter (km/s)
            "v_into_jupiter_x": v_into_jupiter[:, 0],
            "v_into_jupiter_y": v_into_jupiter[:, 1],
            "v_into_jupiter_z": v_into_jupiter[:, 2],
            "v_out_of_jupiter_x": v_out_of_jupiter[:, 0],
            "v_out_of_jupiter_y": v_out_of_jupiter[:, 1],
            "v_out_of_jupiter_z": v_out_of_jupiter[:, 2],

            # Comet arrival
            "arrival_speed": rows_2["vinf_end"],
            "vrel_x": rows_2["vinf_end_x"],
            "vrel_y": rows_2["vinf_end_y"],
            "vrel_z": rows_2["vinf_end_z"],
            "comet_distance_au": rows_2["end_date"].map(comet_distance_by_date),
        })

        result_frames.append(frame)

    if not result_frames:
        print("No Earth -> Jupiter -> comet combinations met the constraints.")
        print("Try a larger MAX_FLYBY_DELTA_V or MAX_ARRIVAL_SPEED, or wider date ranges.")
        return None

    results = pd.concat(result_frames, ignore_index=True)

    # -------------------------------------------------
    # Launch from LEO
    # -------------------------------------------------

    v_inf_departure = results[["vinf_departure_x", "vinf_departure_y", "vinf_departure_z"]].to_numpy()

    if CSV_FRAME == "ecliptic":
        v_inf_departure_equatorial = ecliptic_to_equatorial(v_inf_departure)
    else:
        v_inf_departure_equatorial = v_inf_departure

    ra, dec = right_ascension_declination(v_inf_departure_equatorial)
    leo_dv, _, _ = leo_burn(results["vinf_departure"].to_numpy(), LEO_ALTITUDE)

    results.insert(6, "leo_burn_dv", leo_dv)
    results.insert(7, "c3", results["vinf_departure"] ** 2)
    results["vinf_departure_eq_x"] = v_inf_departure_equatorial[:, 0]
    results["vinf_departure_eq_y"] = v_inf_departure_equatorial[:, 1]
    results["vinf_departure_eq_z"] = v_inf_departure_equatorial[:, 2]
    results["asymptote_ra_deg"] = ra
    results["asymptote_dec_deg"] = dec

    # On-board propellant budget: flyby burn + arrival matching burn
    results["onboard_dv"] = results["flyby_dv"] + results["arrival_speed"]

    results = results.sort_values(["onboard_dv", "leo_burn_dv"]).reset_index(drop=True)

    results.to_csv(RESULTS_CSV_FILE, index=False)

    print(f"Combinations that meet the constraints: {len(results):,}")
    print(f"Results saved to: {RESULTS_CSV_FILE}\n")

    return results


def print_summary(results):

    columns = ["departure_date", "jupiter_date", "arrival_date", "leo_burn_dv", "periapsis_altitude_km",
               "flyby_dv", "arrival_speed", "onboard_dv", "comet_distance_au"]

    print("Best 10 by on-board delta-v (flyby burn + arrival speed):")
    print(results[columns].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    pure = results[results["flyby_dv"] <= 0.05]
    if len(pure):
        print("\nBest 10 near-pure gravity assists (flyby burn <= 0.05 km/s), by arrival speed:")
        print(pure.sort_values("arrival_speed")[columns].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = results.iloc[0]

    v_inf_equatorial = best[["vinf_departure_eq_x", "vinf_departure_eq_y", "vinf_departure_eq_z"]].to_numpy(dtype=float)
    position, velocity_before, velocity_after, burn, inclination = leo_departure_state(v_inf_equatorial, LEO_ALTITUDE, PARKING_INCLINATION_DEG)

    print("\nBest case departure from LEO (geocentric J2000 equatorial):")
    print(f"  launch {best['departure_date']}, v-infinity {best['vinf_departure']:.3f} km/s, "
          f"RA {best['asymptote_ra_deg']:.2f} deg, Dec {best['asymptote_dec_deg']:.2f} deg")
    print(f"  example burn point for a {inclination:.1f} deg parking orbit at {LEO_ALTITUDE:.0f} km:")
    print(f"    position         {np.array2string(position, precision=1)} km")
    print(f"    velocity before  {np.array2string(velocity_before, precision=3)} km/s")
    print(f"    velocity after   {np.array2string(velocity_after, precision=3)} km/s")
    print(f"    burn vector      {np.array2string(burn, precision=3)} km/s  (|dv| = {np.linalg.norm(burn):.3f} km/s)")


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Earth -> Jupiter -> 3I/ATLAS Lambert trajectory sweep")
    print("=" * 60)

    print(f"Departure dates: {len(leaving_date_list):,}")

    print(f"Jupiter dates:   {len(jupiter_date_list):,}")

    print(f"Arrival dates:   {len(arriving_date_list):,}")

    print(f"Using {NUM_PROCESSES} worker processes")

    print(f"Chunk size: {CHUNK_SIZE:,}\n")

    # =====================================================
    # Load ephemeris from CSV
    # =====================================================

    print("Loading Earth ephemeris...")
    earth = CelestialBody("Earth", MU_EARTH_SI, ephemeris_start, ephemeris_end, csv_file=EARTH_CSV)

    print("Loading Jupiter ephemeris...")
    jupiter = CelestialBody("Jupiter", MU_JUPITER_SI, ephemeris_start, ephemeris_end, csv_file=JUPITER_CSV)

    print("Loading 3I/ATLAS ephemeris...")
    comet = CelestialBody("3I/ATLAS", MU_COMET_SI, ephemeris_start, ephemeris_end, csv_file=COMET_CSV)

    print("Ephemeris loaded.\n")

    # =====================================================
    # Sample at the sweep dates and convert to km
    # =====================================================

    earth_r_array, earth_v_array = sample_body(earth, leaving_times)

    jupiter_r_array, jupiter_v_array = sample_body(jupiter, jupiter_times)

    comet_r_array, comet_v_array = sample_body(comet, arriving_times)

    mu_jupiter = jupiter.mu * M3_TO_KM3

    # =====================================================
    # Only arrive while the comet is inside 30 AU
    # =====================================================

    comet_distance_au = np.linalg.norm(comet_r_array, axis=1) / AU
    inside_limit = comet_distance_au <= MAX_COMET_DISTANCE_AU

    if not np.any(inside_limit):
        raise ValueError(f"The comet is beyond {MAX_COMET_DISTANCE_AU} AU on every arrival date.")

    if not np.all(inside_limit):
        print(f"Comet passes {MAX_COMET_DISTANCE_AU} AU after {arriving_date_list[np.flatnonzero(inside_limit)[-1]]}; "
              f"later arrival dates are dropped.")

    arrival_dates = [date for date, ok in zip(arriving_date_list, inside_limit) if ok]
    arrival_times = arriving_times[inside_limit]
    comet_r_array = comet_r_array[inside_limit]
    comet_v_array = comet_v_array[inside_limit]

    comet_distance_by_date = dict(zip(arrival_dates, comet_distance_au[inside_limit]))
    jupiter_r_by_date = dict(zip(jupiter_date_list, jupiter_r_array))
    jupiter_v_by_date = dict(zip(jupiter_date_list, jupiter_v_array))

    print(f"Arrival dates inside {MAX_COMET_DISTANCE_AU} AU: {len(arrival_dates):,}\n")

    # =====================================================
    # Create shared memory
    # =====================================================

    arrays = {
        "earth_r": earth_r_array,
        "earth_v": earth_v_array,
        "jupiter_r": jupiter_r_array,
        "jupiter_v": jupiter_v_array,
        "comet_r": comet_r_array,
        "comet_v": comet_v_array,
    }

    times = {
        "earth": leaving_times,
        "jupiter": jupiter_times,
        "comet": arrival_times,
    }

    shared_blocks = {}

    try:
        print("Creating shared ephemeris memory...")

        for label, array in arrays.items():
            shared_blocks[label], _ = create_shared_array(array)

        shared_array_info = {label: (shared_blocks[label].name, arrays[label].shape) for label in arrays}

        print("Shared memory ready.\n")

        # =================================================
        # Create process pool
        # =================================================

        with ProcessPoolExecutor(
            max_workers=NUM_PROCESSES,
            initializer=initialize_worker,
            initargs=(shared_array_info, times)
        ) as executor:

            # Stage 1: Earth -> Jupiter
            run_leg(executor, "leg1", LEG1_CSV_FILE,
                    leaving_date_list, jupiter_date_list,
                    leaving_times, jupiter_times,
                    MIN_LEG_DAYS, MAX_LEG1_DAYS)

            # Stage 2: Jupiter -> comet
            run_leg(executor, "leg2", LEG2_CSV_FILE,
                    jupiter_date_list, arrival_dates,
                    jupiter_times, arrival_times,
                    MIN_LEG_DAYS, np.inf)

    finally:

        # =====================================================
        # Clean up shared memory
        # =====================================================

        for shm in shared_blocks.values():
            shm.close()
            shm.unlink()

    # Stage 3: flyby matching
    results = match_flybys(mu_jupiter, jupiter_r_by_date, jupiter_v_by_date, comet_distance_by_date)

    if results is not None:
        print_summary(results)

    print("=" * 60)
    print("Calculation complete.")
    print("=" * 60)


# =========================================================
# Windows multiprocessing entry point
# =========================================================

if __name__ == "__main__":
    # dt = datetime(2025, 11, 16)
    # print(datetime_to_jd(dt))
    # dt = datetime(2027, 11, 19)
    # print(datetime_to_jd(dt))
    # exit()
    main()
