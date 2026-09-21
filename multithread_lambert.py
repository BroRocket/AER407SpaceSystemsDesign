# ---------------------------------------------------------
# Earth -> 3I/ATLAS Lambert trajectory sweep
# ---------------------------------------------------------

# 1004083 SPK ID for 3I/ATLAS

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
from multiprocessing import shared_memory
import csv
import os

import numpy as np

from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbital_Mission_Planning.lambert import lambert


# =========================================================
# Configuration
# =========================================================

CSV_FILE = "lambert_results_detailed.csv"

# Number of CPU processes to use.
# Leaving 2 cores free keeps Windows responsive.
NUM_PROCESSES = cpu_count() - 2

# Number of Lambert cases given to each worker at once.
CHUNK_SIZE = 2000

# Number of chunks allowed to be running simultaneously.
# Keeping this relatively small avoids excessive memory usage.
MAX_PENDING_CHUNKS = NUM_PROCESSES * 2

# Sun gravitational parameter
MU_SUN = 1.32712440018e11  # km^3/s^2

# =========================================================
# Mission date ranges
# =========================================================

start_date_leaving = datetime(2025, 1, 1)
end_date_leaving = datetime(2026, 6, 30)

start_date_arriving = datetime(2025, 3, 1)
end_date_arriving = datetime(2030, 12, 31)


# =========================================================
# Generate dates
# =========================================================

leaving_date_list = [(start_date_leaving + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date_leaving - start_date_leaving).days + 1)]

arriving_date_list = [(start_date_arriving + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date_arriving - start_date_arriving).days + 1)]


# =========================================================
# Precompute time values
#
# Since the dates are exactly one day apart, we can represent
# them as integer seconds instead of repeatedly converting
# datetime objects inside the Lambert loop.
# =========================================================

reference_date = start_date_leaving

leaving_times = np.array([(start_date_leaving + timedelta(days=i) - reference_date).total_seconds() for i in range(len(leaving_date_list))],dtype=np.float64)

arriving_times = np.array([(start_date_arriving + timedelta(days=i) - reference_date).total_seconds() for i in range(len(arriving_date_list))],dtype=np.float64)

# =========================================================
# Worker globals
#
# Each worker attaches to the shared-memory arrays during
# initialization.
# =========================================================

earth_r = None
earth_v = None
comet_r = None
comet_v = None

earth_r_shm = None
earth_v_shm = None
comet_r_shm = None
comet_v_shm = None

worker_leaving_times = None
worker_arriving_times = None

# =========================================================
# Worker initialization
# =========================================================

def initialize_worker(
    earth_r_name,
    earth_r_shape,
    earth_v_name,
    earth_v_shape,
    comet_r_name,
    comet_r_shape,
    comet_v_name,
    comet_v_shape,
    leaving_times_data,
    arriving_times_data
):
    """
    Attach each worker process to the shared ephemeris arrays.

    The arrays are treated as read-only by the workers.
    """

    global earth_r
    global earth_v
    global comet_r
    global comet_v

    global earth_r_shm
    global earth_v_shm
    global comet_r_shm
    global comet_v_shm

    global worker_leaving_times
    global worker_arriving_times

    # Attach to shared memory
    earth_r_shm = shared_memory.SharedMemory(name=earth_r_name)

    earth_v_shm = shared_memory.SharedMemory(name=earth_v_name)

    comet_r_shm = shared_memory.SharedMemory(name=comet_r_name)

    comet_v_shm = shared_memory.SharedMemory(name=comet_v_name)

    # Create NumPy views onto shared memory
    earth_r = np.ndarray(earth_r_shape, dtype=np.float64, buffer=earth_r_shm.buf)

    earth_v = np.ndarray(earth_v_shape, dtype=np.float64, buffer=earth_v_shm.buf)

    comet_r = np.ndarray(comet_r_shape, dtype=np.float64, buffer=comet_r_shm.buf)

    comet_v = np.ndarray(comet_v_shape, dtype=np.float64, buffer=comet_v_shm.buf)

    # Time arrays are tiny, so copying these into each worker
    # is not an issue.
    worker_leaving_times = leaving_times_data
    worker_arriving_times = arriving_times_data


# =========================================================
# Solve one chunk of Lambert cases
# =========================================================

def solve_chunk(jobs):
    """
    Solve a chunk of Lambert cases.

    jobs contains tuples of:

        (departure_index, arrival_index)

    The worker accesses the ephemeris through shared memory.
    """
    results = []

    for departure_index, arrival_index in jobs:

        # Retrieve states from shared memory
        r_earth = earth_r[departure_index]
        v_earth = earth_v[departure_index]

        r_comet = comet_r[arrival_index]
        v_comet = comet_v[arrival_index]

        # Time of flight
        dt = (worker_arriving_times[arrival_index] - worker_leaving_times[departure_index])

        # -------------------------------------------------
        # Lambert
        # -------------------------------------------------
        try:

            v_departure, v_arrival = lambert(
                r_earth,
                r_comet,
                float(dt),
                MU_SUN, 
            )

        except (ValueError, RuntimeError, FloatingPointError):
            # Invalid/no Lambert solution
            continue

        # -------------------------------------------------
        # Departure burn vector
        # -------------------------------------------------

        delta_v_vector = (v_departure - v_earth)

        delta_v = np.linalg.norm(delta_v_vector)

        # Unit direction of departure burn
        if delta_v > 0.0:
            delta_v_unit = (delta_v_vector / delta_v)
        else:
            delta_v_unit = np.zeros(3)


        # -------------------------------------------------
        # Arrival relative velocity
        # -------------------------------------------------

        relative_velocity_vector = (v_arrival - v_comet)
        v_relative_comet = np.linalg.norm(relative_velocity_vector)


        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append((
            departure_index,
            arrival_index,
            dt / 86400.0,

            delta_v,

            delta_v_vector[0],
            delta_v_vector[1],
            delta_v_vector[2],

            delta_v_unit[0],
            delta_v_unit[1],
            delta_v_unit[2],

            v_relative_comet,

            relative_velocity_vector[0],
            relative_velocity_vector[1],
            relative_velocity_vector[2]
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
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Earth -> 3I/ATLAS Lambert trajectory sweep")
    print("=" * 60)

    print(f"Departure dates: {len(leaving_date_list):,}")

    print(f"Arrival dates:   {len(arriving_date_list):,}")

    total_cases = (len(leaving_date_list) * len(arriving_date_list))
    print(f"Total possible cases: {total_cases:,}")

    print(f"Using {NUM_PROCESSES} worker processes")

    print(f"Chunk size: {CHUNK_SIZE:,}\n")

    # =====================================================
    # Get ephemeris
    # =====================================================

    orbits = Ephemeris()

    print("Downloading Earth ephemeris...")
    earth_data = orbits.get_state("'399'", "'2025-01-01'","'2026-06-30'")

    print("Downloading 3I/ATLAS ephemeris...")
    comet_data = orbits.get_state("'DES=1004083'", "'2025-03-01'", "'2030-12-31'")

    print("Ephemeris downloaded.\n")

    # =====================================================
    # Convert ephemeris to NumPy arrays
    # =====================================================

    earth_r_array = np.asarray([state["r"] for state in earth_data], dtype=np.float64)

    earth_v_array = np.asarray([state["v"] for state in earth_data], dtype=np.float64)

    comet_r_array = np.asarray([state["r"] for state in comet_data], dtype=np.float64)

    comet_v_array = np.asarray([state["v"] for state in comet_data], dtype=np.float64)

    # =====================================================
    # Sanity check ephemeris sizes
    # =====================================================

    if len(earth_r_array) != len(leaving_date_list):
        raise ValueError("Number of Earth ephemeris states does not match number of departure dates.")

    if len(comet_r_array) != len(arriving_date_list):
        raise ValueError("Number of comet ephemeris states does not match number of arrival dates.")

    # =====================================================
    # Load previously completed cases
    # =====================================================

    completed_cases = set()

    file_exists = os.path.exists(CSV_FILE)

    if file_exists:
        print(f"Loading completed cases from {CSV_FILE}...")

        with open(CSV_FILE, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:
                completed_cases.add(
                    (row["departure_date"], row["arrival_date"])
                )

        print(f"Previously completed: {len(completed_cases):,}")

    else:
        print("No existing CSV found. Starting from zero.\n")


    # =====================================================
    # Create shared memory
    # =====================================================

    earth_r_shm = None
    earth_v_shm = None
    comet_r_shm = None
    comet_v_shm = None

    try:
        print("Creating shared ephemeris memory...")

        earth_r_shm, _ = create_shared_array(earth_r_array)

        earth_v_shm, _ = create_shared_array(earth_v_array)

        comet_r_shm, _ = create_shared_array(comet_r_array)

        comet_v_shm, _ = create_shared_array(comet_v_array)

        print("Shared memory ready.\n")

        # =================================================
        # Open CSV
        # =================================================

        csv_mode = "a" if file_exists else "w"

        with open(CSV_FILE, csv_mode, newline="") as file:

            writer = csv.writer(file)

            # Write header for a new file
            if not file_exists:

                writer.writerow([
                    "departure_date",
                    "arrival_date",
                    "time_of_flight_days",

                    "delta_v",

                    "dv_x",
                    "dv_y",
                    "dv_z",

                    "dv_unit_x",
                    "dv_unit_y",
                    "dv_unit_z",

                    "v_relative_comet",

                    "vrel_x",
                    "vrel_y",
                    "vrel_z"
                ])

                file.flush()

            # =================================================
            # Create process pool
            # =================================================

            with ProcessPoolExecutor(
                max_workers=NUM_PROCESSES,
                initializer=initialize_worker,
                initargs=(
                    earth_r_shm.name,
                    earth_r_array.shape,

                    earth_v_shm.name,
                    earth_v_array.shape,

                    comet_r_shm.name,
                    comet_r_array.shape,

                    comet_v_shm.name,
                    comet_v_array.shape,

                    leaving_times,
                    arriving_times
                )
            ) as executor:

                # =============================================
                # Generate chunks without creating one giant
                # list of all 334,000 jobs.
                # =============================================

                def generate_chunks():

                    current_chunk = []

                    for i, departure_date in enumerate(leaving_date_list):
                        for j, arrival_date in enumerate(arriving_date_list):

                            # Skip already-completed cases
                            if (departure_date, arrival_date) in completed_cases:
                                continue
                            elif (datetime.strptime(departure_date, "%Y-%m-%d") + timedelta(days=30)) >= datetime.strptime(arrival_date, "%Y-%m-%d"):
                                continue

                            current_chunk.append((i, j))

                            if len(current_chunk) >= CHUNK_SIZE:
                                yield current_chunk
                                current_chunk = []

                    if current_chunk:
                        yield current_chunk

                # =============================================
                # Submit chunks while limiting the number
                # simultaneously in memory.
                # =============================================
                pending = set()

                chunks_submitted = 0
                cases_completed = len(completed_cases)

                print("Starting Lambert calculations...\n")

                for chunk in generate_chunks():

                    future = executor.submit(solve_chunk, chunk)
                    pending.add(future)

                    chunks_submitted += 1

                    # Keep only a limited number of chunks
                    # waiting/running at once.
                    if len(pending) >= MAX_PENDING_CHUNKS:

                        done, pending = wait(pending, return_when=FIRST_COMPLETED)

                        for completed_future in done:
                            chunk_results = (completed_future.result())

                            for result in chunk_results:

                                (
                                    departure_index,
                                    arrival_index,
                                    tof_days,

                                    delta_v,

                                    dv_x,
                                    dv_y,
                                    dv_z,

                                    dv_unit_x,
                                    dv_unit_y,
                                    dv_unit_z,

                                    v_relative_comet,

                                    vrel_x,
                                    vrel_y,
                                    vrel_z
                                ) = result

                                writer.writerow([
                                    leaving_date_list[departure_index],

                                    arriving_date_list[arrival_index],

                                    tof_days,

                                    delta_v,

                                    dv_x,
                                    dv_y,
                                    dv_z,

                                    dv_unit_x,
                                    dv_unit_y,
                                    dv_unit_z,

                                    v_relative_comet,

                                    vrel_x,
                                    vrel_y,
                                    vrel_z
                                ])

                                cases_completed += 1

                            # Flush after each completed chunk
                            file.flush()

                            percentage = (cases_completed / total_cases* 100)

                            print(f"Completed: {cases_completed:,}/{total_cases:,} ({percentage:.2f}%)")

                # =============================================
                # Wait for remaining chunks
                # =============================================

                while pending:

                    done, pending = wait(pending, return_when=FIRST_COMPLETED)

                    for completed_future in done:
                        chunk_results = (completed_future.result())

                        for result in chunk_results:

                            (
                                departure_index,
                                arrival_index,
                                tof_days,

                                delta_v,

                                dv_x,
                                dv_y,
                                dv_z,

                                dv_unit_x,
                                dv_unit_y,
                                dv_unit_z,

                                v_relative_comet,

                                vrel_x,
                                vrel_y,
                                vrel_z
                            ) = result


                            writer.writerow([
                                leaving_date_list[departure_index],

                                arriving_date_list[arrival_index],

                                tof_days,

                                delta_v,

                                dv_x,
                                dv_y,
                                dv_z,

                                dv_unit_x,
                                dv_unit_y,
                                dv_unit_z,

                                v_relative_comet,

                                vrel_x,
                                vrel_y,
                                vrel_z
                            ])

                            cases_completed += 1

                        file.flush()

                        percentage = (cases_completed/ total_cases * 100)

                        print(f"Completed: {cases_completed:,}/{total_cases:,} ({percentage:.2f}%)")

        print("=" * 60)
        print("Calculation complete.")
        print(f"Results saved to: {CSV_FILE}")
        print("=" * 60)

    finally:

        # =====================================================
        # Clean up shared memory
        # =====================================================

        if earth_r_shm is not None:
            earth_r_shm.close()
            earth_r_shm.unlink()

        if earth_v_shm is not None:
            earth_v_shm.close()
            earth_v_shm.unlink()

        if comet_r_shm is not None:
            comet_r_shm.close()
            comet_r_shm.unlink()

        if comet_v_shm is not None:
            comet_v_shm.close()
            comet_v_shm.unlink()


# =========================================================
# Windows multiprocessing entry point
# =========================================================

if __name__ == "__main__":
    main()