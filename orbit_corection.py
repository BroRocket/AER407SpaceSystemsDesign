
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from datetime import datetime

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "lambert_results_detailed.csv"
OUTPUT_CSV = "lambert_results_corrected.csv"

MU_SUN = 1.32712440041279419e20 # km^3/s^2

# Stop the optimizer when the position residual is this small.
# This is a numerical tolerance, not a guarantee of mission accuracy.
POSITION_TOL_M = 1

# Maximum number of optimizer function evaluations per trajectory.
MAX_NFEV = 100

# Save progress after every successfully processed row.
SAVE_EVERY = 1

#==============================================================

def targeting_residual(v0, r0, t0, tf, comet_position, bodies):
    """
    v0: trial heliocentric departure velocity [km/s]
    r0: heliocentric departure position [km]
    t0, tf: departure and arrival epochs
    comet_position: comet position at tf [km]
    propagate: your multi-body propagator
    """

    probe = Spacecraft("test", r0, v0)
    probe.propogate_orbit(t0, tf, MU_SUN, bodies)

    r_sc_final = np.array([probe.r[0][-1], probe.r[1][-1], probe.r[2][-1]])

    return r_sc_final - comet_position


def correct_trajectory(row, orbits_api: Ephemeris):
    departure_date = datetime.strptime(row["departure_date"], "%Y-%m-%d")
    arrival_date = datetime.strptime(row["arrival_date"], "%Y-%m-%d")

    comet = CelestialBody("3I/ATLAS", 0, orbits_api, "'DES=1004083'", departure_date, arrival_date, vec_type=2)
    #print("Loaded Comet")
    earth = CelestialBody("Earth", 3.986004418e14, orbits_api, "'399'", departure_date, arrival_date, vec_type=2)
    #print("Loaded Earth")
    mars = CelestialBody("Mars", 4.282837e13, orbits_api, "'499'", departure_date, arrival_date, vec_type=2)
    #print("Loaded Mars")
    jupiter = CelestialBody("Jupiter", 1.26686534e17, orbits_api, "'599'", departure_date, arrival_date, vec_type=2)
    #print("Loaded Jupiter")

    # Lambert burn vector stored in the CSV + earth vel
    v0_guess  = [row["dv_x"]*1000 + earth.v[0][0], row["dv_y"]*1000 + earth.v[1][0], row["dv_z"]*1000 + earth.v[2][0]]

    # Run least squares to target the comet's arrival position
    result = least_squares(
        targeting_residual,
        x0=v0_guess,
        args=(
            [earth.r[0][0] + 6371000 + 400000, earth.r[1][0], earth.r[2][0]],
            0,
            (arrival_date-departure_date).total_seconds(),
            np.array([comet.r[0][-1], comet.r[1][-1], comet.r[2][-1]]),
            [earth, mars, jupiter]
        ),
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
    )

    v0_corrected = result.x

    # Corrected departure excess velocity relative to Earth
    dv_corrected_vector = v0_corrected - np.array([earth.v[0][0], earth.v[1][0], earth.v[2][0]])
    dv_corrected = np.linalg.norm(dv_corrected_vector)

    # get final state
    probe = Spacecraft("test", [earth.r[0][0] + 6371000 + 400000, earth.r[1][0], earth.r[2][0]], v0_corrected)
    probe.propogate_orbit(0, (arrival_date-departure_date).total_seconds(), MU_SUN, [earth, mars, jupiter])

    # Arrival velocity relative to the comet
    vrel_vector = np.array([probe.v[0][-1], probe.v[1][-1], probe.v[2][-1]]) - np.array([comet.v[0][-1], comet.v[1][-1], comet.v[2][-1]])
    vrel = np.linalg.norm(vrel_vector)

    # Position miss distance
    miss_vector = np.array([probe.r[0][-1], probe.r[1][-1], probe.r[2][-1]]) - np.array([comet.r[0][-1], comet.r[1][-1], comet.r[2][-1]])
    miss_distance = np.linalg.norm(miss_vector)

    return {
        "departure_date": departure_date.strftime("'%Y-%m-%d'"),
        "arrival_date": arrival_date.strftime("'%Y-%m-%d'"),

        "lambert_delta_v": row["delta_v"]*1000,
        "corrected_delta_v": dv_corrected,

        "corrected_dv_x": dv_corrected_vector[0],
        "corrected_dv_y": dv_corrected_vector[1],
        "corrected_dv_z": dv_corrected_vector[2],

        "corrected_v0_x": v0_corrected[0],
        "corrected_v0_y": v0_corrected[1],
        "corrected_v0_z": v0_corrected[2],

        "arrival_v_relative_comet": vrel,
        "arrival_vrel_x": vrel_vector[0],
        "arrival_vrel_y": vrel_vector[1],
        "arrival_vrel_z": vrel_vector[2],

        "miss_distance_m": miss_distance,

        "optimizer_success": result.success,
        "optimizer_message": result.message,
        "optimizer_nfev": result.nfev,
        "optimizer_cost": result.cost
    }


# ============================================================
# BATCH PROCESSING WITH RESTART SUPPORT
# ============================================================

def run_batch():
    df = pd.read_csv(INPUT_CSV)
    orbits_api = Ephemeris()

    required_columns = {"departure_date", "arrival_date", "delta_v", "dv_x", "dv_y", "dv_z"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    # Load already-completed rows if resuming
    if os.path.exists(OUTPUT_CSV):
        completed_df = pd.read_csv(OUTPUT_CSV)

        completed_keys = set(zip(completed_df["departure_date"], completed_df["arrival_date"]))

        print(f"Resuming: {len(completed_keys):,} trajectories already have output rows.")

    else:
        completed_keys = set()

    total = len(df)
    processed = 0
    failed = 0

    for index, row in df.iterrows():

        key = (str(row["departure_date"]), str(row["arrival_date"]))

        if key in completed_keys:
            continue

        try:
            result = correct_trajectory(row, orbits_api)

            # Append one completed result immediately
            result_df = pd.DataFrame([result])

            result_df.to_csv(OUTPUT_CSV, mode="a", header=not os.path.exists(OUTPUT_CSV), index=False)

            completed_keys.add(key)
            processed += 1

        except Exception as exc:
            failed += 1

            print(f"\nFAILED row {index}: {key[0]} -> {key[1]}\n{exc}")

        if (processed + failed) % 100 == 0:
            print(f"Progress: {processed + failed:,}/{total:,} | successful: {processed:,} | failed: {failed:,}")

    print("\nBatch complete.")
    print(f"Successful: {processed:,}")
    print(f"Failed: {failed:,}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":


    departurte_date = datetime(2025, 1, 1)
    arrival_date = datetime(2028, 12, 31)

    orbits = Ephemeris()
    comet = CelestialBody("3I/ATLAS", 0, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
    print("Loaded Comet")
    mercury = CelestialBody("Venus", 2.2031870799e13, orbits, "'199'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Mercury")
    venus = CelestialBody("Venus", 3.24858592e14, orbits, "'299'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Venus")
    earth = CelestialBody("Earth", 3.986004418e14, orbits, "'399'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Earth")
    mars = CelestialBody("Mars", 4.282837e13, orbits, "'499'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Mars")
    jupiter = CelestialBody("Jupiter", 1.26686534e17, orbits, "'599'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Jupiter")
    saturn = CelestialBody("Saturn", 3.7931187e16, orbits, "'699'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Saturn")
    uranus = CelestialBody("Uranus", 5.793939e15, orbits, "'799'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Uranus")
    neptune = CelestialBody("Neptune", 6.836529e15, orbits, "'899'", departurte_date, arrival_date, vec_type=2)
    print("Loaded Neptune")

    
    bodies = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

    run_batch()


#Lambert departure velocity is your initial guess
# departurte_date = datetime(2025, 9, 17)
# arrival_date = datetime(2028, 6, 17)

# orbits = Ephemeris()
# comet = CelestialBody("3I/ATLAS", 0, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
# print("Loaded Comet")
# mercury = CelestialBody("Venus", 2.2031870799e13, orbits, "'199'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Mercury")
# venus = CelestialBody("Venus", 3.24858592e14, orbits, "'299'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Venus")
# earth = CelestialBody("Earth", 3.986004418e14, orbits, "'399'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Earth")
# mars = CelestialBody("Mars", 4.282837e13, orbits, "'499'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Mars")
# jupiter = CelestialBody("Jupiter", 1.26686534e17, orbits, "'599'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Jupiter")
# saturn = CelestialBody("Saturn", 3.7931187e16, orbits, "'699'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Saturn")
# uranus = CelestialBody("Uranus", 5.793939e15, orbits, "'799'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Uranus")
# neptune = CelestialBody("Neptune", 6.836529e15, orbits, "'899'", departurte_date, arrival_date, vec_type=2)
# print("Loaded Neptune")

# bodies = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

# v0_guess = [1723.289 + earth.v[0][0], 39738.198 + earth.v[1][0], -4225.025 + earth.v[2][0]]

# solution = least_squares(
#     targeting_residual,
#     x0=v0_guess,
#     args=(
#         [earth.r[0][0] + 6371000 + 400000, earth.r[1][0], earth.r[2][0]],
#         0,
#         (arrival_date-departurte_date).total_seconds(),
#         np.array([comet.r[0][-1], comet.r[1][-1], comet.r[2][-1]]),
#         bodies
#     ),
#     xtol=1e-10,
#     ftol=1e-10,
#     gtol=1e-10
# )

# print(solution.x)