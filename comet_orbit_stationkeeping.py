import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from datetime import timedelta, datetime

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris

# ============================================================
# Constants
# ============================================================

MU_COMET = 2202.52                    # m^3/s^2
MU_SUN = 1.32712440041279419e20       # m^3/s^2

YEAR = 365.25 * 86400.0               # seconds

START_DATE = datetime(2033, 2, 25)


# ============================================================
# Utility functions
# ============================================================

def norm(x):
    return np.linalg.norm(x)


def get_comet_state(comet: CelestialBody, date):
    """
    Adapt this function if your CelestialBody.get_state()
    returns its data in a different format.

    Expected:
        position [m]
        velocity [m/s]
    """
    t = (date - START_DATE).total_seconds()

    r = comet.get_state(t)
    v = comet.get_velocity(t)

    return r, v


# ============================================================
# Solar differential acceleration
# ============================================================

def solar_differential_acceleration(r_comet, r_rel):
    """
    Acceleration of spacecraft relative to comet due to the Sun.

    a_rel_sun =
        a_sun(spacecraft) - a_sun(comet)

    where

        r_spacecraft = r_comet + r_rel
    """

    r_sc = r_comet + r_rel

    a_sc = -MU_SUN * r_sc / norm(r_sc)**3
    a_comet = -MU_SUN * r_comet / norm(r_comet)**3

    return a_sc - a_comet


# ============================================================
# Relative spacecraft dynamics
# ============================================================

def make_dynamics(comet, epoch):
    """
    Creates the ODE function for the spacecraft relative
    to the comet.

    State:
        y = [x, y, z, vx, vy, vz]

    All relative to the comet.
    """

    def dynamics(t, y):

        r_rel = y[:3]
        v_rel = y[3:]

        date = START_DATE + timedelta(seconds=float(t))

        r_comet, v_comet = get_comet_state(comet, date)

        # Gravity from comet
        a_comet = -MU_COMET * r_rel / norm(r_rel)**3

        # Differential gravity from Sun
        a_sun = solar_differential_acceleration(
            r_comet,
            r_rel
        )

        a_total = a_comet + a_sun

        return np.hstack((v_rel, a_total))

    return dynamics


# ============================================================
# Initial circular orbit
# ============================================================

def circular_orbit_initial_state(
    comet,
    epoch,
    altitude,
    plane_normal=np.array([0.0, 0.0, 1.0])
):
    """
    Creates an initial circular orbit around the comet.

    altitude:
        altitude above comet center [m]

    plane_normal:
        normal vector defining orbital plane.
    """

    r0_mag = altitude

    # Start on +x axis
    r_hat = np.array([1.0, 0.0, 0.0])

    # Make sure r_hat lies in orbital plane
    plane_normal = plane_normal / norm(plane_normal)

    r_hat -= np.dot(r_hat, plane_normal) * plane_normal
    r_hat /= norm(r_hat)

    # Tangential direction
    t_hat = np.cross(plane_normal, r_hat)
    t_hat /= norm(t_hat)

    r_rel = r0_mag * r_hat

    # Two-body circular velocity
    v_circular = np.sqrt(MU_COMET / r0_mag)

    v_rel = v_circular * t_hat

    return np.hstack((r_rel, v_rel))


# ============================================================
# Station keeping simulation
# ============================================================

def simulate_station_keeping(
    comet,
    epoch,
    altitude_km,
    simulation_days=365.25,
    correction_fraction_of_orbit=0.25,
    rtol=1e-9,
    atol=1e-6
):
    """
    Simulates station keeping around the comet.

    A correction is performed every fraction of an orbit.

    Returns:
        total_dv [m/s]
        number_of_burns
        history
    """

    altitude = altitude_km * 1000.0

    # --------------------------------------------------------
    # Initial circular orbit
    # --------------------------------------------------------

    y = circular_orbit_initial_state(
        comet,
        epoch,
        altitude
    )

    # Circular orbit characteristics
    v_circular = np.sqrt(MU_COMET / altitude)

    orbital_period = (
        2.0 * np.pi *
        np.sqrt(altitude**3 / MU_COMET)
    )

    correction_interval = (
        correction_fraction_of_orbit *
        orbital_period
    )

    total_time = simulation_days * 86400.0

    dynamics = make_dynamics(comet, epoch)

    # --------------------------------------------------------
    # Simulation variables
    # --------------------------------------------------------

    current_time = 0.0
    total_dv = 0.0
    burn_count = 0

    history = []

    # --------------------------------------------------------
    # Propagate between station keeping burns
    # --------------------------------------------------------

    while current_time < total_time:

        next_time = min(
            current_time + correction_interval,
            total_time
        )

        sol = solve_ivp(
            dynamics,
            (current_time, next_time),
            y,
            method="DOP853",
            rtol=rtol,
            atol=atol
        )

        y = sol.y[:, -1]
        current_time = next_time

        # ----------------------------------------------------
        # Current relative state
        # ----------------------------------------------------

        r = y[:3]
        v = y[3:]

        r_mag = norm(r)

        # ----------------------------------------------------
        # Desired circular velocity at CURRENT position
        # ----------------------------------------------------

        r_hat = r / r_mag

        # Angular momentum direction
        h = np.cross(r, v)

        if norm(h) < 1e-12:
            break

        h_hat = h / norm(h)

        # Tangential direction
        t_hat = np.cross(h_hat, r_hat)
        t_hat /= norm(t_hat)

        # Desired circular velocity
        v_required = np.sqrt(MU_COMET / r_mag)

        v_desired = v_required * t_hat

        # ----------------------------------------------------
        # Required impulsive correction
        # ----------------------------------------------------

        delta_v_vector = v_desired - v

        delta_v = norm(delta_v_vector)

        total_dv += delta_v
        burn_count += 1

        # ----------------------------------------------------
        # Apply burn
        # ----------------------------------------------------

        y[3:] = v_desired

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        history.append({
            "time_days": current_time / 86400.0,

            "radius_km": r_mag / 1000.0,

            "velocity_mps": norm(v),

            "desired_velocity_mps": v_required,

            "delta_v_mps": delta_v,

            "total_dv_mps": total_dv
        })

    return total_dv, burn_count, history


# ============================================================
# Run sweep
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Your existing comet object
    # --------------------------------------------------------

    orbits = Ephemeris()

    departurte_date = START_DATE
    arrival_date = datetime(2035, 6, 10)

    comet = CelestialBody(
        "3I/ATLAS",
        2202.52,
        orbits,
        "'DES=1004083'",
        departurte_date,
        arrival_date,
        vec_type=2
    )

    epoch = departurte_date

    # --------------------------------------------------------
    # Altitudes to investigate
    # --------------------------------------------------------

    altitudes_km = np.array([
        50,
        100,
        1000,
        10000,
        100000,
        500000,
    ])

    # --------------------------------------------------------
    # Correction frequencies to investigate
    #
    # 0.25 = correction every 1/4 orbit
    # 0.5  = correction every 1/2 orbit
    # 1.0  = correction every orbit
    # 2.0  = correction every 2 orbits
    # 5.0  = correction every 5 orbits
    # --------------------------------------------------------

    correction_fractions = [
        0.25,
        0.5,
        1.0,
        2.0,
        5.0
    ]

    # Dictionary to store results
    results = {}

    # --------------------------------------------------------
    # Run simulations
    # --------------------------------------------------------

    for correction_fraction in correction_fractions:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Correction interval = "
            f"{correction_fraction} orbit(s)"
        )

        print(
            "=" * 60
        )

        station_keeping_dv = []
        burn_counts = []
        orbital_periods = []

        for altitude in altitudes_km:

            print(
                f"\nSimulating altitude = "
                f"{altitude:.2f} km"
            )

            # ------------------------------------------------
            # Nominal orbital period
            # ------------------------------------------------

            r = altitude * 1000.0

            period = (
                2 * np.pi *
                np.sqrt(
                    r**3 /
                    MU_COMET
                )
            )

            print(
                f"  Orbital period = "
                f"{period / 3600:.2f} hours"
            )

            # ------------------------------------------------
            # Station keeping simulation
            # ------------------------------------------------

            dv, burns, history = simulate_station_keeping(
                comet=comet,
                epoch=epoch,
                altitude_km=altitude,
                simulation_days=365.25,

                correction_fraction_of_orbit=
                    correction_fraction,

                rtol=1e-9,
                atol=1e-6
            )

            station_keeping_dv.append(dv)
            burn_counts.append(burns)
            orbital_periods.append(period)

            print(
                f"  Station keeping Δv = "
                f"{dv:.8f} m/s/year"
            )

            print(
                f"  Number of burns = "
                f"{burns}"
            )

        # ----------------------------------------------------
        # Save this correction frequency's results
        # ----------------------------------------------------

        results[correction_fraction] = {
            "dv": np.array(station_keeping_dv),
            "burns": np.array(burn_counts),
            "periods": np.array(orbital_periods)
        }

    # ========================================================
    # Plot Δv/year
    # ========================================================

    plt.figure(figsize=(10, 7))

    for correction_fraction in correction_fractions:

        dv = results[correction_fraction]["dv"]

        plt.loglog(
            altitudes_km,
            dv,
            "o-",
            label=(
                f"Every "
                f"{correction_fraction:g} orbit"
            )
        )

    plt.xlabel(
        "Altitude above comet [km]"
    )

    plt.ylabel(
        "Station Keeping Δv [m/s/year]"
    )

    plt.title(
        "3I/ATLAS Station Keeping Δv "
        "vs Altitude"
    )

    plt.grid(
        True,
        which="both"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()