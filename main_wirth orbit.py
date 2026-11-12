
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbit_Mission_Simulation.visualizer import Visualizer
# New helper (same folder as this file, or adjust the import path to where you keep it)
from targeting import NBody, leo_departure_state, correct_velocity, jupiter_approach_target

AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s
DAY = 86400.0 #24 * 60 * 60

orbits = Ephemeris()

# ---------------------------------------------------------------------------
# Lambert case to simulate
#
# Pick a row of the results CSV written by multithread_jupiter_lambert.py
# (rows are sorted best first, so CASE_INDEX = 0 is the lowest on-board
# delta-v). Dates, the departure v-infinity, the Jupiter in/out v-infinity
# and the flyby periapsis all come from that row.
# ---------------------------------------------------------------------------
RESULTS_CSV = os.path.join("Jupiter Gravity Assist Data", "jupiter_assist_results.csv")
LEG1_CSV = os.path.join("Jupiter Gravity Assist Data", "leg1_earth_jupiter.csv")   # only used for older results CSVs
LEG2_CSV = os.path.join("Jupiter Gravity Assist Data", "leg2_jupiter_comet.csv")   # only used for older results CSVs
CASE_INDEX = 0 #344 what alll previous design was based off of
DAYS_AFTER_RENDEZVOUS = 365   # keep simulating this long after the rendezvous
STANDOFF = 50e3 # m

def case_vector(case, prefix):
    return np.array([case[prefix + "_x"], case[prefix + "_y"], case[prefix + "_z"]], dtype=float)


def jupiter_vinf_vectors(case):
    """
    Jupiter-relative v-infinity in and out (km/s) for one results row.

    Newer results CSVs have them as columns. For an older results CSV they are
    looked up in the two leg CSVs instead (leg 1 vinf_end, leg 2 vinf_start).
    """
    if "vinf_jupiter_in_x" in case.index:
        return case_vector(case, "vinf_jupiter_in"), case_vector(case, "vinf_jupiter_out")

    leg1 = pd.read_csv(LEG1_CSV)
    leg2 = pd.read_csv(LEG2_CSV)
    row1 = leg1[(leg1["start_date"] == case["departure_date"]) & (leg1["end_date"] == case["jupiter_date"])].iloc[0]
    row2 = leg2[(leg2["start_date"] == case["jupiter_date"]) & (leg2["end_date"] == case["arrival_date"])].iloc[0]
    return case_vector(row1, "vinf_end"), case_vector(row2, "vinf_start")


case = pd.read_csv(RESULTS_CSV).iloc[CASE_INDEX]

departurte_date = datetime.strptime(case["departure_date"], "%Y-%m-%d")
jupiter_date = datetime.strptime(case["jupiter_date"], "%Y-%m-%d")
rendezvous_date = datetime.strptime(case["arrival_date"], "%Y-%m-%d")
arrival_date = rendezvous_date + timedelta(days=DAYS_AFTER_RENDEZVOUS)  # end of simulation

print(f"Simulating case {CASE_INDEX}: launch {case['departure_date']}, Jupiter {case['jupiter_date']}, "
      f"rendezvous {case['arrival_date']}")
print(f"  Lambert: LEO burn {case['leo_burn_dv']:.3f} km/s, Jupiter periapsis altitude {case['periapsis_altitude_km']:,.0f} km, "
      f"flyby burn {case['flyby_dv']*1000:.1f} m/s, arrival {case['arrival_speed']:.3f} km/s")

# departurte_date = datetime(2028, 6, 17) #1
# arrival_date = datetime(2028, 6, 30)

comet = CelestialBody("3I/ATLAS", 2202.52, departurte_date, arrival_date, csv_file="Ephemeris/3IATLAS.csv") # all in m^3/s^2
print("Loaded Comet")
mercury = CelestialBody("Mercury", 2.2031870799e13, departurte_date, arrival_date, csv_file="Ephemeris/MERCURY.csv")
print("Loaded Mercury")
venus = CelestialBody("Venus", 3.24858592e14, departurte_date, arrival_date, csv_file="Ephemeris/VENUS.csv")
print("Loaded Venus")
earth = CelestialBody("Earth", 3.986004418e14, departurte_date, arrival_date, csv_file="Ephemeris/EARTH.csv")
print("Loaded Earth")
mars = CelestialBody("Mars", 4.282837e13, departurte_date, arrival_date, csv_file="Ephemeris/MARS.csv")
print("Loaded Mars")
jupiter = CelestialBody("Jupiter", 1.26686534e17, departurte_date, arrival_date, csv_file="Ephemeris/JUPITER.csv")
print("Loaded Jupiter")
saturn = CelestialBody("Saturn", 3.7931187e16, departurte_date, arrival_date, csv_file="Ephemeris/SATURN.csv")
print("Loaded Saturn")
uranus = CelestialBody("Uranus", 5.793939e15, departurte_date, arrival_date, csv_file="Ephemeris/URANUS.csv")
print("Loaded Uranus")
neptune = CelestialBody("Neptune", 6.836529e15, departurte_date, arrival_date, csv_file="Ephemeris/NEPTUNE.csv")
print("Loaded Neptune")

# comet = CelestialBody("3I/ATLAS", 2202.52, departurte_date, arrival_date, orbits, "'DES=1004083'") # all in m^3/s^2
# print("Loaded Comet")
# mercury = CelestialBody("Mercury", 2.2031870799e13, departurte_date, arrival_date,  orbits, "'199'")
# print("Loaded Mercury")
# venus = CelestialBody("Venus", 3.24858592e14, departurte_date, arrival_date, orbits, "'299'")
# print("Loaded Venus")
# earth = CelestialBody("Earth", 3.986004418e14, departurte_date, arrival_date,  orbits, "'399'")
# print("Loaded Earth")
# mars = CelestialBody("Mars", 4.282837e13, departurte_date, arrival_date,  orbits, "'499'")
# print("Loaded Mars")
# jupiter = CelestialBody("Jupiter", 1.26686534e17, departurte_date, arrival_date,  orbits, "'599'")
# print("Loaded Jupiter")
# saturn = CelestialBody("Saturn", 3.7931187e16, departurte_date, arrival_date, orbits, "'699'")
# print("Loaded Saturn")
# uranus = CelestialBody("Uranus", 5.793939e15, departurte_date, arrival_date,  orbits, "'799'")
# print("Loaded Uranus")
# neptune = CelestialBody("Neptune", 6.836529e15, departurte_date, arrival_date,  orbits, "'899'")
# print("Loaded Neptune")

bodies = [comet, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune] # [earth, mars, jupiter]

# ---------------------------------------------------------------------------
# Route from the chosen Lambert case (J2000 ecliptic, TDB, SI units).
# Times are seconds after departurte_date. The Lambert sweep uses 00:00 on
# each date, so launch, flyby and rendezvous all start at midnight.
# ---------------------------------------------------------------------------
t_launch = 0        # Midnight
t_jupiter = (jupiter_date - departurte_date).total_seconds()
t_arrival = (rendezvous_date - departurte_date).total_seconds()
v_inf_departure = case_vector(case, "vinf_departure") * 1000   # km/s -> m/s, Earth-relative

# Patched-conic heliocentric velocity arriving at / leaving Jupiter: Jupiter's
# velocity at the flyby (from the same ephemeris CSV) plus the v-infinity vectors.
vinf_jupiter_in, vinf_jupiter_out = jupiter_vinf_vectors(case)
v_jupiter_flyby = jupiter.get_velocity(t_jupiter)
v_into_jupiter = v_jupiter_flyby + vinf_jupiter_in * 1000     # m/s
v_out_of_jupiter = v_jupiter_flyby + vinf_jupiter_out * 1000  # m/s
jupiter_periapsis_radius = case["periapsis_radius_km"] * 1e3  # m


# 1) Start the probe AT THE BURN POINT in the 400 km parking orbit with the post-burn
#    velocity. (Starting from the heliocentric "lambert_boost" velocity inside Earth's
#    gravity well loses ~1.1 km/s of excess speed while climbing out.)
r0, v0, burn = leo_departure_state(earth.get_state(t_launch), earth.get_velocity(t_launch),
                                   v_inf_departure, altitude_m=400e3, inclination_deg=28.5)
print(f"Rocket Delta V Required from 400km parking orbit: {burn:.1f} m/s")
 
# 2) Differential correction. The patched conic aims at Jupiter's CENTRE; here the
#    launch velocity is adjusted (~20 m/s) so the probe instead flies through the
#    flyby hyperbola half a day before closest approach.
nbody = NBody(bodies, MU_SUN)
t_aim = t_jupiter - 0.5*DAY
aim_point = jupiter_approach_target(jupiter.get_state(t_aim), v_into_jupiter, v_out_of_jupiter,
                                    jupiter.get_velocity(t_jupiter), jupiter_periapsis_radius, jupiter.mu, 0.5*DAY)
print("Correcting launch velocity so the probe flies the Jupiter flyby...")
y_launch, _ = correct_velocity(nbody, np.r_[r0, v0], t_launch, t_aim, aim_point, tol_m=100.)
print(f"Corrected post-burn state (m, m/s): {y_launch.tolist()}")
 
# >>> CHANGED (single injection burn) ------------------------------------------
# The launch correction is not a separate burn. It only changes the velocity the
# probe has straight after the LEO burn, so the real manoeuvre is ONE injection
# burn from the parking-orbit velocity to the corrected velocity y_launch[3:].
# The parking-orbit velocity at the burn point is circular speed along the same
# direction as the uncorrected (tangential) burn.
r_earth_launch = earth.get_state(t_launch)
v_earth_launch = earth.get_velocity(t_launch)
r_parking = np.linalg.norm(r0 - r_earth_launch)                 # m from Earth's centre
flight_direction = (v0 - v_earth_launch)/np.linalg.norm(v0 - v_earth_launch)
v_parking = v_earth_launch + np.sqrt(earth.mu/r_parking)*flight_direction   # heliocentric, before the burn
injection_burn_vector = y_launch[3:] - v_parking
injection_burn = np.linalg.norm(injection_burn_vector)
print(f"Uncorrected tangential LEO burn: {burn:.1f} m/s; N-body correction to it: {np.linalg.norm(y_launch[3:] - v0):.3f} m/s")
print(f"Single interplanetary injection burn: {injection_burn:.1f} m/s, vector {injection_burn_vector.tolist()} m/s")
print(f"  (the correction changes the burn by {injection_burn - burn:+.1f} m/s)")
# <<< CHANGED -------------------------------------------------------------------
print(f"Initial posiiton: {y_launch[:3]/1000}")
print(f"Earth initila position: {earth.get_state(t_launch)/1000}")
print(f"Orbit Altitude: {np.linalg.norm(earth.get_state(t_launch)/1000 - y_launch[:3]/1000) - 6371}")
 
# >>> CHANGED (correction burn moved to Jupiter periapsis) ---------------------
# Was: t_tcm = t_jupiter + 10*DAY. The correction burn is now made AT closest
# approach, where the probe is fastest (~87 km/s). A burn there changes the
# outgoing speed far more per m/s (Oberth effect), so it costs less.
# Find the periapsis time in the N-body model: coast from launch to the aim
# point half a day before Jupiter, then search the next day for the minimum.
y_aim = nbody.propagate(y_launch, t_launch, t_aim)
approach_sol = nbody.propagate(y_aim, t_aim, t_aim + 1.0*DAY, dense=True)
periapsis = nbody.closest_approach(approach_sol, jupiter.name, t_aim, t_aim + 1.0*DAY)
print(f"Jupiter periapsis before the burn: {periapsis['distance_m']/1e3 - 71492:,.0f} km altitude "
      f"at {periapsis['t']/DAY:.3f} days, {periapsis['relative_speed_m_s']/1e3:.2f} km/s")
 
# Where to make the correction burn: "periapsis", or a number of days after the
# flyby (the old behaviour was 10). Try both: which is cheaper depends on how much of
# the burn is along the flight direction (periapsis helps) or sideways (it doesn't).
CORRECTION_BURN_AT = 1 #1 #"periapsis"
 
# 3) Leg 1: launch -> correction burn, with your Spacecraft class.
t_final = (arrival_date-departurte_date).total_seconds()
if CORRECTION_BURN_AT == "periapsis":
    t_tcm = periapsis["t"]
else:
    t_tcm = t_jupiter + float(CORRECTION_BURN_AT)*DAY
# <<< CHANGED -------------------------------------------------------------------
grid = np.array(earth.times_in_seconds)
leg1 = Spacecraft("Hitchhiker 1", y_launch[:3], y_launch[3:])
leg1.propogate_orbit(t_launch, t_tcm, MU_SUN, celestial_bodies=bodies,
                     t_eval=np.r_[grid[(grid >= t_launch) & (grid < t_tcm)], t_tcm])  # t_eval only picks saved points
 
# 4) Powered flyby: correction burn AT Jupiter periapsis so the probe meets the comet on
#    the rendezvous date. (CHANGED: was 10 days after the flyby.) It also covers the
#    small periapsis burn the Lambert case asked for (flyby_dv).
y_tcm = np.r_[leg1.r[:, -1], leg1.v[:, -1]]
print("Solving the periapsis burn...")
# Aim at a point STANDOFF metres from the comet's centre on its sunward side (clear of the dust tail),
# not at the centre itself: the nucleus is up to a few km across.
r_comet_arr = comet.get_state(t_arrival)
standoff_dir = -r_comet_arr/np.linalg.norm(r_comet_arr)
y_after, _ = correct_velocity(nbody, y_tcm, t_tcm, t_arrival, r_comet_arr + STANDOFF*standoff_dir, tol_m=100.)
tcm = y_after[3:] - y_tcm[3:]
print(f"Correction burn ({CORRECTION_BURN_AT}) at {t_tcm/DAY:.3f} days: {np.linalg.norm(tcm):.3f} m/s, vector {tcm.tolist()} m/s")
v_rel_jupiter = y_tcm[3:] - jupiter.get_velocity(t_tcm)
along_track = np.dot(tcm, v_rel_jupiter/np.linalg.norm(v_rel_jupiter))
print(f"  of which {along_track:+.1f} m/s is along the Jupiter-relative flight direction, "
      f"{np.sqrt(max(np.linalg.norm(tcm)**2 - along_track**2, 0.0)):.1f} m/s sideways")
 
# 5) Coast to the standoff point, reached on the arrival date.
t_rv = t_arrival
 
leg2 = Spacecraft("Hitchhiker 1", y_after[:3], y_after[3:])
leg2.propogate_orbit(t_tcm, t_rv, MU_SUN, celestial_bodies=bodies,
                     t_eval=np.r_[t_tcm, grid[(grid > t_tcm) & (grid < t_rv)], t_rv])
 
# 6) Rendezvous burn at the standoff point. "follow": match the comet's velocity exactly
#    (the probe then slowly falls toward the nucleus under its weak gravity).
#    "orbit": also add the circular-orbit speed around the comet at the standoff distance,
#    in the comet's orbital plane. With mu = 2202.52 m^3/s^2 that is ~0.15 m/s at 100 km
#    (period ~49 days). The comet's mass is very uncertain, and outgassing is not modelled.
rendezvous_mode = "orbit"   # or "follow"
v_before_rv = leg2.v[:, -1]
r_rel = leg2.r[:, -1] - comet.get_state(t_rv)
v_comet_rv = comet.get_velocity(t_rv)
v_target = v_comet_rv.copy()
if rendezvous_mode == "orbit":
    n_hat = np.cross(comet.get_state(t_rv), v_comet_rv); n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_rel); t_hat /= np.linalg.norm(t_hat)
    v_target += np.sqrt(comet.mu/np.linalg.norm(r_rel))*t_hat
dv_rv = v_target - v_before_rv
print(f"Rendezvous burn at {t_rv/DAY:.3f} days, {np.linalg.norm(r_rel)/1e3:.3f} km from the comet: "
      f"{np.linalg.norm(dv_rv):.1f} m/s, vector {dv_rv.tolist()} m/s")
 
# 7) Leg 3: follow the comet until the end of the simulation.
leg3 = Spacecraft("Hitchhiker 1", leg2.r[:, -1], v_target)
leg3.propogate_orbit(t_rv, t_final, MU_SUN, celestial_bodies=bodies,
                     t_eval=np.r_[t_rv, grid[(grid > t_rv) & (grid <= t_final)]])
sep = np.array([np.linalg.norm(leg3.r[:, i] - comet.get_state(t)) for i, t in enumerate(leg3.t)])
rel = np.linalg.norm(leg3.v[:, -1] - comet.get_velocity(leg3.t[-1]))
print(f"After the burn, over {(t_final - t_rv)/DAY:.1f} days: distance from comet centre stays between "
      f"{sep.min()/1e3:.2f} and {sep.max()/1e3:.2f} km (end: {sep[-1]/1e3:.2f} km, relative speed {rel*1000:.1f} mm/s)")
pd.DataFrame({"Time (s)": leg3.t, "Distance from comet (m)": sep}).to_csv("Distance From Comet After Rendezvous.csv", index=False)
 
# Join the legs into one Spacecraft object for the plots below
probe = Spacecraft("Hitchhiker 1", y_launch[:3], y_launch[3:])
probe.t = np.r_[leg1.t, leg2.t[1:], leg3.t[1:]]
probe.r = np.hstack([leg1.r, leg2.r[:, 1:], leg3.r[:, 1:]])
probe.v = np.hstack([leg1.v, leg2.v[:, 1:], leg3.v[:, 1:]])
print("Loaded probe")
 
distance_from_sun = []
for x, y, z in zip(probe.r[0], probe.r[1], probe.r[2]):
    distance_from_sun.append(np.linalg.norm([x, y, z])/AU)
 
df_distance = pd.DataFrame({"Time (s)": probe.t, "Time (Years)": probe.t/(60*60*24*365.25), "Distance From Sun [AU]": distance_from_sun, "X [m]": probe.r[0], "Y [m]": probe.r[1], "Z [m]": probe.r[2]})
 
df_distance.to_csv("Distance From Sun Over Time AU.csv", index=False)
 
vis = Visualizer(probe, bodies)
 
#svis.trajectory_2D('x', 'y')
# vis.trajectory_2D('x', 'z')
# vis.trajectory_2D('y', 'z')
#vis.trajectory_3D()
#vis.animate_trajectory_2D('x', 'y')
#vis.ani.save("whole_trajectory_animation.gif", writer="pillow", fps=30)
#vis.animate_trajecctory_3D()
 
# Closest approaches. The saved points are 30 minutes apart (~150,000 km at Jupiter flyby
# speed), so the sampled minimum is refined by re-integrating densely around it.
def closest(body):
    d = [np.linalg.norm(body.get_state(t) - probe.r[:, i]) for i, t in enumerate(probe.t)]
    i = int(np.argmin(d))
    a, b = max(i - 1, 0), min(i + 1, len(probe.t) - 1)
    sol = nbody.propagate(np.r_[probe.r[:, a], probe.v[:, a]], probe.t[a], probe.t[b], dense=True)
    return nbody.closest_approach(sol, body.name, probe.t[a], probe.t[b], n=2001), sol
 
cj, _ = closest(jupiter)
print(f"Jupiter periapsis: {cj['distance_m']/1e3:,.0f} km from centre ({cj['distance_m']/71492e3:.3f} R_J, "
      f"{cj['distance_m']/1e3 - 71492:,.0f} km altitude) at {cj['t']/DAY:.3f} days, {cj['relative_speed_m_s']/1e3:.2f} km/s")


# >>> NEW (Jupiter-relative position CSV, 100 R_J in -> 100 R_J out) ---------------
# Probe position relative to Jupiter's centre, in Jupiter radii, from the moment it
# comes within JUPITER_CSV_LIMIT_RJ on the way in until it passes that distance again
# on the way out. Frame: same as the ephemeris CSVs (J2000 ecliptic), axes not rotated.
#
# The trajectory is re-integrated densely, in two pieces either side of the
# correction burn (t_tcm), so the velocity jump of a periapsis burn is included.
JUPITER_CSV_FILE = "Probe Position Relative To Jupiter.csv"
JUPITER_CSV_LIMIT_RJ = 100.0
JUPITER_CSV_STEP_S = 60.0          # one row per minute
JUPITER_SEARCH_DAYS = 30.0         # how far either side of periapsis to look for the crossings
R_J_M = 71492e3                    # m, Jupiter equatorial radius

t_search_start = max(t_launch, cj["t"] - JUPITER_SEARCH_DAYS*DAY)
t_search_end = min(t_rv, cj["t"] + JUPITER_SEARCH_DAYS*DAY)

# Piece 1: from the last saved leg-1 point before the search window, up to the burn
i_from = max(int(np.searchsorted(leg1.t, t_search_start)) - 1, 0)
t_piece1_end = min(t_tcm, t_search_end)
piece1 = nbody.propagate(np.r_[leg1.r[:, i_from], leg1.v[:, i_from]], leg1.t[i_from], t_piece1_end, dense=True)

# Piece 2: from just after the burn (only needed if the burn is inside the window)
piece2 = None
if t_tcm < t_search_end:
    piece2 = nbody.propagate(y_after, t_tcm, t_search_end, dense=True)


def probe_state_near_jupiter(t):
    if t <= t_tcm or piece2 is None:
        return piece1.sol(t)
    return piece2.sol(t)


def distance_from_jupiter_rj(t):
    return np.linalg.norm(probe_state_near_jupiter(t)[:3] - jupiter.get_state(t)) / R_J_M


if distance_from_jupiter_rj(t_search_start) < JUPITER_CSV_LIMIT_RJ or distance_from_jupiter_rj(t_search_end) < JUPITER_CSV_LIMIT_RJ:
    raise ValueError(f"The probe is still inside {JUPITER_CSV_LIMIT_RJ} R_J at the edge of the search window; "
                     f"increase JUPITER_SEARCH_DAYS.")

# Times of the inbound and outbound crossings of the limit
t_in = brentq(lambda t: distance_from_jupiter_rj(t) - JUPITER_CSV_LIMIT_RJ, t_search_start, cj["t"], xtol=1e-3)
t_out = brentq(lambda t: distance_from_jupiter_rj(t) - JUPITER_CSV_LIMIT_RJ, cj["t"], t_search_end, xtol=1e-3)

jupiter_times = np.r_[np.arange(t_in, t_out, JUPITER_CSV_STEP_S), t_out]
if t_in < t_tcm < t_out:
    jupiter_times = np.unique(np.r_[jupiter_times, t_tcm])   # include the burn instant

rows = []
for t in jupiter_times:
    y = probe_state_near_jupiter(t)
    r_rel_j = (y[:3] - jupiter.get_state(t)) / R_J_M
    v_rel_j = (y[3:] - jupiter.get_velocity(t)) / 1000.0
    rows.append({
        "Time (s)": t,
        "Date": departurte_date + timedelta(seconds=float(t)),
        "Time from periapsis (h)": (t - cj["t"]) / 3600.0,
        "x (R_J)": r_rel_j[0],
        "y (R_J)": r_rel_j[1],
        "z (R_J)": r_rel_j[2],
        "Distance (R_J)": np.linalg.norm(r_rel_j),
        "Altitude (km)": np.linalg.norm(r_rel_j) * R_J_M / 1000.0 - R_J_M / 1000.0,
        "vx rel (km/s)": v_rel_j[0],
        "vy rel (km/s)": v_rel_j[1],
        "vz rel (km/s)": v_rel_j[2],
        "Speed rel (km/s)": np.linalg.norm(v_rel_j),
    })

df_jupiter = pd.DataFrame(rows)
df_jupiter.to_csv(JUPITER_CSV_FILE, index=False)
print(f"Jupiter-relative position saved to {JUPITER_CSV_FILE}: {len(df_jupiter):,} rows, "
      f"{(t_out - t_in)/3600:.1f} h from {JUPITER_CSV_LIMIT_RJ:.0f} R_J in to {JUPITER_CSV_LIMIT_RJ:.0f} R_J out "
      f"(closest {df_jupiter['Distance (R_J)'].min():.4f} R_J)")
# <<< NEW --------------------------------------------------------------------------
 
print(f"Comet at rendezvous: {comet.get_state(t_rv).tolist()} m, {v_comet_rv.tolist()} m/s")
print(f"Probe after burn: {leg2.r[:, -1].tolist()} m, {v_target.tolist()} m/s")
print(f"Probe before burn: {leg2.r[:, -1].tolist()} m, {v_before_rv.tolist()} m/s")
print(f"Time of rendezvous (After Departure Date): {t_rv} s")
# >>> CHANGED (delta-v summary: one injection burn, periapsis burn) ------------
print(f"Total delta-v: injection {injection_burn:.1f} + correction burn ({CORRECTION_BURN_AT}) {np.linalg.norm(tcm):.1f} "
      f"+ rendezvous {np.linalg.norm(dv_rv):.1f} m/s")
# <<< CHANGED -------------------------------------------------------------------

# ============================================================
# Dense Jupiter flyby animation
# ============================================================

# Closest-approach time from your existing closest() function
t_peri = cj["t"]

# Animation window: 6 hours before and after closest approach
window_hours = 6
t_start = t_peri - window_hours * 3600
t_end = t_peri + window_hours * 3600

# Find a saved probe state BEFORE the start of the animation.
# This provides an initial condition for the new integration.
i_start = np.searchsorted(probe.t, t_start) - 1
i_start = max(i_start, 0)

t_initial = probe.t[i_start]
y_initial = np.concatenate([
    probe.r[:, i_start],
    probe.v[:, i_start]
])

# Re-integrate the probe trajectory through the flyby.
# Dense output allows us to sample the solution at arbitrary times.
flyby_sol = nbody.propagate(
    y_initial,
    t_initial,
    t_end,
    dense=True
)

# Sample the trajectory densely for smooth animation.
n_frames = 800
times = np.linspace(t_start, t_end, n_frames)

# State history: first 3 rows are position, last 3 are velocity.
states = flyby_sol.sol(times)
probe_positions = states[:3, :]

# Calculate probe positions relative to Jupiter, in km.
jupiter_positions = np.column_stack([
    jupiter.get_state(t) for t in times
])

relative_positions = (
    probe_positions - jupiter_positions
) / 1000.0

# Time relative to closest approach, in hours.
relative_times = (times - t_peri) / 3600.0

# Jupiter's mean equatorial radius, in km.
R_J = 71492.0

# ------------------------------------------------------------
# Set up the 3D figure
# ------------------------------------------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Draw Jupiter as a sphere centred at the origin.
theta = np.linspace(0, 2 * np.pi, 50)
phi = np.linspace(0, np.pi, 30)

sphere_x = R_J * np.outer(np.cos(theta), np.sin(phi))
sphere_y = R_J * np.outer(np.sin(theta), np.sin(phi))
sphere_z = R_J * np.outer(np.ones_like(theta), np.cos(phi))

ax.plot_surface(
    sphere_x, sphere_y, sphere_z,
    color="orange",
    alpha=0.8,
    linewidth=0,
    label = "Jupiter"
)

# Full trajectory through the flyby window.
ax.plot(
    relative_positions[0],
    relative_positions[1],
    relative_positions[2],
    color="steelblue",
    alpha=0.45,
    linewidth=1.2,
    label="Probe trajectory"
)

# Animated probe marker and trajectory trail.
probe_marker, = ax.plot(
    [], [], [],
    marker="o",
    color="red",
    markersize=6,
    linestyle="None",
    label="Hitchhiker 1"
)

trail, = ax.plot(
    [], [], [],
    color="red",
    linewidth=2
)

# Keep equal spatial scaling on all axes.
max_range = np.max(np.abs(relative_positions)) * 1.05
max_range = max(max_range, 2 * R_J)

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("Relative X [km]")
ax.set_ylabel("Relative Y [km]")
ax.set_zlabel("Relative Z [km]")

ax.set_title("Jupiter Flyby")
ax.legend(loc="upper right")

time_label = ax.text2D(
    0.03, 0.93, "",
    transform=ax.transAxes
)

# ------------------------------------------------------------
# Animation update function
# ------------------------------------------------------------

trail_length = 40  # Number of recent frames shown in the trail.

def update(frame):
    x = relative_positions[0, frame]
    y = relative_positions[1, frame]
    z = relative_positions[2, frame]

    # Update the probe marker.
    probe_marker.set_data([x], [y])
    probe_marker.set_3d_properties([z])

    # Update the trailing trajectory.
    start = max(0, frame - trail_length)
    trail.set_data(
        relative_positions[0, start:frame + 1],
        relative_positions[1, start:frame + 1]
    )
    trail.set_3d_properties(
        relative_positions[2, start:frame + 1]
    )

    time_label.set_text(
        f"Date: {departurte_date + timedelta(seconds = t_peri) + timedelta(hours = relative_times[frame])}\n"
        f"T - T_periapsis = {relative_times[frame]:+.2f} hours"
    )

    return probe_marker, trail, time_label

ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=10,
    blit=False
)

plt.tight_layout()
plt.show()

# Optional: save the animation as a GIF.
# Uncomment this line if you want to save it.
#ani.save("Jupiter_Flyby.gif", writer=PillowWriter(fps=30))


# ============================================================
# Probe motion relative to 3I/ATLAS after rendezvous burn
# ============================================================

# Simulate the first 60 days after the rendezvous burn.
# Increase this to observe more of the predicted orbit.
# Can't go past the end of the loaded ephemeris (DAYS_AFTER_RENDEZVOUS).
window_days = min(160, DAYS_AFTER_RENDEZVOUS)
t_end_comet = t_rv + window_days * DAY

# Initial state immediately AFTER the rendezvous burn.
# v_target includes the braking burn and any orbital velocity.
y0_comet = np.concatenate([
    leg2.r[:, -1],
    v_target
])

# Re-integrate with dense output.
# NBody includes the gravitational bodies defined in your model.
comet_sol = nbody.propagate(
    y0_comet,
    t_rv,
    t_end_comet,
    dense=True
)

# Sample many points for a smooth animation.
n_frames = 1500
times_comet = np.linspace(t_rv, t_end_comet, n_frames)

# Probe's heliocentric position from the propagated solution.
probe_states = comet_sol.sol(times_comet)
probe_pos = probe_states[:3, :]

# Comet's position at each time.
comet_pos = np.column_stack([
    comet.get_state(t) for t in times_comet
])

# Convert to comet-relative coordinates in km.
relative_pos = (probe_pos - comet_pos) / 1000.0

# Distance from the comet, useful for checking the orbit.
distance_km = np.linalg.norm(relative_pos, axis=0)

print(
    f"Initial separation: {distance_km[0]:.2f} km\n"
    f"Minimum separation: {distance_km.min():.2f} km\n"
    f"Maximum separation: {distance_km.max():.2f} km\n"
    f"Final separation:   {distance_km[-1]:.2f} km"
)

# ============================================================
# Set up 3D animation
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Draw a small illustrative comet nucleus at the origin.
# The actual nucleus radius is uncertain.
nucleus_radius_km = 0.5*(0.5+5) # average size from measuremtns

theta = np.linspace(0, 2 * np.pi, 32)
phi = np.linspace(0, np.pi, 20)

cx = nucleus_radius_km * np.outer(
    np.cos(theta), np.sin(phi)
)
cy = nucleus_radius_km * np.outer(
    np.sin(theta), np.sin(phi)
)
cz = nucleus_radius_km * np.outer(
    np.ones_like(theta), np.cos(phi)
)

ax.plot_surface(
    cx, cy, cz,
    color="saddlebrown",
    alpha=0.9,
    linewidth=0,
    label = "3I/ATLAS"
)

# Full predicted relative trajectory.
ax.plot(
    relative_pos[0],
    relative_pos[1],
    relative_pos[2],
    color="steelblue",
    alpha=0.45,
    linewidth=1,
    label="Predicted relative trajectory"
)

# Animated probe and its recent trail.
probe_marker, = ax.plot(
    [], [], [],
    marker="o",
    color="red",
    markersize=7,
    linestyle="None",
    label="Hitchhiker 1"
)

trail, = ax.plot(
    [], [], [],
    color="red",
    linewidth=1.8
)

# Use equal axis scaling.
# A minimum of 120 km shows the nominal 100 km standoff orbit.
axis_limit_km = max(
    120.0,
    1.15 * np.max(np.abs(relative_pos))
)

ax.set_xlim(-axis_limit_km, axis_limit_km)
ax.set_ylim(-axis_limit_km, axis_limit_km)
ax.set_zlim(-axis_limit_km, axis_limit_km)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("Relative X [km]")
ax.set_ylabel("Relative Y [km]")
ax.set_zlabel("Relative Z [km]")

ax.set_title("Hitchhiker 1: Motion Relative to 3I/ATLAS")
ax.legend(loc="upper right")

info = ax.text2D(
    0.03, 0.93, "",
    transform=ax.transAxes
)

# ============================================================
# Animation update
# ============================================================

trail_length = 60
start_date = departurte_date + timedelta(seconds=t_rv)

def update(frame):
    x = relative_pos[0, frame]
    y = relative_pos[1, frame]
    z = relative_pos[2, frame]

    probe_marker.set_data([x], [y])
    probe_marker.set_3d_properties([z])

    start = max(0, frame - trail_length)

    trail.set_data(
        relative_pos[0, start:frame + 1],
        relative_pos[1, start:frame + 1]
    )
    trail.set_3d_properties(
        relative_pos[2, start:frame + 1]
    )

    info.set_text(
        f"Elapsed: {start_date} + {((times_comet[frame] - t_rv) / DAY):.2f} days\n"
        f"Separation: {distance_km[frame]:.2f} km"
    )

    return probe_marker, trail, info

ani_comet = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=10,
    blit=False
)

plt.tight_layout()
plt.show()

# Optional: save the animation.
# ani_comet.save(
#     "Hitchhiker1_Comet_Relative_Motion.gif",
#     writer=PillowWriter(fps=30)
# )

# ============================================================
# Earth departure animation
# ============================================================

# Show the first 24 hours after the departure burn.
window_hours = 12
t_end_earth = t_launch + window_hours * 3600

# Corrected spacecraft state at the departure burn.
# y_launch = [x, y, z, vx, vy, vz]
y0_earth = np.asarray(y_launch, dtype=float).copy()

# Propagate the trajectory with dense output.
earth_departure_sol = nbody.propagate(
    y0_earth,
    t_launch,
    t_end_earth,
    dense=True
)

# Dense animation sampling.
n_frames = 1000
times_earth = np.linspace(
    t_launch, t_end_earth, n_frames
)

# Probe positions in the simulation's heliocentric frame.
probe_states_earth = earth_departure_sol.sol(times_earth)
probe_positions_earth = probe_states_earth[:3, :]

# Earth positions at the same times.
earth_positions = np.column_stack([
    earth.get_state(t) for t in times_earth
])

# Earth-centred relative position, in km.
relative_pos_earth = (
    probe_positions_earth - earth_positions
) / 1000.0

# Distance from Earth's centre and altitude above Earth's surface.
distance_earth_km = np.linalg.norm(
    relative_pos_earth, axis=0
)
earth_radius_km = 6371.0
altitude_km = distance_earth_km - earth_radius_km

print(
    f"Initial altitude: {altitude_km[0]:,.1f} km\n"
    f"Final altitude after {window_hours} hours: "
    f"{altitude_km[-1]:,.1f} km"
)

# ============================================================
# Set up 3D figure
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Draw Earth as a sphere centred at the origin.
theta = np.linspace(0, 2 * np.pi, 48)
phi = np.linspace(0, np.pi, 30)

ex = earth_radius_km * np.outer(
    np.cos(theta), np.sin(phi)
)
ey = earth_radius_km * np.outer(
    np.sin(theta), np.sin(phi)
)
ez = earth_radius_km * np.outer(
    np.ones_like(theta), np.cos(phi)
)

ax.plot_surface(
    ex, ey, ez,
    color="royalblue",
    alpha=0.85,
    linewidth=0,
    label = "Earth"
)

# Complete trajectory during the selected window.
ax.plot(
    relative_pos_earth[0],
    relative_pos_earth[1],
    relative_pos_earth[2],
    color="steelblue",
    alpha=0.4,
    linewidth=1,
    label="Departure trajectory"
)

# Animated spacecraft marker and trail.
probe_marker, = ax.plot(
    [], [], [],
    marker="o",
    color="red",
    markersize=6,
    linestyle="None",
    label="Hitchhiker 1"
)

trail, = ax.plot(
    [], [], [],
    color="red",
    linewidth=1.8
)

# Equal scale for all three axes.
axis_limit_km = max(
    1.1 * np.max(np.abs(relative_pos_earth)),
    2.0 * earth_radius_km
)

ax.set_xlim(-axis_limit_km, axis_limit_km)
ax.set_ylim(-axis_limit_km, axis_limit_km)
ax.set_zlim(-axis_limit_km, axis_limit_km)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("Earth-relative X [km]")
ax.set_ylabel("Earth-relative Y [km]")
ax.set_zlabel("Earth-relative Z [km]")
ax.set_title("Hitchhiker 1: Earth Departure")
ax.legend(loc="upper right")

info = ax.text2D(
    0.03, 0.93, "",
    transform=ax.transAxes
)

# ============================================================
# Animate
# ============================================================

trail_length = 100

def update(frame):
    x, y, z = relative_pos_earth[:, frame]

    probe_marker.set_data([x], [y])
    probe_marker.set_3d_properties([z])

    start = max(0, frame - trail_length)

    trail.set_data(
        relative_pos_earth[0, start:frame + 1],
        relative_pos_earth[1, start:frame + 1]
    )
    trail.set_3d_properties(
        relative_pos_earth[2, start:frame + 1]
    )

    elapsed_hours = (times_earth[frame] - t_launch) / 3600

    info.set_text(
        f"Elapsed: {departurte_date + timedelta(seconds=t_launch)} + {elapsed_hours:.2f} hours\n"
        f"Altitude: {altitude_km[frame]:,.0f} km"
    )

    return probe_marker, trail, info

ani_earth = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=20,
    blit=False
)

plt.tight_layout()
plt.show()

# Optional: save the animation as a GIF.
# ani_earth.save(
#     "Hitchhiker1_Earth_Departure.gif",
#     writer=PillowWriter(fps=30)
# )