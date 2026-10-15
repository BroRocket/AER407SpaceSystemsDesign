
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbit_Mission_Simulation.visualizer import Visualizer
# New helper (same folder as this file, or adjust the import path to where you keep it)
from targeting import NBody, leo_departure_state, correct_velocity, jupiter_approach_target

AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s
DAY = 86400.0 #24 * 60 * 60

orbits = Ephemeris()

departurte_date = datetime(2025, 11, 18) #12025-10-02T09:20:36
arrival_date = datetime(2028, 12, 30) # end of simulation: ~100 days after the 2028-03-21 rendezvous

# departurte_date = datetime(2028, 6, 17) #1
# arrival_date = datetime(2028, 6, 30)

comet = CelestialBody("3I/ATLAS", 2202.52, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
print("Loaded Comet")
mercury = CelestialBody("Mercury", 2.2031870799e13, orbits, "'199'", departurte_date, arrival_date, vec_type=2)
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

bodies = [comet, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune] # [earth, mars, jupiter]

# ---------------------------------------------------------------------------
# November route from the patched-conic search (J2000 ecliptic, TDB).
# Times are seconds after departurte_date (00:00 TDB on 2025-11-18; Horizons
# vector tables use TDB by default).
# ---------------------------------------------------------------------------
t_launch = 0*3600 + 40*60 + 34.69        # 2025-11-18 00:40:34.69 TDB  (NOT midnight)
t_jupiter = (2461115.6153743 - 2460997.5)*DAY      # 2026-03-16 02:46:08 TDB
t_arrival = (2461851.975104989 - 2460997.5)*DAY    # 2028-03-21 11:24:09 TDB
v_inf_departure = np.array([-14066.741427851426, 50440.26855549723, 467.327219836863])  # m/s, Earth-relative
# Patched-conic heliocentric velocity arriving at / leaving Jupiter, and the flyby periapsis radius
v_into_jupiter = np.array([-40155.2401425902, 54537.06353236099, 424.9650195957534])
v_out_of_jupiter = np.array([-5082.775612493418, 60341.21433570397, -4262.7210616715265])
jupiter_periapsis_radius = 78132.45348615931e3   # m (6,640 km above the 1-bar level)

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
y_launch, _ = correct_velocity(nbody, np.r_[r0, v0], t_launch, t_aim, aim_point, tol_m=1000.)
print(f"Corrected post-burn state (m, m/s): {y_launch.tolist()}")
print(f"Launch velocity correction: {np.linalg.norm(y_launch[3:] - v0):.3f} m/s")
print(f"Initial posiiton: {y_launch[:3]/1000}")
print(f"Earth initila position: {earth.get_state(t_launch)/1000}")
print(f"Orbit Altitude: {np.linalg.norm(earth.get_state(t_launch)/1000 - y_launch[:3]/1000) - 6371}")

# 3) Leg 1: launch -> 10 days after the flyby, with your Spacecraft class.
t_final = (arrival_date-departurte_date).total_seconds()
t_tcm = t_jupiter + 10*DAY
grid = np.array(earth.times_in_seconds)
leg1 = Spacecraft("Hitchhiker 1", y_launch[:3], y_launch[3:])
leg1.propogate_orbit(t_launch, t_tcm, MU_SUN, celestial_bodies=bodies,
                     t_eval=np.r_[grid[(grid >= t_launch) & (grid < t_tcm)], t_tcm])  # t_eval only picks saved points

# 4) Trajectory-correction burn 10 days after Jupiter so the probe meets the comet on the
#    arrival date. Small flyby errors are amplified enormously (1,000 km at periapsis ->
#    ~0.35 km/s outgoing), so real missions always plan a correction here.
y_tcm = np.r_[leg1.r[:, -1], leg1.v[:, -1]]
print("Solving the post-flyby correction burn...")
# Aim at a point STANDOFF metres from the comet's centre on its sunward side (clear of the dust tail),
# not at the centre itself: the nucleus is up to a few km across.
standoff = 100e3   # m
r_comet_arr = comet.get_state(t_arrival)
standoff_dir = -r_comet_arr/np.linalg.norm(r_comet_arr)
y_after, _ = correct_velocity(nbody, y_tcm, t_tcm, t_arrival, r_comet_arr + standoff*standoff_dir, tol_m=1000.)
tcm = y_after[3:] - y_tcm[3:]
print(f"Correction burn at {t_tcm/DAY:.1f} days: {np.linalg.norm(tcm):.3f} m/s, vector {tcm.tolist()} m/s")

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
isp = 320.0   # s, a typical bipropellant engine; change to suit
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

df_distance = pd.DataFrame({"Time (s)": probe.t, "Time (Years)": probe.t/(60*60*24*365.25), "Distance From Sun [AU]": distance_from_sun})

df_distance.to_csv("Distance From Sun Over Time AU.csv", index=False)

vis = Visualizer(probe, bodies)

vis.trajectory_2D('x', 'y')
# vis.trajectory_2D('x', 'z')
# vis.trajectory_2D('y', 'z')
#vis.trajectory_3D()
vis.animate_trajectory_2D('x', 'y')
#vis.ani.save("whole_trajectory_animation.gif", writer="pillow", fps=30)
vis.animate_trajecctory_3D()

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

print(f"Comet at rendezvous: {comet.get_state(t_rv).tolist()} m, {v_comet_rv.tolist()} m/s")
print(f"Probe after burn: {leg2.r[:, -1].tolist()} m, {v_target.tolist()} m/s")
print(f"Probe before burn: {leg2.r[:, -1].tolist()} m, {v_before_rv.tolist()} m/s")
print(f"Time of rendezvous (After Departure Date): {t_rv} s")
print(f"Total delta-v: LEO burn {burn:.1f} + launch correction {np.linalg.norm(y_launch[3:] - v0):.1f} "
      f"+ post-flyby correction {np.linalg.norm(tcm):.1f} + rendezvous {np.linalg.norm(dv_rv):.1f} m/s")


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
        f"Date: {datetime(2025, 11, 18) + timedelta(seconds = t_start + relative_times[frame])}\n"
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
window_days = 160
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
start_date = datetime(2025, 11, 18) + timedelta(seconds=t_rv)

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
        f"Elapsed: {datetime(2025, 11, 18, 0, 40, 34)} + {elapsed_hours:.2f} hours\n"
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