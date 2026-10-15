
from datetime import datetime
import numpy as np
import pandas as pd

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbit_Mission_Simulation.visualizer import Visualizer
# New helper (same folder as this file, or adjust the import path to where you keep it)
from targeting import NBody, leo_departure_state, correct_velocity, jupiter_approach_target

AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s
DAY = 86400.0

orbits = Ephemeris()

departurte_date = datetime(2025, 11, 18) #12025-10-02T09:20:36
arrival_date = datetime(2028, 4, 20) # 2028, 6, 17) # peri_earth 2026, 12, 31

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
print(f"Rocket Delta V Required from 400km parking orbit: {burn:.1f} m/s\n Initial Velocity {v0}")

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
y_after, _ = correct_velocity(nbody, y_tcm, t_tcm, t_arrival, comet.get_state(t_arrival), tol_m=1000.)
tcm = y_after[3:] - y_tcm[3:]
print(f"Correction burn at {t_tcm/DAY:.1f} days: {np.linalg.norm(tcm):.3f} m/s, vector {tcm.tolist()} m/s")

leg2 = Spacecraft("Hitchhiker 1", y_after[:3], y_after[3:])
leg2.propogate_orbit(t_tcm, t_final, MU_SUN, celestial_bodies=bodies,
                     t_eval=np.r_[t_tcm, grid[(grid > t_tcm) & (grid <= t_final)]])

# Join the two legs into one Spacecraft object for the plots below
probe = Spacecraft("Hitchhiker 1", y_launch[:3], y_launch[3:])
probe.t = np.r_[leg1.t, leg2.t[1:]]
probe.r = np.hstack([leg1.r, leg2.r[:, 1:]])
probe.v = np.hstack([leg1.v, leg2.v[:, 1:]])
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
#vis.ani.save("orbit_animation.gif", writer="pillow", fps=30)
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

cc, sol = closest(comet)
y = sol.sol(cc['t'])
relative = comet.get_velocity(cc['t']) - y[3:]
relative_v = np.linalg.norm(relative)
print(f"Minimum Seperation between Comet on closet approach: {cc['distance_m']:.0f} m")
print(f"Relative speed to comet at closet appraoch: {relative_v} m/s")
print(f"Comet: {comet.get_state(cc['t']).tolist()}m, {comet.get_velocity(cc['t']).tolist()}m/s\nProbe: {y[:3].tolist()}m, {y[3:].tolist()}m/s")
print(f"Time at closest appraoch (After Departure Date): {cc['t']} s")
print(f"Relative velocity unit Vector: {relative/relative_v}")
