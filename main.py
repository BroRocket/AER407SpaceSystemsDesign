
from datetime import datetime
import numpy as np

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbit_Mission_Simulation.visualizer import Visualizer

AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s

orbits = Ephemeris()

departurte_date = datetime(2025, 9, 1)
arrival_date = datetime(2026, 12, 31) # 2028, 6, 17)

comet = CelestialBody("3I/ATLAS", 0, orbits, "'DES=1004083'", departurte_date, arrival_date, vec_type=2) # all in m^3/s^2
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

bodies = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune] # [earth, mars, jupiter]

#lambert_v1 = [ 5783.5129679 , 70617.7458126 , -4371.48300469]
lambert_v1 = [-9025.33155895, 59006.35163239, -2759.07709438] # for peri earht
#lambert_v0 = [1723.289 + earth.v[0][0], 39738.198 + earth.v[1][0], -4225.025 + earth.v[2][0]]

def probe_trhust(t):
    trhust_duration = 20000
    if t < 86745600 - trhust_duration/2:
        return np.array([0, 0, 0])
    elif t > 86745600 + trhust_duration/2:
        return np.array([0, 0, 0])
    else: 
        thrust = 100000000 * np.array([ -0.81916498 , -0.57018647 , 0.06209773])
        burn_duration = t - (86745600 - trhust_duration/2)
        mass = 750 - burn_duration * 0.01
        return thrust/mass


probe = Spacecraft("Hitchhiker 1", [earth.r[0][0] + 6371000 + 400000, earth.r[1][0], earth.r[2][0]], lambert_v1)

probe.propogate_orbit(0, (arrival_date-departurte_date).total_seconds(), MU_SUN, celestial_bodies=bodies, t_eval=earth.times_in_seconds)
print("Loaded probe")

# print(probe.t[:6])
# print(earth.times_in_seconds[:6])
# print(len(probe.r[0]))
# print(len(earth.r[0]))
#exit()
vis = Visualizer(probe, bodies + [comet])

#vis.trajectory_2D('x', 'y')
# vis.trajectory_2D('x', 'z')
# vis.trajectory_2D('y', 'z')
#vis.trajectory_3D()
vis.animate_trajectory_2D('x', 'y')
#vis.ani.save("orbit_animation.gif", writer="pillow", fps=30)
vis.animate_trajecctory_3D()

min_d = 100000000000
ind = 0
for i in range(0, len(probe.r[0])):
    d = np.linalg.norm(np.array([comet.r[0][i], comet.r[1][i], comet.r[2][i]]) - np.array([probe.r[0][i], probe.r[1][i], probe.r[2][i]]))
    if d < min_d:
        min_d = d
        ind = i
        relative_v = np.linalg.norm(np.array([comet.v[0][i], comet.v[1][i], comet.v[2][i]]) - np.array([probe.v[0][i], probe.v[1][i], probe.v[2][i]]))


print(f"Rocket Delta V Required from 400km parking oorbit: {np.linalg.norm([5783.2706395 - earth.v[0][0],  70617.52066621 - earth.v[1][0] - 7672.58, -4371.45302152 - earth.v[0][0]])}")
print(f"Minimum Seperation between Comet on closet approach: {min_d} m")
print(f"Relative speed to comet at closet appraoch: {relative_v} m/s")  
print(f"Comet: {[comet.r[0][ind], comet.r[1][ind], comet.r[2][ind]]}m, {[comet.v[0][ind], comet.v[1][ind], comet.v[2][ind]]}m/s\nProbe: {[probe.r[0][ind], probe.r[1][ind], probe.r[2][ind]]}m, {[probe.v[0][ind], probe.v[1][ind], probe.v[2][ind]]}m/s")
print(f"Time at closest appraoch (After Departure Date): {probe.t[ind]} s")
print(f"Relative velocity unit Vector: {(np.array([comet.v[0][ind], comet.v[1][ind], comet.v[2][ind]]) - np.array([probe.v[0][ind], probe.v[1][ind], probe.v[2][ind]]))/relative_v}")