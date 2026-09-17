
import matplotlib.pyplot as plt
from datetime import datetime

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft
from ProbeOrbit.Orbital_Mission_Planning.ephemeris import Ephemeris
from ProbeOrbit.Orbit_Mission_Simulation.visualizer import Visualizer

AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s

orbits = Ephemeris()

departurte_date = datetime(2025, 9, 17)
arrival_date = datetime(2027, 10, 20) # 2028, 6, 17)

comet = CelestialBody("3I/ATLAS", 0, orbits, "'DES=1004083'", departurte_date, arrival_date)
print("Loaded Comet")
earth = CelestialBody("Earth", 3.986004418e14, orbits, "'399'", departurte_date, arrival_date, vec_type=2)
print("loaded Earth")
mars = CelestialBody("Mars", 4.282837e13, orbits, "'499'", departurte_date, arrival_date, vec_type=2)
print("loaded Mars")
jupiter = CelestialBody("Jupiter", 1.26686534e17, orbits, "'599'", departurte_date, arrival_date, vec_type=2)
print("loaded Jupiter")
probe = Spacecraft("Hitchhiker 1", [earth.r[0][0] + 6371000 + 500000, earth.r[1][0], earth.r[2][0]], [1723.289 + earth.v[0][0], 39738.198 + earth.v[1][0], -4225.025 + earth.v[2][0]])

probe.propogate_orbit(0, (arrival_date-departurte_date).total_seconds(), MU_SUN, celestial_bodies=[earth, mars, jupiter], t_eval=earth.times_in_seconds)
print("loaded probe")

#print(probe.t[:6])
#print(earth.times_in_seconds[:6])
#print(len(probe.r[0]))
#print(len(earth.r[0]))
#exit()
vis = Visualizer(probe, [earth, mars, jupiter, comet])

vis.trajectory_2D('x', 'y')
# vis.trajectory_2D('x', 'z')
# vis.trajectory_2D('y', 'z')
#vis.trajectory_3D()
vis.animate_trajectory_2D('x', 'y')
#vis.animate_trajectory_2D('y', 'z')
#vis.ani.save("orbit_animation.gif", writer="pillow", fps=30)