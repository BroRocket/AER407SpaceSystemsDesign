
import matplotlib.pyplot as plt
from datetime import datetime

from orbiting_bodies import CelestialBody, Spacecraft
from Orbital_Mission_Planning.ephemeris import Ephemeris


AU = 149597870700  # m
MU_SUN = 1.32712440041279419e20 #m/s

orbits = Ephemeris()

departurte_date = datetime(2025, 9, 17)
arrival_date = datetime(2028, 6, 17)

comet = CelestialBody("3I/ATLAS", 0, orbits, "'DES=1004083'", departurte_date, arrival_date)

earth = CelestialBody("Earth", 3.986004418e14, "'399'", departurte_date, arrival_date)

probe = Spacecraft("Hitchhiker 1", earth.r[0], [1723.289, 39738.198, -4225.025])
probe.propogate_orbit(0, (arrival_date-departurte_date).total_seconds(), MU_SUN)


plt.figure(figsize=(10, 10))

# Sun
plt.scatter(0, 0, s=200, label="Sun")

# Earth
earth_x = []
earth_y = []
for r in earth.r:
    earth_x.append(r[0] / AU)
    earth_y.append(r[1] / AU)

plt.plot(earth_x, earth_y, label="Earth")

#comet
comet_x = []
comet_y = []
for r in comet.r:
    comet_x.append(r[0] / AU)
    comet_y.append(r[1] / AU)

plt.plot(earth_x, earth_y, label="3I/ATLAS")

#spacecraft
plt.plot(probe.r[0], probe.r[1], label=probe.name)

plt.xlabel("x [AU]")
plt.ylabel("y [AU]")

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()