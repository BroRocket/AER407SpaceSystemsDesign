
import numpy as np
import matplotlib.pyplot as plt

N = 100

diameter = 1.129/1000
Ath = (np.pi * diameter**2) / 4

expected_vp = 101325 * 0.9
expected_rho = 1141
expected_p1 = 2250000

vps = np.linspace(expected_vp*0.8, expected_vp*1.2, N)
rhos = np.linspace(expected_rho*0.8, expected_rho*1.2, N)
p1s = np.linspace(expected_p1*0.8, expected_p1*1.2, N)

m_dot_vp = Ath * np.sqrt(2*expected_rho*(expected_p1 - vps))
m_dot_rho = Ath * np.sqrt(2*rhos*(expected_p1 - expected_vp))
m_dot_p1 = Ath * np.sqrt(2*expected_rho*(p1s - expected_vp))

desired_mdot = np.full(N, 0.0713)

plt.plot(vps, m_dot_vp)
plt.plot(vps, desired_mdot, label = "M_dot Target", linestyle="-")
plt.title("Kerosene: Variation of Mass Flow Rate with Vapour Pressure") 
plt.xlabel("Vapour Pressure (Pa)")
plt.ylabel("m_dot (kg/s)")
plt.show()

plt.plot(rhos, m_dot_rho)
plt.plot(rhos, desired_mdot, label = "M_dot Target", linestyle="-")
plt.title("Kerosene: Variation of Mass Flow Rate with Density") 
plt.xlabel("Density (kg/m^3)")
plt.ylabel("m_dot (kg/s)")
plt.show()

plt.plot(p1s, m_dot_p1)
plt.plot(p1s, desired_mdot, label = "M_dot Target", linestyle="-")
plt.title("Kerosene: Variation of Mass Flow Rate with Upstream Pressure") 
plt.xlabel("Upstream Pressure (Pa)")
plt.ylabel("m_dot (kg/s)")
plt.show()

