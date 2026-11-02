Thermal Control/thermal.py

import math

length = 3.0
width = 3.0
height = 2.1

temp_min = -20
temp_max = 50
margin = 11

sigma = 5.67e-8
q_sol = 1353

alpha_s = 0.85
epsilon_outer = 0.85
epsilon_MLI_eff = 0.02
epsilon_radiator = 0.85

power_launch = 100
decay = 0.017

pipe_capacity = 5080
distance_rtg = 1

def to_kelvin(t_c):
  return t_c + 273.15

def to_celsius(t_k):
  return t_k - 273.15

temp_hot = to_kelvin(temp_max - margin)
temp_cold = to_kelvin(temp_min + margin)

area_total = 2 * (length * width + length * height + width * height)
area_face = max(length*width, length*height, width*height)

print("HITCHHIKER1 THERMAL HAND CALCULATIONS")
print(f"Bus {length} x {width} x {height} m: total area {area_total:.2f} m^2, largest face {area_face:.2f} m^2")
print(f"Design limits: {to_celsius(temp_cold):.0f} C to {to_celsius(temp_hot):.0f} C "
      f"({temp_cold:.2f} K to {temp_hot:.2f} K)\n")

temp_outer = (alpha_s*q_sol/ (epsilon_outer*sigma)) ** 0.25
print("Sunlit blanket temperature")
print(f"        T_o = [{alpha_s} x {q_sol:.0f} / ({epsilon_outer} x {sigma})]^(1/4)"
      f" = {temp_outer:.1f} K ({to_celsius(temp_outer):.1f} C)\n")

q_sun = epsilon_MLI_eff * sigma * (temp_outer ** 4 - temp_hot ** 4)
Q_sun = q_sun * area_face
print("Sunlight leaking through the blanket")
print(f"        q_sun = {epsilon_MLI_eff} x {sigma} x ({temp_outer:.1f}^4 - {temp_hot:.2f}^4) = {q_sun:.2f} W/m^2")
print(f"        Q_sun = {q_sun:.2f} x {area_face:.2f} = {Q_sun:.1f} W\n")

Q_hot = power_launch + Q_sun
print("Hot-case heat load")
print(f"        Q_hot = P_max + Q_sun = {power_launch:.0f} + {Q_sun:.1f} = {Q_hot:.1f} W\n")

area_rad = Q_hot / (epsilon_radiator * sigma * (temp_hot ** 4))
print("Radiator area")
print(f"        A_rad = {Q_hot:.1f} / ({epsilon_radiator} x {sigma} x {temp_hot:.2f}^4) = {area_rad:.3f} m^2\n")

area_MLI = area_total - area_rad
E_cold = sigma * temp_cold ** 4
Q_mli = epsilon_MLI_eff * E_cold * area_MLI
Q_radiator = epsilon_radiator * E_cold * area_rad
Q_out = Q_mli + Q_radiator
print("Heat lost at the cold design limit")
print(f"        sigma T^4 = {sigma} x {temp_cold:.2f}^4 = {E_cold:.1f} W/m^2,  A_MLI = {area_total:.2f} - {area_rad:.3f} = {area_MLI:.2f} m^2")
print(f"        Q_out = {epsilon_MLI_eff} x {E_cold:.1f} x {area_MLI:.2f} + {epsilon_radiator} x {E_cold:.1f} x {area_rad:.3f}"
      f" = {Q_mli:.1f} + {Q_radiator:.1f} = {Q_out:.1f} W\n")

power_electric = power_launch * (1 - decay)**25
Q_RTG = Q_out - power_electric

print(f"RTG heat needed after 25 years")
print(f"        P(25) = {power_launch:.0f} x {1 - decay:.3f}^{25} = {power_electric:.1f} W")
print(f"        Q_RTG = {Q_out:.1f} - {power_electric:.1f} = {Q_RTG:.1f} W\n")

L_cm = distance_rtg * 100
per_pipe = pipe_capacity / L_cm
N = Q_RTG * L_cm / pipe_capacity
N_pipes = math.ceil(N)
print("Heat pipes")
print(f"        One pipe over {distance_rtg} m carries {pipe_capacity:.0f} / {L_cm:.0f} = {per_pipe:.1f} W")
print(f"        N = {Q_RTG:.1f} x {L_cm:.0f} / {pipe_capacity:.0f} = {N:.2f} -> {N_pipes} pipes"
      f" ({N_pipes * per_pipe:.0f} W)")
