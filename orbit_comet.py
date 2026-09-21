import numpy as np
import matplotlib.pyplot as plt

####Data Sources####
# Hubble Space Telescope Observations of the Interstellar Interloper 3I/ATLAS
# David Jewitt, Man-To Hui, Max Mutchler, Yoonyoung Kim, and Jessica Agarwal

# Published 2025 August 21 • © 2025. The Author(s). Published by the American Astronomical Society.
# The Astrophysical Journal Letters, Volume 990, Number 1

#Upper Limit on the Non-Gravitational Acceleration and Lower Limits on the Nucleus Mass and Diameter of 3I/ATLAS Richard Cloete ,1 Abraham Loeb ,1 and Peter Vereˇ s 2

G = 6.67430e-11

r_comet_lower = 0.22 * 1000
r_comet_upper = 2.8 *1000 

m_comet_lower = 4.4e10
m_comet_upper = 3.3e13 * 1.1 # could exceed

orbit_radius_lower = 500
orbit_radius_upper = 100000

N_RADIUS = 60
N_ORBIT = 100

r_comet = np.linspace(
    r_comet_lower,
    r_comet_upper,
    N_RADIUS
)

orbit_radius = np.geomspace(
    orbit_radius_lower,
    orbit_radius_upper,
    N_ORBIT
)

# Create 2D mesh
R_COMET, R_ORBIT = np.meshgrid(
    r_comet,
    orbit_radius
)

mass_values = np.geomspace(
    m_comet_lower,
    m_comet_upper,
    2
)

fig = plt.figure(figsize=(18, 12))

for i, m_comet in enumerate(mass_values):

    ax = fig.add_subplot(
        1, 2, i + 1,
        projection="3d"
    )

    # Circular orbital velocity [m/s]
    V_ORBIT = np.sqrt(
        G * m_comet / R_ORBIT
    )

    # Exclude trajectories passing through the comet
    V_ORBIT[R_ORBIT <= R_COMET] = np.nan

    surface = ax.plot_surface(
        R_COMET / 1000,       # Comet radius [km]
        R_ORBIT / 1000,       # Orbit radius [km]
        V_ORBIT,              # Velocity [m/s]
        cmap="viridis",
        edgecolor="none",
        antialiased=True
    )

    ax.set_title(
        f"Comet mass = {m_comet:.2e} kg"
    )

    ax.set_xlabel("Comet Radius [km]")
    ax.set_ylabel("Orbit Radius [km]")
    ax.set_zlabel("Circular Velocity [m/s]")

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.6,
        pad=0.1,
        label="Velocity [m/s]"
    )

plt.tight_layout()
plt.show()