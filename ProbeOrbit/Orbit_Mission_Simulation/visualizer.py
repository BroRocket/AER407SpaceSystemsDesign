
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft

AU = 149597870700  # m

class Visualizer:
    def __init__(self, spacecraft: Spacecraft, bodies: list[CelestialBody]):
        self.spacecraft = spacecraft
        self.bodies = bodies

    def trajectory_2D(self, dim1: str, dim2: str):

        plt.figure(figsize=(10, 10))
        plt.xlabel(f"{dim1} [AU]")
        plt.ylabel(f"{dim2} [AU]")

        if dim1.lower() == "x":
            dim1 = 0
        elif dim1.lower() == "y":
            dim1 = 1
        elif dim1.lower() == "z":
            dim1 = 2

        if dim2.lower() == "x":
            dim2 = 0
        elif dim2.lower() == "y":
            dim2 = 1
        elif dim2.lower() == "z":
            dim2 = 2

        plt.scatter(0, 0, s=20, label="Sun", c="yellow") # plot sun

        for body in self.bodies:
            plt.plot(body.r[dim1] / AU, body.r[dim2] / AU, label=body.name)

        #spacecraft
        plt.plot(self.spacecraft.r[dim1] / AU, self.spacecraft.r[dim2] / AU, label=self.spacecraft.name)

        plt.axis("equal")
        plt.grid()
        plt.legend()

        plt.show()

    def trajectory_3D(self, aspect_equal = False):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(0, 0, 0, s=20, label="Sun", c="yellow")

        for body in self.bodies:
            ax.plot(body.r[0] / AU, body.r[1] / AU, body.r[2] / AU, label=body.name)

        ax.plot(self.spacecraft.r[0] / AU, self.spacecraft.r[1] / AU, self.spacecraft.r[2] / AU, label=self.spacecraft.name)

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_zlabel("Z [AU]")

        if aspect_equal is True:
            ax.set_aspect('equal')

        plt.legend()
        plt.show()

    def animate_trajectory_2D(self):
        pass

    def animate_trajecctory_3D(self):
        pass