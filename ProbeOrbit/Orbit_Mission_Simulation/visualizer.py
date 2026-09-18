
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ProbeOrbit.Orbit_Mission_Simulation.orbiting_bodies import CelestialBody, Spacecraft

AU = 149597870700  # m

class Visualizer:
    def __init__(self, spacecraft: Spacecraft, bodies: list[CelestialBody]):
        self.spacecraft = spacecraft
        self.bodies = bodies

        self.dim_map = {"x": 0, "y": 1, "z": 2}

    def trajectory_2D(self, dim1: str, dim2: str):

        plt.figure(figsize=(10, 10))
        d1_idx = self.dim_map[dim1.lower()]
        d2_idx = self.dim_map[dim2.lower()]

        plt.xlabel(f"{dim1.upper()} [AU]")
        plt.ylabel(f"{dim2.upper()} [AU]")

        plt.scatter(0, 0, s=20, label="Sun", c="yellow") # plot sun

        for body in self.bodies:
            plt.plot(body.r[d1_idx] / AU, body.r[d2_idx] / AU, label=body.name)

        #spacecraft
        plt.plot(self.spacecraft.r[d1_idx] / AU, self.spacecraft.r[d2_idx] / AU, label=self.spacecraft.name)

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

    def animate_trajectory_2D(self, dim1: str, dim2: str):

        fig, ax = plt.subplots()
        d1_idx = self.dim_map[dim1.lower()]
        d2_idx = self.dim_map[dim2.lower()]

        ax.set_xlabel(f"{dim1.upper()} [AU]")
        ax.set_ylabel(f"{dim2.upper()} [AU]")

        all_objects = self.bodies + [self.spacecraft]

        # 2. Calculate Axis Limits in AU
        dim1_min = min(min(body.r[d1_idx] / AU) for body in all_objects)
        dim1_max = max(max(body.r[d1_idx] / AU) for body in all_objects)
        dim2_min = min(min(body.r[d2_idx] / AU) for body in all_objects)
        dim2_max = max(max(body.r[d2_idx] / AU) for body in all_objects)
        lim = max(abs(dim1_min), abs(dim2_min), abs(dim1_max), abs(dim2_max))

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True)

        plt.scatter(0, 0, s=20, label="Sun", c="yellow") # plot sun

        # 3. Create line objects (trailing paths) and point markers (current positions)
        lines = []
        points = []
        for body in all_objects:
            # Trailing trajectory line
            (line,) = ax.plot([], [], lw=1.5, label=body.name)
            lines.append(line)
            # Leading point marker showing current position
            (point,) = ax.plot([], [], marker="o", markersize=5, color=line.get_color())
            points.append(point)

        ax.legend(loc="upper right")

        # 4. Initialization Function
        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return lines + points

        # 5. Frame Update Function
        def update(frame):
            for line, point, body in zip(lines, points, all_objects):
                # Slice position data up to the current frame and scale to AU
                dim1_data = body.r[d1_idx][:(frame*100)] / AU
                dim2_data = body.r[d2_idx][:(frame*100)] / AU

                line.set_data(dim1_data, dim2_data)
                
                # Place the point marker at the latest coordinate
                if len(dim1_data) > 0:
                    point.set_data([dim1_data[-1]], [dim2_data[-1]])
                    
            return lines + points

        # 6. Create Animation
        num_frames = int(len(self.spacecraft.r[0])/100)  
        
        # Store animation reference in 'self' to prevent Python garbage collection
        self.ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=5, blit=True)
        plt.show()

    def animate_trajecctory_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_zlabel("Z [AU]")
        
        all_objects = self.bodies + [self.spacecraft]

        # 2. Calculate Axis Limits in AU
        x_min = min(min(body.r[0] / AU) for body in all_objects)
        x_max = max(max(body.r[0] / AU) for body in all_objects)
        y_min = min(min(body.r[1] / AU) for body in all_objects)
        y_max = max(max(body.r[1] / AU) for body in all_objects)
        z_min = min(min(body.r[2] / AU) for body in all_objects)
        z_max = max(max(body.r[2] / AU) for body in all_objects)
        lim = max(abs(x_min), abs(y_min), abs(z_min), x_max, y_max, z_max)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.grid(True)

        plt.scatter(0, 0, s=20, label="Sun", c="yellow") # plot sun

        #Create line objects (trailing paths) and point markers (current positions)
        lines = []
        points = []
        for body in all_objects:
            # Trailing trajectory line
            (line,) = ax.plot([], [], [], lw=1.5, label=body.name)
            lines.append(line)
            # Leading point marker showing current position
            (point,) = ax.plot([], [], [], marker="o", markersize=5, color=line.get_color())
            points.append(point)

        ax.legend(loc="upper right")

        #Initialization Function
        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])

                point.set_data([], [])
                point.set_3d_properties([])
            return lines + points

        # Frame Update Function
        def update(frame):
            for line, point, body in zip(lines, points, all_objects):
                # Slice position data up to the current frame and scale to AU
                x_data = body.r[0][:(frame*100)] / AU
                y_data = body.r[1][:(frame*100)] / AU
                z_data = body.r[2][:(frame*100)] / AU

                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)

                # Update current position marker
                if len(x_data) > 0:
                    point.set_data([x_data[-1]], [y_data[-1]])
                    point.set_3d_properties([z_data[-1]])
                    
            return lines + points

        # 6. Create Animation
        num_frames = len(self.spacecraft.r[0]/100)
        
        # Store animation reference in 'self' to prevent Python garbage collection
        self.ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=5, blit=True)
        plt.show()