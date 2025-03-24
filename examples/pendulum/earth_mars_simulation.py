import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from selfmod import *

class PendulumSimulation:
    def __init__(self, length=1.0, gravity=9.81, initial_angle=np.pi/4, damping=0.0):
        """
        Initialize pendulum simulation parameters
        
        Args:
        - length: Length of pendulum (m)
        - gravity: Gravitational acceleration (m/s^2)
        - initial_angle: Starting angle from vertical (radians)
        - damping: Damping coefficient to simulate friction
        """
        self.length = length
        self.gravity = gravity
        self.angle = initial_angle
        self.angular_velocity = 0
        self.damping = damping
        
        # Simulation parameters
        self.dt = 0.02  # Time step
        self.time = 0
    
    def update(self):
        """
        Update pendulum state using simple angular acceleration equation
        """
        # Angular acceleration
        angular_acceleration = (-self.gravity / self.length) * np.sin(self.angle) - self.damping * self.angular_velocity
        
        # Update angular velocity and angle
        self.angular_velocity += angular_acceleration * self.dt
        self.angle += self.angular_velocity * self.dt
        
        self.time += self.dt
        
        return self.angle
    
    def simulate(self, total_time=10):
        """
        Simulate pendulum motion
        
        Args:
        - total_time: Total simulation time (s)
        
        Returns:
        - angles: Array of angles over time
        - times: Array of corresponding times
        """
        steps = int(total_time / self.dt)
        angles = np.zeros(steps)
        times = np.zeros(steps)
        
        for i in range(steps):
            angles[i] = self.angle
            times[i] = self.time
            self.update()
        
        return angles, times
    
    def create_animation(self, filename='pendulum.gif', total_time=10):
        """
        Create and save pendulum animation as GIF
        
        Args:
        - filename: Output GIF filename
        - total_time: Total simulation time (s)
        """
        # Reset the simulation to initial state
        self.angle = np.pi/4
        self.angular_velocity = 0
        self.time = 0
        
        angles, times = self.simulate(total_time)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot ceiling
        ax.add_patch(plt.Rectangle((-0.25, 1), 0.5, 0.1, fc='gray'))
        
        # Pendulum line and bob
        line, = ax.plot([], [], 'k-', linewidth=2)
        bob, = ax.plot([], [], 'ro', markersize=20)  # Increased bob size
        
        # Title with time
        # title = ax.text(0.5*(0.25*2 + 0.5), 1.6, '', fontsize=12)
        title = ax.text(-0.44, 1.6, '', fontsize=18)

        def init():
            line.set_data([], [])
            bob.set_data([], [])
            title.set_text('')
            return line, bob, title
        
        def animate(i):
            # Ensure we don't go out of bounds
            if i >= len(angles):
                i = len(angles) - 1

            angle = angles[i]
            x = self.length * np.sin(angle)
            y = 1 - self.length * np.cos(angle)

            line.set_data([0, x], [1, y])
            bob.set_data(x, y)
            title.set_text(f'Mars: $t=${times[i]:.2f} s')

            return line, bob, title

        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(angles), interval=20, blit=True)

        # Save as GIF
        anim.save(filename, writer='mp4', fps=50)
        plt.close(fig)

# Example usage
# sim = PendulumSimulation(gravity=9.81, length=1.0, initial_angle=np.pi/3)
# sim.create_animation('earth_pendulum.gif')

sim = PendulumSimulation(gravity=3.73, length=1.0, initial_angle=np.pi/3)
sim.create_animation('mars_pendulum.gif')


print("Pendulum simulation complete. GIF saved as 'earth_pendulum.gif'")