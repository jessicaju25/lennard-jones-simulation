import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Lennard-Jones potential parameters
EPSILON = 1.0  # Depth of the potential well
SIGMA = 1.0    # Distance at which the potential is zero
DT = 0.01      # Time step
NUM_PARTICLES = 10  # Number of molecules
BOX_SIZE = 10.0  # Simulation box size
STEPS = 200     # Number of simulation steps

# Initialize positions randomly within the box
positions = np.random.rand(NUM_PARTICLES, 2) * BOX_SIZE

# Initialize velocities randomly
velocities = (np.random.rand(NUM_PARTICLES, 2) - 0.5) * 2

def lennard_jones_force(r):
    """Computes the Lennard-Jones force for a given distance r."""
    if r == 0:
        return 0
    r6 = (SIGMA / r) ** 6
    r12 = r6 ** 2
    force = 24 * EPSILON * (2 * r12 - r6) / (r * r)
    return force

def compute_forces(positions):
    """Computes the net force on each particle."""
    forces = np.zeros_like(positions)
    for i in range(NUM_PARTICLES):
        for j in range(i + 1, NUM_PARTICLES):
            displacement = positions[i] - positions[j]
            distance = np.linalg.norm(displacement)
            if distance < 2.5 * SIGMA:  # Cutoff distance to save computation
                force_mag = lennard_jones_force(distance)
                force_vector = force_mag * (displacement / distance)
                forces[i] += force_vector
                forces[j] -= force_vector
    return forces

def update_positions(positions, velocities, forces, dt):
    """Uses Verlet integration to update particle positions."""
    new_positions = positions + velocities * dt + 0.5 * forces * dt ** 2
    new_positions = new_positions % BOX_SIZE  # Apply periodic boundary conditions
    return new_positions

def update_velocities(velocities, forces, new_forces, dt):
    """Uses Verlet integration to update particle velocities."""
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt
    return new_velocities

# Initialize figure for visualization
fig, ax = plt.subplots()
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
particles, = ax.plot([], [], 'bo', markersize=5)

def animate(frame):
    global positions, velocities

    forces = compute_forces(positions)
    new_positions = update_positions(positions, velocities, forces, DT)
    new_forces = compute_forces(new_positions)
    new_velocities = update_velocities(velocities, forces, new_forces, DT)

    positions[:] = new_positions
    velocities[:] = new_velocities

    particles.set_data(positions[:, 0], positions[:, 1])
    return particles,

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=50)
plt.show()
