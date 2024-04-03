import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def gaussian_wave(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-(x - center) ** 2 / (width ** 2))


def standing_wave(x: np.ndarray, width: float) -> np.ndarray:
    return np.sin(2 * np.pi * x / width)


a = 1.0  # Length of the system
dx = 0.01  # Discrete spatial stepsize
c = 1.0  # Wave speed

dt = dx / c  # CFL condition

x = np.arange(0, a * (1 + dx), dx)
f = np.zeros((len(x), 3))

nsteps = 1000

peak_positions = []
crack_positions = [0.3, 0.6, 0.1, 0.5, 0.9]

# Initial condition
f[:, 0] = gaussian_wave(x, 0.1, 0.05)
# f[:, 0] = standing_wave(x, a)


# First time step in the leap frog algorithm
f[1:-1, 1] = f[1:-1, 0] + \
             0.5 * c ** 2 * (f[:-2, 0] + f[2:, 0] - 2 * f[1:-1, 0]) * (dt ** 2 / dx ** 2)


fig, ax = plt.subplots()
ax.set(ylim=[-1, 1])
graph = ax.plot(x, f[:, 2], 'b')[0]


def update(frame: int):
    f[1:-1, 2] = -f[1:-1, 0] + \
        2 * f[1:-1, 1] + \
        c**2 * (f[:-2, 1] + f[2:, 1] - 2 * f[1:-1, 1]) * (dt**2 / dx**2)

    for crack_pos in crack_positions:
        f[int(crack_pos / dx), 2] *= 0.5

    # Push the data back for the leapfrogging
    f[:, 0] = f[:, 1]
    f[:, 1] = f[:, 2]

    # Enforce the boundary conditions
    f[0, :] = 0
    f[-1, :] = 0

    graph.set_ydata(f[:, 2])
    return graph

animation = FuncAnimation(fig=fig, func=update, frames=nsteps, interval=20, repeat=False)
plt.show()


for k in range(0, nsteps):
    f[1:-1, 2] = -f[1:-1, 0] + \
                 2 * f[1:-1, 1] + \
                 c ** 2 * (f[:-2, 1] + f[2:, 1] - 2 * f[1:-1, 1]) * (dt ** 2 / dx ** 2)

    for crack_pos in crack_positions:
        f[int(crack_pos / dx), 2] *= 0.01

    # Push the data back for the leapfrogging
    f[:, 0] = f[:, 1]
    f[:, 1] = f[:, 2]

    # Enforce the boundary conditions
    f[0, :] = 0
    f[-1, :] = 0

    xc_current = np.argmax(abs(f[:, 2]))
    peak_positions.append(xc_current)

velocities = [(peak_positions[i + 1] - peak_positions[i]) / dt for i in range(nsteps - 1)]
average_velocity = sum(velocities) / len(velocities)

print(peak_positions)
print("Средняя скорость волны:", average_velocity)



