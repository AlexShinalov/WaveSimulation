import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def gaussian_wave(x: np.ndarray, y: np.ndarray, center: tuple[float, float], width: tuple[float, float]) -> np.ndarray:
    return np.exp(-(x - center[0]) ** 2 / (width[0] ** 2)) * np.exp(-(y - center[1]) ** 2 / (width[1] ** 2))


a = 1.0     # Width of the system
b = 10.0    # Height of the system
dx = 0.02   # Discrete spatial stepsize
c = 1.0     # Wave speed

dt = .707 * dx / c  # CFL condition

x_axis = np.arange(0, a * (1 + dx), dx)
y_axis = np.arange(0, b * (1 + dx), dx)
x, y = np.meshgrid(x_axis, y_axis)

f = np.zeros((len(y_axis), len(x_axis), 3))

f[:, :, 0] = gaussian_wave(x, y, (.5, 0), (.05, .05))

peak_positions = []
crack_positions = [(0.5, 5)]

# First time step in the leap frog algorithm
f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + \
                .5 * c ** 2 * (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2) + \
                .5 * c ** 2 * (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2)


fig, ax = plt.subplots()

img = plt.imshow(f[:,:,2], vmin=-.1, vmax=.1, cmap='seismic')

def update(frame):
    f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] + \
                        c ** 2 * (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2) + \
                        c ** 2 * (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2)

    for crack_pos in crack_positions:
        f[int(crack_pos[1] / dx), int(crack_pos[0] / dx), 2] *= .5

    f[:, :, 0] = f[:, :, 1]
    f[:, :, 1] = f[:, :, 2]

    xc_current = np.argmax(abs(f[:, 2]))
    peak_positions.append(xc_current)

    img.set_array(f[:,:,2])
    return [img]


anim = FuncAnimation(fig, update, frames=1, interval=20)
plt.show()

peak_displacements = [abs(peak_positions[i] - peak_positions[i-1]) for i in range(1, len(peak_positions))]
wave_speeds = [displacement / dt for displacement in peak_displacements]

print("Скорость волны:", wave_speeds)

velocities = [abs(peak_positions[i+1] - peak_positions[i]) / dt for i in range(len(peak_positions) - 1)]
average_velocity = sum(velocities) / len(velocities)
print(average_velocity)
