import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


def animate_wave(a, b, dx, c, crack_positions):
    dt = .707 * dx / c

    x_axis = np.arange(0, a * (1 + dx), dx)
    y_axis = np.arange(0, b * (1 + dx), dx)
    x, y = np.meshgrid(x_axis, y_axis)

    f = np.zeros((len(y_axis), len(x_axis), 3))

    def gaussian_wave(x: np.ndarray, y: np.ndarray, center: tuple[float, float],
                      width: tuple[float, float]) -> np.ndarray:
        return np.exp(-(x - center[0]) ** 2 / (width[0] ** 2)) * np.exp(-(y - center[1]) ** 2 / (width[1] ** 2))

    f[:, :, 0] = gaussian_wave(x, y, (0, .5), (.05, .05))

    wave_speeds = []
    # crack_positions = [[np.random.rand() * a, np.random.rand() * b] for _ in range(100)]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    img = ax1.imshow(f[:, :, 2], vmin=-.1, vmax=.1, cmap='seismic')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    graph, = ax2.plot([], [])
    ax2.set_ylim(0, .01)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Wave speed')

    time_limit = 20.0
    current_time = 0.0

    def update(frame):
        nonlocal f, wave_speeds, time_limit, current_time

        f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] + \
                           c ** 2 * (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2) + \
                           c ** 2 * (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2)

        speed = np.average(np.abs(f[:, :, 2] - f[:, :, 0]) * dt / dx)
        wave_speeds.append(speed)

        for crack_pos in crack_positions:
            f[int(crack_pos[1] / dx), int(crack_pos[0] / dx), 2] *= .5

        f[:, :, 0] = f[:, :, 1]
        f[:, :, 1] = f[:, :, 2]

        img.set_array(f[:, :, 2])

        if current_time >= time_limit:
            time_limit *= 2
            ax2.set_xlim(0, current_time)

        graph.set_data(range(len(wave_speeds)), wave_speeds)
        ax2.set_xlim(0, len(wave_speeds))

        return [img, graph]

    anim = FuncAnimation(fig, update, frames=1, interval=20)
    plt.show()


# Создание GUI
def run_simulation():
    a_val = float(a_entry.get())
    b_val = float(b_entry.get())
    dx_val = float(dx_entry.get())
    c_val = float(c_entry.get())
    animate_wave(a_val, b_val, dx_val, c_val, crack_positions)



root = tk.Tk()
root.title("Wave Simulation")

tk.Label(root, text="Width of the system:").grid(row=0, column=0)
a_entry = tk.Entry(root)
a_entry.grid(row=0, column=1)
a_entry.insert(0, "10.0")
a_entry.bind('<KeyRelease>', lambda _: canvas.config(width=float(a_entry.get() or '1') * CANVAS_SALE))

tk.Label(root, text="Height of the system (b):").grid(row=1, column=0)
b_entry = tk.Entry(root)
b_entry.grid(row=1, column=1)
b_entry.insert(0, "1.0")
b_entry.bind('<KeyRelease>', lambda _: canvas.config(height=float(b_entry.get() or '1') * CANVAS_SALE))

tk.Label(root, text="Discrete spatial stepsize (dx):").grid(row=2, column=0)
dx_entry = tk.Entry(root)
dx_entry.grid(row=2, column=1)
dx_entry.insert(0, "0.02")

tk.Label(root, text="Wave speed:").grid(row=3, column=0)
c_entry = tk.Entry(root)
c_entry.grid(row=3, column=1)
c_entry.insert(0, "1.0")

run_button = tk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=4, columnspan=2)


crack_positions = []
def create_crack(event):
    crack_positions.append((event.x / CANVAS_SALE, event.y / CANVAS_SALE))
    canvas.create_rectangle(event.x - CIRCLE_RADIUS, event.y - CIRCLE_RADIUS, event.x + CIRCLE_RADIUS, event.y + CIRCLE_RADIUS)

def create_random_crack(count):
    for _ in range(count):
        x, y = (np.random.rand() * float(a_entry.get()), np.random.rand() * float(b_entry.get()))
        crack_positions.append((x, y))
        canvas.create_oval(x * CANVAS_SALE - CIRCLE_RADIUS, y * CANVAS_SALE - CIRCLE_RADIUS, x * CANVAS_SALE + CIRCLE_RADIUS, y * CANVAS_SALE + CIRCLE_RADIUS)


tk.Button(root, text='Create 100 cracks', command=lambda: create_random_crack(100)).grid(row=5, columnspan=2)


CANVAS_SALE = 50
CIRCLE_RADIUS = 1

canvas = tk.Canvas(root, bg='white', width=10 * CANVAS_SALE, height=1 * CANVAS_SALE)
canvas.grid(row=6, columnspan=2)
canvas.bind('<Button-1>', create_crack)

root.mainloop()
