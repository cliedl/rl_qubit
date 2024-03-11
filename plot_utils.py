import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from time import sleep


def render_episode(states, delay):

    # Create the figure and axis for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initial plot setup
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r", alpha=0.5)  # Sphere
    point, = ax.plot([], [], [], 'bo', markersize=10)  # Initial point

    # Set up the initial state
    E, S_real, S_imag = states[0]
    point.set_data(np.array([2*S_real, 2*S_imag]))
    point.set_3d_properties(2*E-1)

    # Adjust the main plot to make room for the slider and button
    plt.subplots_adjust(bottom=0.35)

    # Slider setup
    axcolor = 'lightgoldenrodyellow'
    # Position for the slider
    axtime = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(axtime, 'Time', 0, len(
        states) - 1, valinit=0, valfmt='%0.0f')

    # Button setup
    playax = plt.axes([0.8, 0.025, 0.1, 0.04])  # Position for the button
    button = Button(playax, 'Play', color=axcolor, hovercolor='0.975')

    # Animation function
    def animate(event):
        animating = True
        while animating and slider.val < len(states) - 1:
            slider.set_val(slider.val + 1)
            # Adjust the pause time to control animation speed
            plt.pause(delay)
            if slider.val >= len(states) - 1:
                animating = False

    # Update function for the slider
    def update(val):
        time_step = int(slider.val)
        E, S_real, S_imag = states[time_step]
        point.set_data(np.array([S_real, S_imag]))
        point.set_3d_properties(2*E-1)
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)

    # Connect the animate function to the play button
    button.on_clicked(animate)

    plt.show()
