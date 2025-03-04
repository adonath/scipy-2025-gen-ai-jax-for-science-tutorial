"""Utility functions for plotting for the JAX and generative AI Scipy tutorial."""
import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jax import numpy as jnp


def animate_trajectory(x, filename, x_range=(-2, 2), y_range=(-2, 2), **kwargs):
    """Animate diffusion trajectory

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_frames, n_particles, 2) representing the trajectory.
    filename : str
        The filename to save the animation to.
    x_range : tuple, optional
        The range of x values to plot over, in the form (min, max).
    y_range : tuple, optional
        The range of y values to plot over, in the form (min, max).
    kwargs : dict, optional
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    """
    fig, ax = plt.subplots()

    kwargs.setdefault("markersize", 0.2)
    kwargs.setdefault("linestyle", "")
    kwargs.setdefault("marker", "o")

    (line,) = ax.plot([], [], **kwargs)
    ax.set_aspect("equal")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def animate(idx):
        line.set_data(x[idx, :, 0], x[idx, :, 1])  # Update line data
        return (line,)

    ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=100, blit=True)
    ani.save(filename, writer="pillow")
    return ani


def plot_vector_field(model, ax=None, x_range=(-2, 2, 10), y_range=(-2, 2, 10)):
    """Plot vector field of grad(log(p)), aka score function

    Parameters
    ----------
    model : callable
        A function that takes a 2D input and returns a 2D output.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    x_range : tuple, optional
        The range of x values to plot over, in the form (min, max, num_points).
    y_range : tuple, optional
        The range of y values to plot over, in the form (min, max, num_points).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the vector field plotted.
    """
    ax = ax or plt.gca()

    xx, yy = jnp.meshgrid(jnp.linspace(*x_range), jnp.linspace(*y_range))
    grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    scores = jax.vmap(model)(grid)

    scores_norm = jnp.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / scores_norm * jnp.log(1 + scores_norm)
    ax.quiver(xx, yy, scores_log1p[:, 0], scores_log1p[:, 1], color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
