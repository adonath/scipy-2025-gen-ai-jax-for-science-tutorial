import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Introduction to Diffusion Models and their connection to Langevin Dynamics

        Before we take a look at actual larger scale "Diffusion Models" (DM), we study a simple toy model or one dimensional case of a DM. DMs in general are tightly connected to an approach named "Langevin Dynamics", which was introduced by Max Welling et al. in 2012 in "Bayesian Learning via Stochastic Gradient Langevin Dynamics", as a way to sample from high dimensional distributions in a Bayesian learning setting. 

        Leaving aside details on proof of convergence etc. the overall idea is very simple. Imagine any standard learning problem, that is solved with maximum likelihood and (stochastic) gradient decent (SGD). Starting from an initial estimate for the model paramters we follow path of the steepest gardient to find the minimum of the log-likelihood function. Now Langevin Dynamics combines SGD with a stochastic parameter update in each step. Both contributions to the parameter updated are balanced usign hyper-parameters. Where the contribution of the gradient update becomes smaller with time (aka learning rate scheduling), while the stochastic update becomes more dominant. This eventually turns the optimization algorithm into a sampling algorithm. What is particulary nice about SGLD is that it draws a connection (or linearly interpolates) between SGD and Sampling. 

        Instead of starting from a single value for the initial parameter distributions, we can also start with a (prior) distribution of the parameters. This will evolve an ensemble of points into the distribution determined by its score function.

        The principle was also illustrated in the original publication that introduced diffusion models, by Song et al. So in this notebook we will reproduce Fig. 2 of https://arxiv.org/abs/2011.13456

        Further References:

        - https://en.wikipedia.org/wiki/Stochastic_gradient_Langevin_dynamics
        - https://icml.cc/2011/papers/398_icmlpaper.pdf
        - Key read: https://yang-song.net/blog/2021/score/
        - https://github.com/yang-song/score_sde
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from jax import numpy as jnp
    from jax.scipy import stats
    import jax
    from functools import partial
    from jax import random
    from collections import namedtuple
    return jax, jnp, namedtuple, np, partial, plt, random, stats


@app.cell
def _(jnp, np):
    def gaussian(x, norm, mu, sigma):
        return norm * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * np.pi))


    def gmm(x, norm, mu, sigma):
        values = jnp.sum(gaussian(x, norm, mu, sigma), axis=0) / mu.shape[0]

        if values.shape == (1,):
            return values.reshape(())

        return values


    def log_gmm(x, norm, mu, sigma):
        return jnp.log(gmm(x, norm, mu, sigma))


    norm, mu, sigma = jnp.array([1, 1])[:, None], jnp.array([-1, 1])[:, None], jnp.array([0.25, 0.25])[:, None]

    x_plot = jnp.linspace(-2, 2, 1000)
    y = gmm(x_plot, norm, mu, sigma)
    return gaussian, gmm, log_gmm, mu, norm, sigma, x_plot, y


@app.cell
def _(plt, x_plot, y):
    plt.plot(x_plot, y)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    return


@app.cell
def _(mo):
    mo.md(r"""In Jax we can directly get the score function by taking the gradient of the log:""")
    return


@app.cell
def _(jax, log_gmm, mu, norm, partial, sigma):
    gmm_log_part = partial(log_gmm, norm=norm, mu=mu, sigma=sigma)
    score_fun = jax.vmap(jax.grad(gmm_log_part))
    return gmm_log_part, score_fun


@app.cell
def _(mo):
    mo.md(r"""Now we can see what it looks like:""")
    return


@app.cell
def _(plt, score_fun, x_plot):
    plt.plot(x_plot, score_fun(x_plot))
    plt.xlabel('x')
    plt.ylabel('d/dx log p(x)')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Using a simple Python loop the algorithm would look like:

        ```python

        n_iter = 200
        alpha_0 = 0.001
        p0 = 1 #0.9999
        beta = 1.
        n_samples = 100_000

        key = random.PRNGKey(42)

        x_init = random.normal(key, (n_samples,)) #jnp.zeros((n_samples,)) #

        x = x_init

        sample_trace_list = []

        for idx in range(n_iter):
            key, subkey = random.split(key)
            alpha = alpha_0 * (p0 ** idx) ** 2
            x = x + alpha * score_fun(x) + jnp.sqrt(2 * alpha * beta) * random.normal(subkey, (n_samples,))
            sample_trace_list.append(x)
        ```

        Convince yourself that this is actually very similar to a "normal" ML training loop, where:

        - $\alpha$ corresponds to the learning rate
        - We are using (optionally) a "learning rate schedule", determined by $\alpha_0$ and $p_0$.

        The schedule however is only needed, once we use stochastic gradient decent. But in this example we evaluate the gradient of the log-likelihood (score) on the full batch, so we can set $p_0=0$. 

        To make the code more efficient we replace the Python loop by a call to [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan), which is exactly made for this purpose:
        """
    )
    return


@app.cell
def _(jax, jnp, namedtuple, partial, random, score_fun):
    Args = namedtuple("Args", ["key", "idx", "x", "alpha_0", "p_0", "beta"])

    def sample(score, args, _):
        key, subkey = random.split(args.key)
        alpha = args.alpha_0 * (args.p_0 ** args.idx) ** 2
        dx = random.normal(subkey, args.x.shape)
        x = args.x + alpha * score(args.x) + jnp.sqrt(2 * alpha * args.beta) * dx
        return Args(key, args.idx + 1, x, args.alpha_0, args.p_0, args.beta), x


    n_samples = 100_000
    n_iter = 200

    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    init = Args(
        key=key,
        idx=0,
        x=random.normal(subkey, (n_samples,)),
        alpha_0=0.001,
        p_0=1, #0.9999
        beta=1.,
    )

    result, sample_trace = jax.lax.scan(partial(sample, score_fun), init, length=n_iter)
    return (
        Args,
        init,
        key,
        n_iter,
        n_samples,
        result,
        sample,
        sample_trace,
        subkey,
    )


@app.cell
def _(mo):
    mo.md(r"""I have found it a good practice to use named tuples to handle the arguments of the body functions. Named tuples are automatically handled as PyTrees by jax.""")
    return


@app.cell
def _(gmm, init, mu, norm, plt, result, sigma, x_plot):
    plt.hist(result.x, bins=200, density=True, histtype='step', label="Transformed samples");
    plt.hist(init.x, bins=100, density=True, histtype='step', label="Initial samples");
    plt.plot(x_plot, gmm(x_plot, norm, mu, sigma), label="True distribution")
    plt.xlim(-4, 4)
    plt.ylim(0, 1.)
    plt.legend();
    return


@app.cell
def _(mo):
    mo.md(r"""We have basically written a sampler, that can sample from any 1d distribution as long as we have its score function available! However, in practice you will find that the sampling is not very stable, meaning that the final distribution depends strongly on the choice of the hyper-parameters $\alpha_0$, $p_0$ and $\beta$.""")
    return


@app.cell
def _(jax, jnp, partial):
    default_hist = partial(jnp.histogram, bins=100, range=(-2, 2), density=True)

    batched_histogram = jax.vmap(default_hist)
    return batched_histogram, default_hist


@app.cell
def _(batched_histogram, plt, random, sample_trace):
    def plot_diffusion_trace(trace, n_traces=5, ax=None, x_min=-2, x_max=2):
        hist_values, hist_edges = batched_histogram(trace)    

        n_iter, n_samples = trace.shape

        ax = plt.subplot() or ax
        ax.imshow(hist_values.T[:, :], extent=[0, n_iter, x_min, x_max], aspect="auto", origin="lower")

        # plot some example traces
        key = random.PRNGKey(9823)
        for idx in random.randint(key, (n_traces,), 0, n_samples):
            ax.plot(trace[:, idx])

        ax.set_xlabel("# Iteration")
        ax.set_ylabel("x")
        return ax

    plot_diffusion_trace(sample_trace)
    return (plot_diffusion_trace,)


@app.cell
def _(mo):
    mo.md(r"""Now we take a look at the inverse process, which is the actual diffusion process.""")
    return


@app.cell
def _(key, mu, random, sigma):
    n_samples_ = 500_000

    x_init = sigma * random.normal(key, (2, n_samples_ // 2,)) + mu
    return n_samples_, x_init


@app.cell
def _(plt, x_init):
    _ax = plt.subplot()
    _ax.hist(x_init.flatten(), bins=100, density=True, histtype='step', label="Initial samples")
    _ax
    return


@app.cell
def _(jnp, key, n_iter, random, x_init):
    beta_t = jnp.linspace(0, 1, n_iter)
    x = x_init.flatten()

    sample_trace_diffusion_beta = []

    for idx, _beta in enumerate(beta_t):
        _key, _sub_key = random.split(key)
        x = jnp.sqrt(1. - _beta) * x + jnp.sqrt(_beta) * random.normal(key=_sub_key, shape=x.shape)
        sample_trace_diffusion_beta.append(x)

    sample_trace_diffusion_beta = jnp.stack(sample_trace_diffusion_beta, axis=0)
    return beta_t, idx, sample_trace_diffusion_beta, x


@app.cell
def _(plot_diffusion_trace, sample_trace_diffusion_beta):
    plot_diffusion_trace(sample_trace_diffusion_beta, x_min=-4, x_max=4)
    return


@app.cell
def _(mo):
    mo.md(r"""We can quickly verify that the distribution corresponds to a unit (normal) Gaussian distribution:""")
    return


@app.cell
def _(x):
    print(f"Mean: {x.mean():.2f}")
    print(f"Variance: {x.var():.2f}")
    return


@app.cell
def _(plt, x):
    _ax = plt.subplot()
    _ax.hist(x, bins=100, density=True, histtype='step', label="Diffused samples")
    _ax.set_xlabel("x")
    _ax.set_ylabel("p(x)")
    _ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
