# JAX and Generative AI for Science

Presented by **Axel Donath**, **Patrick Kidger**, **Johanna Haffner** and **Francois Lanusse**

## Abstract 
In this tutorial we will dive into the world of generative AI for science with JAX and the JAX scientific ecosystem ('autodifferentiable GPU-capable scipy').
Starting from prior experience with NumPy, SciPy and the Python standard library we will first introduce JAX's own
NumPy-like API, Just-In-Time compilation system, native support for tree based data structures, and function transforms
for differentiable and scalable programming. Equipped with these new tools, participants will tackle
two important generative AI paradigms, as applied to scientific problems:
diffusion models for generating astronomical galaxy images, and transformer based large language models for understanding the 
sequence and structure of proteins.

(NB abstract is intended to be around 100 words or less -- this is at 101.)

## Requirements
Experience with NumPy and some SciPy, dataclasses and functools from the standard library. Knowledge on basic statistics, calculcus and linear algebra. No previous knowledge of either JAX or generative AI expected. Astronomy and biology are used as applications and we expect both parts to be of interest to the whole audience.

**Optionally** participants can prepare for the tutorial by reading the following resources. Not required, but
might provide a smoother experience when working on the exercises:

- JAX magical Numpy Tutorial from Scipy 2021: https://ericmjl.github.io/notes/tutorial-proposals/magical-numpy-with-jax-scipy-2021/ 
- Thoughts on JAX coming from PyTorch: https://kidger.site/thoughts/torch2jax/


### Introduction (30 Min)
Short introduction and motivation for JAX and Generative AI in science. To bring everybody on the same level we will start with:

- JAX basics overview: NumPy API
- Function transforms: `jit`, `vmap`, `grad`, `scan`
- Pytrees and tree manipulation: `tree_map` and tree support in function transforms
- Trace time vs runtime, devices and `jax.debug.<>` 
- JAX scientific ecosystem overview (Equinox, Diffrax, Optax)


#### Warm-Up Exercises (30 Min)
- Figure out set-up, open laptops, debug environments, ... !
- Hands on with the above, including understand Langevin Dynamics sampling by replicating Fig 1 from Song et al 2021 using pure JAX

-- BREAK --- (15 Min)

### Diffusion Models for Astronomical Image Generation (Axel + Francois)

**Overview and Intro (15 minutes)**

In this section we will introduce diffusion models from a theoretical perspective and 
demostrate their application for image reconstruction in radio astronomy and gravitational
lensing. 

**Diffusion Exercises (45 Min)**

The exercices will cover score matching and time conditioned score functions on the "Swiss role" example. This will help participants understanding the continuous-time diffusion process and its relation to ODEs. This knowledge they will use to implement the full forward and reverse diffusion process for U-Net based DDPM, train it on a small-szie astro dataset, and generate full size images from the pretrained Astro-DDPM model, which we will make available.

Further ressources: 
    - https://docs.kidger.site/equinox/examples/score_based_diffusion/
    - https://github.com/Smith42/astroddpm


-- BREAK --- (15 Min)

### Protein Language Models (Patrick and Johanna)

**Overview and Intro (15 minutes)**

This will cover an introduction to encoder-only transformer architectures, and to representation learning / masked language modelling / self-supervised training. This will be framed in terms of applications to both protein sequences (without requiring participants to already be familiar with proteins) and natural language.

**Exercise (45 minutes)**

Models with pretrained weights will be available from https://github.com/patrick-kidger/esm2quinox. These will serve as a useful technical reference for those new to either JAX or new to transformers. Participants will additionally have access to a dataset of protein thermostability measurements. Putting these two together, they will attempt to generate novel proteins with high predicted thermostability.

ESMFold wil be used to visualize structures for the generated candidates, and we may run a small competition to see who can get the best sequences (as evaluated by the instructor's somewhat larger model!)

### Wrap-Up and Outlook (15 Min)

Overall, participants will leave with an introduction to generative AI, the scientific JAX ecosystem, and what can be done with an 'autodifferentiable GPU-capable scipy'. Concluding topics for further reading include:

- Sharding and autoparallelism
  - https://jax-ml.github.io/scaling-book/tpus/
- More ecosystem: numpyro, Lineax, Optimistix, ...
  - https://docs.kidger.site/equinox/awesome-list/
  - https://github.com/lockwo/awesome-jax



## Setup Instructions
To execute the notebooks and examples in this tutorial, you will have the choice between three different environments:

### Working on Nebari (recommended)
As some of the exercises have higher computational demands we recommend to work in an Nebari session, which will provide access to GPUs. Nebari will provide a pre-defined enviroment for this tutorial. Detailed instructions will follow, but will be similar to the [instructions from 2024](https://docs.google.com/document/d/11YWMZKW6Y4tXnMs3Jekc1S7BQWTR6THZazDaq3WoNxw/edit?tab=t.0#heading=h.wtozhevy8waj). Access to the GPU servers will likely require a special coupon, that we will share at the beginning of the tutorial session.


### Google Colab
Alternatively if Nebari is not available, you can use Google Colab as an alternative. In this case you have to install the requirements in a Colab session. This can be done by running the following command in a code cell:

```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```

This wil make sure to install the requirements in the current Python environment.

Open questions:
- How to mount / open the whole repo in Colab?
- Do participants need a pro account?


### Working locally (not recommended)
Some of the tutorial exercices can also be done locally if you have a very capable machine, however keep in mind this might not be sufficient for all exercises and will keep your machine busy for a while on the advanced exercises.

Working locally requires a working Python >=3.10 installation.

In any case start by cloning the repository:

```bash
git clone https://github.com/adonath/scipy-2025-gen-ai-jax-for-science-tutorial
cd scipy-2025-gen-ai-jax-for-science-tutorial
```


#### Using A System Python
If you are using a system Python, we recommend to create a virtual environment first. This can be done with the following command:

```bash
python -m venv scipy-2025-jax-gen-ai-tutorial
source ./scipy-2025-jax-gen-ai-tutorial/bin/activate
python -m pip install -r requirements.txt
```

### Using uv
If you are using [uv](https://docs.astral.sh/uv/getting-started/installation/) we would also recommend to create a new environment with the following command:

```bash
uv venv scipy-2025-jax-gen-ai-tutorial --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

#### Using Conda / Mamba

If you are using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl), we recommend to create a new environment with the following command:

```bash 
conda create --name scipy-2025-jax-gen-ai-tutorial --file requirements.txt
conda activate scipy-2025-jax-gen-ai-tutorial
```