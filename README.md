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

## Setup Instructions

Use colab or https://www.nebari.dev -- TBD! We do want GPUs.

- Do we need separarate envs for the bio and astro part? No.
- We'll provide preprocessed datasets for the astro section; premade transformer for the bio section.
- We expect that everything will be of interest to everyone; the focus is on the JAX and GenAI rather than on the scientific applications.

### Working on Nebari (recommended)
As some of the exercises have higher computational demands we recommend to work in an Nebari session, which will
provide access to GPUs.

### Google Colab
You can open then tutorial in a Google Colab session. This typically offers some access to GPUs.

### Working locally (not recommended)
The tutorial exerices can also be done locally if you have a very capable machine.

## Requirements
NumPy and some SciPy. Dataclasses, functools. Knowledge on basic statistics, calculcus and linear algebra.
No previous knowledge of either JAX or generative AI expected.
Astro and bio are used as applications and we expect both parts to be of interest to the whole audience.

### Intro (30 Min)
Intro, Motivation for JAX and Gen AI in science demonstrate setup. To bring everybody on the same level:
- JAX basics overview: NumPy API, jit, vmap grad, PyTrees (Axel)
- Trace time vs runtime (https://kidger.site/thoughts/torch2jax/)
- JAX science ecosystem overview (Equinox, Diffrax, Optax) (Patrick)
- `jax.debug.<>`

### Warm-Up (30 Min)
- Figure out set-up, open laptops, debug environments, ... !
- Hands on with the above.

-- BREAK ---

### Diffusion Models for Astronomical Image Generation and Reconstruction (Axel + Francois)

Overview & Intro (15 Min) (Francois)

- Continuous-time diffusion (ODEs will be more familiar for a scipy audience)

Exercise (45 Min):

- Train toy model themselves (but provide architecture pre-built?)
- Have participants implement the forward and reverse diffusion using Equinox and Diffrax.
- Want training to take 5 minutes on a toy dataset. (Images of some sort?)
- Then provide a pretrained large UNet for them to try out at the end.
- Reference points:
    - https://docs.kidger.site/equinox/examples/score_based_diffusion/
    - https://github.com/Smith42/astroddpm

-- BREAK ---

### Protein language models (Patrick and Johanna)

Protein language models (Patrick and Johanna)

**Overview and Intro (15 minutes)**

This will cover an introduction to encoder-only transformer architectures, and to representation learning / masked language modelling / self-supervised training. This will be framed in terms of applications to both protein sequences (without requiring participants to already be familiar with proteins) and natural language.

**Exercise (45 minutes)**

Models with pretrained weights will be available from https://github.com/patrick-kidger/esm2quinox. These will serve as a useful technical reference for those new to either JAX or new to transformers. Participants will additionally have access to a dataset of protein thermostability measurements. Putting these two together, they will attempt to generate novel proteins with high predicted thermostability.

ESMFold wil be used to visualize structures for the generated candidates, and we may run a small competition to see who can get the best sequences (as evaluated by the instructor's somewhat larger model!)

### Wrap-Up and Outlook (15 Min)

Overall, participants will leave with an introduction to the scientific JAX ecosystem, and what can be done with an 'autodifferentiable GPU-capable scipy'. Concluding topics for further reading include:

- Sharding and autoparallelism
  - https://jax-ml.github.io/scaling-book/tpus/
- More ecosystem: numpyro, Lineax, Optimistix, ...
  - https://docs.kidger.site/equinox/awesome-list/
  - https://github.com/lockwo/awesome-jax
