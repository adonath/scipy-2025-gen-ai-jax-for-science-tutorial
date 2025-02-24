# Jax and Generative AI for Science

Presented by **Axel Donath**, **Patrick Kidger**, **Johanna Haffner** and **Francois Lanusse**

## Abstract 
In this tutorial we will dive into the world of generative AI for science with Jax and the Jax scientific ecosystem.
Starting from limited prior experience with Numpy, Scipy and the Python standard library we will first introduce Jax's
Numpy API, just in time compilation system, native support for tree based data structures and function transforms
for differentiable and scalable programming. Equipped with these new tools, participants will tackle the understanding
of the two most important model architectures behind the recent AI revolution in a scientific context from first principles:
diffusion models for generating astronomical galaxy images and transformer based large language models for predicting the 
structure of proteins. In the astronomy part participants will refine their undertanding of Jax tools and diffusion models,
by coding forward and reverse diffusion processes for a toy problem, then scaling up to training on small scale
astronomical image data and finally generating higher resolution images from a pretrained model. TBC


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

Overview & Intro (15 Min)

- Mostly about language models in general, not much protein-specific. (As we're trying to be science-topic-agnostic.)

Exercise (45 Min):

- Use pretrained https://github.com/patrick-kidger/esm2quinox
- Fine-tune on some custom datasets.
- Predict mutations + use a structure prediction head to see what they look like (aka get some pretty pictures)!

### Wrap-Up and Outlook (15 Min)

- Sharding and autoparallelism
  - https://jax-ml.github.io/scaling-book/tpus/
- More ecosystem: numpyro, lineax, optimistix, ...
  - https://docs.kidger.site/equinox/awesome-list/
  - https://github.com/lockwo/awesome-jax
