# Jax and Generative AI for Science

Presented by **Axel Donath**, **Patrick Kidger**, **Johanna Haffner** and **Francois Lanusse**

This tutorial is an introduction to generative AI for Science with Jax and the Jax scientific ecosystem. 



## Setup Instructions
Use either `uv`, `pixi`, `conda`,  to create a new env from `requirements.txt`.

Some thoughts and TODOs:
- do we need separarate envs for the bio and astro part?
- do we need a container?
- there will likely be access to https://www.nebari.dev, not sure they provide GPUs
- actually running training will not be possible, except if models are really small. Provide pre-trained weights, maybe finetune.
- Upoad pre-trained weights to Zenodo and provide download script, or maybe have the data ready in a Nebari env.
- It may be unlikely that particpants work on both Astro and Bio, so maybe we structure that exercise time such that they can use the time for their preferred task. However warm-up is for all...


## Requirements
Certainly Python and "Array Based Programming", Dataclasses
Do we require basic Jax knowledge of Numpy API, function transforms, PyTrees?


## Content Overview
Mix of notebooks and scripts?
- Notebooks are nice for presentation but not really for production...
- Maybe Warm-Up and Overviews in Notebooks
- Should we try Marimo? Safe, but more "annoying" for people that have not worked with it before, because of the limitation on variable names. 

### Intro (10 Min)
Intro, Motivation for Jax and Gen AI in Science demonstrate Setup 

### Warm-Up (20 Min)
To bring everybody on the same level:
- Jax basics overview: Numpy API, Lax API, function transforms, PyTrees & dataclasses (Axel)
- Jax science ecosystem overview (equinox, diffrax, optax, numpyro?, ...) (Patrick?)
- Demonstrate some `jax.debug.<>`, which will be useful for exercises...

Exercises (30 Min):
- Toy diffusion model (Figure 2 from Song et al.)
- Self attention visualisation or similar?

-- BREAK ---

### Diffusion Models for Astronomical Image Generation and Reconstruction (Axel + Francois)

Overview & Intro (15 Min) (Francois)


Exercise (45 Min):
- Work with a pre-trained model, e.g. https://github.com/Smith42/astroddpm
- What do we want the people to do for the exercises?
  - Provide e.g. the UNet implementation (mention alternatives) and implement the forward and reverse diffusion part?
  - 
- Mayber also offer two options:
  - Option 1 (more science): use it for an image reconstruction task as a diffusion prior
  - Option 2 (more fun): could finetune to generate images following a galaxy class prompt (use CLIP...) or sample images / inpainting

-- BREAK ---

### Protein Languae Models (Patrick and Johanna)
Overview & Intro (15 Min)

Exercise (45 Min):
TBD


### Wrap-Up and Outlook (15 Min)
- Provide summary...
- Scaling-up : sharding etc., should be covered before?
- Further ressources:
  - https://jax-ml.github.io/scaling-book/tpus/
  - 





