# JAX and Generative AI for Science

<p align="center">
<img width="50%" src="https://raw.githubusercontent.com/adonath/scipy-2025-gen-ai-jax-for-science-tutorial/main/jax-gen-ai-banner.jpg" alt="banner"/>
</p>

Presented by **Axel Donath**, **Patrick Kidger**, **Johanna Haffner** and **Francois Lanusse**

## Abstract 
In this tutorial we will dive into the world of generative AI for science with JAX and the JAX scientific ecosystem ("autodifferentiable GPU-capable Scipy"). Starting from prior experience with NumPy, SciPy and the Python standard library we will first introduce JAX's own NumPy-like API, Just-In-Time compilation system, native support for tree based data structures, and function transforms for differentiable and scalable programming. Equipped with these new tools, participants will tackle two important generative AI paradigms, as applied to scientific problems: diffusion models for generating astronomical galaxy images, and transformer based large language models for understanding the sequence and structure of proteins.

## Description

Generative AI plays an increasingly important role in scientific discovery, from generating new molecules to emulating complex physical systems. In this tutorial we will introduce JAX and the JAX scientific ecosystem and its application to generative AI for science. JAX is a Python library that provides a NumPy-like API, Just-In-Time compilation to XLA, and support for automatic differentiation and scalable programming and vectorization techniques. Its function based approach and native support for hardware accelerators, make it the ideal choice for many applications in generative AI
and scientific computing in particular. In the tutorial we will cover two paradigms of generative AI: diffusion models and transformer based large language models. The tutorial is designed for participants with prior experience in Python, NumPy and SciPy, but no previous knowledge of JAX or generative AI is required.

In the first block we will provide an in-depth introduction to JAX to bring all the participants to the same level.
We will cover the JAX NumPy API, Just-In-Time compilation system, function transforms for differentiable and scalable programming, and JAX's native support for tree based data structures. An introductional hands-on exercise will follow, where particpants apply and practice the new JAX concepts to prepare them for real generative AI applications.

In the second block we will cover the field of diffusion models in the context of astronomy. We will take the participants from the understanding of Langevin dynamics sampling, covered in the introduction, to score matching and time conditioned score functions. This will help participants understanding the continuous-time diffusion process and its relation to ordinary and stochastic differential equations. The exercises will first cover visualization of the diffusion process and flow function on the "Swiss roll" example. With the newly developed intuition we will then move to the implementation of the full forward and reverse diffusion process for a U-Net based denoising diffusion probabilistic model (DDPM). We will train the model on a small-size astro dataset, and generate full size images from the pretrained Astro-DDPM model.

The third block will start with a general introduction on large language models LLM and their application and adaptation to the field of protein science. We will cover the basics of encoder-only transformer architectures, and introduce the concepts of representation learning and self-supervised training. This will be framed in terms of applications to both protein sequences (without requiring participants to already be familiar with proteins) and natural language. The exercises will include a hands-on session with pretrained models from the ESM2 family, which will serve as a useful technical reference for those new to either JAX or new to transformers. Participants will additionally have access to a dataset of protein thermostability measurements. Putting these two together, they will attempt to generate novel proteins with high predicted thermostability. ESMFold will be used to visualize structures for the generated candidates, and we may run a small competition to see who can get the best sequences (as evaluated by the instructor's somewhat larger model!).

Participants will leave with an to generative AI and be ready to apply concepts of generative AI and JAX for their own projects in their scientific domain.



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
