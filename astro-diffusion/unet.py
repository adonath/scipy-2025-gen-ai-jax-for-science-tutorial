"""Pure jax implementation of a U-Net

Architecture is based on: https://github.com/Smith42/astroddpm/blob/master/denoising_diffusion_pytorch.py

It is re-implemented here in pure jax, because "What I cannot build, I do not understand..."


The coding pattern is based on the following ideas:

- Use dataclasses to define the model
- Use `__call__` method to define the forward pass
- Use `from_args` method to create the model from simpler arguments
- However if you have a pre-trained model `.from_args` is basically optional. You just have to get the arrays into
the correct tree structure. 
- This is also where the random keys are split and parameters are initialized
- Never define `__init__`, such that classes can easily be serialized and de-serialized
see https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization without
worries. No need to handle super().__init__() etc.
- The state of the model is fully defined by the dataclass fields, static fields are marked as such
- It also emphasizes that jax is function based, while the classes really just serve as a
container / pytree for the parameters and to handle easier configuration and initialization
from hyper-parameters.
- The iter(random.split(key, N)) pattern is used to easily get new keys for each layer,
in the creation methods.


A very similar UNet architecture implemented in Equinox can be found here:

https://docs.kidger.site/equinox/examples/unet/

Note that equinox has a lot of additional features that make it easier to implement.
However I think the educational value of implementing this in pure jax is a bit higher,
as you get to see how some deep learning concepts and abstractions are implemented under
the hood. The coding pattern outlined above will also generalize to any non-deep learning code,
which you might want to write in jax.

This implementation also serves as a blueprint of how convert existitng PyTorch models
to Jax. 

"""

from collections.abc import Sequence as SequenceBase
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, ClassVar, Sequence

import jax
from einops import rearrange
from jax import lax
from jax import numpy as jnp
from jax.tree_util import SequenceKey, register_dataclass

N_KEYS_MAX = 100


def join_path(path):
    """Join path to Pytree leave"""
    values = [
        getattr(_, "name", str(getattr(_, "idx", getattr(_, "key", None))))
        for _ in path
    ]
    return ".".join(values)


@dataclass
class UNetConfig:
    """UNet config"""

    dim: int
    dim_out: int
    key: jax.Array
    dim_mults: Sequence[int] = (1, 2, 4, 8)
    n_groups: int = 8
    n_channels: int = 3
    n_heads: int = 4

    @classmethod
    def dummy(cls):
        """Dummy config to generate a model of the same pytree structure,
        but with minimal number of parameters"""
        return cls(dim=1, dim_out=1, key=jax.random.PRNGKey(0), n_heads=4)


@jax.tree_util.register_pytree_with_keys_class
class Sequential(SequenceBase):
    """Layer sequence, that looks like a normal list, but has a __call__ method
    to apply the layers in sequence.
    """

    def __init__(self, layers):
        self._layers = layers

    def __getitem__(self, idx):
        return self._layers[idx]

    def __len__(self):
        return len(self._layers)

    def tree_flatten_with_keys(self):
        """Flatten the pytree, required by `register_pytree_node_class`"""
        return (
            tuple((SequenceKey(idx), layer) for idx, layer in enumerate(self._layers)),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the pytree, required by `register_pytree_node_class`"""
        return cls(children)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


@register_dataclass
@dataclass
class Identity:
    """Identity layer"""

    def __call__(self, x):
        return x


@register_dataclass
@dataclass
class Linear:
    """Linear layer"""

    weight: jax.Array
    bias: jax.Array | None = None

    @classmethod
    def from_args(cls, dim_in, dim_out, key, use_bias=True):
        """Create linear layer from config arguments"""
        weight = jax.random.normal(key, (dim_in, dim_out))
        bias = jnp.zeros(dim_out) if use_bias else None
        return cls(weight=weight, bias=bias)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.matmul(x, self.weight.mT)

        if self.bias is not None:
            x = x + self.bias

        return x


@register_dataclass
@dataclass
class Mish:
    """Mish activation"""

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * jnp.tanh(jax.nn.softplus(x))


@partial(register_dataclass, data_fields=("weight", "bias"), meta_fields=("n_groups",))
@dataclass
class GroupNorm:
    """Group normalization layer

    See:
        - https://arxiv.org/abs/1803.08494
        - https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

    """

    n_groups: int
    weight: jax.Array | None = None
    bias: jax.Array | None = None
    eps: ClassVar[float] = 1e-5

    @classmethod
    def from_args(cls, n_groups, n_channel: int | None = None):
        """Create group normalization layer from arguments"""

        weight = jnp.ones(n_groups) if n_channel is not None else None
        bias = jnp.zeros(n_groups) if n_channel is not None else None

        return cls(weight=weight, bias=bias, n_groups=n_groups)

    def __call__(self, x: jax.Array) -> jax.Array:
        n_batch, n_channel, n_w, n_h = x.shape

        x = jnp.reshape(
            x, (n_batch, self.n_groups, n_channel // self.n_groups, n_w, n_h)
        )
        x_mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)

        x_var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
        normed = (x - x_mean) * jax.lax.rsqrt(x_var + self.eps)
        normed = jnp.reshape(normed, (n_batch, n_channel, n_w, n_h))

        shape_ref = (1, n_channel, 1, 1)

        if self.weight is not None:
            normed = normed * jnp.reshape(self.weight, shape_ref)

        if self.bias is not None:
            normed = normed + jnp.reshape(self.bias, shape_ref)

        return normed


@register_dataclass
@dataclass
class Conv2D:
    """2D convolutional layer"""

    weight: jax.Array
    bias: jax.Array | None = None
    padding: str = field(default="SAME", metadata=dict(static=True))
    strides: Sequence[int] = field(default=(1, 1), metadata=dict(static=True))

    @classmethod
    def from_args(
        cls,
        dim_in,
        dim_out,
        kernel_size,
        key,
        padding="SAME",
        strides=(1, 1),
        use_bias=True,
    ):
        """Create 2D convolutional layer from arguments"""
        weight = jax.random.normal(key, (kernel_size, kernel_size, dim_in, dim_out))
        bias = jnp.zeros(dim_out) if use_bias else None
        return cls(
            weight=weight,
            bias=bias,
            padding=padding,
            strides=strides,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.strides,
            padding=self.padding,
        )

        if self.bias is not None:
            x = x + jnp.expand_dims(self.bias, axis=(0, 2, 3))

        return x

    def __repr__(self) -> str:
        dim_in, dim_out, *kernel_size = self.weight.shape
        return f"Conv2D(dim_in={dim_in}, dim_out={dim_out}, kernel_size={kernel_size}, stride={self.strides}, padding={self.padding})"


@register_dataclass
@dataclass
class ConvTranspose2D:
    """2D transposed convolutional layer"""

    weight: jax.Array
    bias: jax.Array | None = None
    padding: str = field(default="SAME", metadata=dict(static=True))
    strides: Sequence[int] = field(default=(2, 2), metadata=dict(static=True))

    @classmethod
    def from_args(
        cls,
        dim_in,
        dim_out,
        kernel_size,
        key,
        padding="SAME",
        strides=(1, 1),
        use_bias=True,
    ):
        """Create 2D convolutional layer from arguments"""
        weight = jax.random.normal(key, (kernel_size, kernel_size, dim_in, dim_out))
        bias = jnp.zeros(dim_out) if use_bias else None
        return cls(
            weight=weight,
            bias=bias,
            padding=padding,
            strides=strides,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = lax.conv_transpose(
            lhs=x,
            rhs=self.weight,
            strides=self.strides,
            padding=self.padding,
            # these settings make it equivalent to PyTorch ConvTranspose2d
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            transpose_kernel=True,
        )

        if self.bias is not None:
            x = x + jnp.expand_dims(self.bias, axis=(0, 2, 3))

        return x


@register_dataclass
@dataclass
class Block:
    """Block"""

    block: Sequential

    @classmethod
    def from_args(cls, dim, dim_out, key, n_groups: int = 8):
        """Create block from init arguments"""
        block = Sequential(
            [
                Conv2D.from_args(dim, dim_out, kernel_size=3, key=key),
                GroupNorm.from_args(n_groups=n_groups, n_channel=dim_out),
                Mish(),
            ]
        )

        return cls(block=block)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.block(x)
        return x


@register_dataclass
@dataclass
class ResnetBlock:
    """Resnet block"""

    mlp: Sequential
    block1: Block
    block2: Block
    res_conv: Conv2D | Identity

    @classmethod
    def from_args(cls, dim, dim_out, dim_time_emb, key, n_groups: int = 8):
        """Create resnet block from arguments"""
        keys = iter(jax.random.split(key, 4))
        res_conv = (
            Conv2D.from_args(dim, dim_out, kernel_size=1, key=next(keys))
            if dim != dim_out
            else Identity()
        )

        mlp = Sequential(
            [
                Mish(),
                Linear.from_args(dim_time_emb, dim_out, key=next(keys)),
            ]
        )

        return cls(
            mlp=mlp,
            block1=Block.from_args(dim, dim_out, n_groups=n_groups, key=next(keys)),
            block2=Block.from_args(dim_out, dim_out, n_groups=n_groups, key=next(keys)),
            res_conv=res_conv,
        )

    def __call__(self, x, t_embd):
        h = self.block1(x)
        h = h + jnp.expand_dims(self.mlp(t_embd), axis=(2, 3))
        h = self.block2(h)

        if self.res_conv is not None:
            h = h + self.res_conv(x)

        return h


@register_dataclass
@dataclass
class Upsample:
    """Upsample block"""

    conv: ConvTranspose2D

    @classmethod
    def from_args(cls, dim, key):
        """Create upsample block from arguments"""
        conv = ConvTranspose2D.from_args(
            dim, dim, kernel_size=4, key=key, strides=(2, 2)
        )
        return cls(conv=conv)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        return x


@register_dataclass
@dataclass
class Downsample:
    """Downsample block"""

    conv: Conv2D

    @classmethod
    def from_args(cls, dim, key):
        """Create downsample block from arguments"""
        return cls(
            conv=Conv2D.from_args(
                dim,
                dim,
                kernel_size=3,
                key=key,
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
            ),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        return x


@register_dataclass
@dataclass
class Residual:
    """Residual block wrapper"""

    fn: Callable

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.fn(x)


@register_dataclass
@dataclass
class Rezero:
    """Rezero wrapper"""

    fn: Any
    g: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fn(x) * self.g


@register_dataclass
@dataclass
class LinearAttention:
    """Linear attention"""

    to_qkv: Conv2D
    to_out: Conv2D
    n_heads: int = field(metadata=dict(static=True))

    @classmethod
    def from_args(cls, dim, key, dim_heads: int = 32, n_heads: int = 4):
        """Create linear attention from arguments"""
        keys = iter(jax.random.split(key, 2))
        dim_hidden = dim_heads * n_heads
        return cls(
            to_qkv=Conv2D.from_args(
                dim, 3 * dim_hidden, key=next(keys), kernel_size=1, use_bias=False
            ),
            to_out=Conv2D.from_args(dim_hidden, dim, key=next(keys), kernel_size=1),
            n_heads=n_heads,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.n_heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("bhdn,bhen->bhde", k, v)
        out = jnp.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.n_heads, h=h, w=w
        )
        return self.to_out(out)


@register_dataclass
@dataclass
class SinusoidalPosEmb:
    """Sinusoidal positional embeddings

    See also:
        - https://arxiv.org/abs/1706.03762
        - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/


    """

    frequencies: jax.Array = field(metadata=dict(static=True))

    @classmethod
    def from_dim(cls, dim, f_max=10_000):
        """Create sinusoidal positional embeddings with given dimension"""
        half_dim = dim // 2
        freqs = jnp.log(f_max) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -freqs)
        return cls(frequencies=freqs)

    def __call__(self, t):
        phi = jnp.expand_dims(t, axis=1) * self.frequencies
        phi = jnp.concatenate((jnp.sin(phi), jnp.cos(phi)), axis=1)
        return phi


@register_dataclass
@dataclass
class UNet:
    """UNet"""

    time_pos_emb: SinusoidalPosEmb
    mlp: Sequential
    downs: list[tuple[ResnetBlock, ResnetBlock, LinearAttention, Downsample]]
    mid_block1: ResnetBlock
    mid_attn: LinearAttention
    mid_block2: ResnetBlock
    ups: list[tuple[ResnetBlock, ResnetBlock, LinearAttention, Downsample]]
    final_conv: Sequential

    @classmethod
    def from_args(
        cls,
        dim,
        dim_out,
        key,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        n_heads: int = 4,
        n_groups: int = 8,
        n_channels: int = 3,
    ):
        """Create UNet from arguments"""
        keys = iter(jax.random.split(key, N_KEYS_MAX))

        kwargs = {}
        kwargs["time_pos_emb"] = SinusoidalPosEmb.from_dim(dim)
        kwargs["mlp"] = Sequential(
            [
                Linear.from_args(dim, 4 * dim, key=next(keys)),
                Mish(),
                Linear.from_args(4 * dim, dim, key=next(keys)),
            ]
        )

        dims = [n_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        downs = []

        for idx, (d_in, d_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)

            layers = (
                ResnetBlock.from_args(d_in, d_out, key=next(keys), dim_time_emb=dim),
                ResnetBlock.from_args(d_out, d_out, key=next(keys), dim_time_emb=dim),
                Residual(
                    Rezero(
                        LinearAttention.from_args(
                            d_out, key=next(keys), n_heads=n_heads
                        ),
                        g=jnp.array([1]),
                    )
                ),
                Downsample.from_args(d_out, key=next(keys))
                if not is_last
                else Identity(),
            )

            downs.append(layers)

        kwargs["downs"] = downs

        ups = []

        for idx, (d_in, d_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (num_resolutions - 1)
            layers = (
                ResnetBlock.from_args(
                    2 * d_out, d_in, key=next(keys), dim_time_emb=dim
                ),
                ResnetBlock.from_args(d_in, d_in, key=next(keys), dim_time_emb=dim),
                Residual(
                    Rezero(
                        LinearAttention.from_args(
                            d_in, key=next(keys), n_heads=n_heads
                        ),
                        g=jnp.array([1]),
                    )
                ),
                Upsample.from_args(d_in, key=next(keys)) if not is_last else Identity(),
            )

            ups.append(layers)

        kwargs["ups"] = ups

        dim_mid = dims[-1]
        kwargs["mid_block1"] = ResnetBlock.from_args(
            dim_mid, dim_mid, dim_time_emb=dim, key=next(keys)
        )
        kwargs["mid_attn"] = Residual(
            Rezero(
                LinearAttention.from_args(dim, key=next(keys), n_heads=n_heads),
                g=jnp.array([1]),
            )
        )
        kwargs["mid_block2"] = ResnetBlock.from_args(
            dim_mid, dim_mid, dim_time_emb=dim, key=next(keys)
        )
        kwargs["final_conv"] = Sequential(
            [
                Block.from_args(dim, dim, key=next(keys)),
                Conv2D.from_args(dim, dim_out, kernel_size=1, key=next(keys)),
            ]
        )
        return cls(**kwargs)

    def n_parameters(self):
        """Number of parameters"""
        n_parameters = sum(
            p.size if isinstance(p, jax.Array) else 0 for p in jax.tree_leaves(self)
        )
        return n_parameters

    @classmethod
    def read(cls, filename, device=None):
        """Read model from Pytorch file"""
        import torch

        data = torch.load(filename, map_location=torch.device("cpu"))

        weights = {}

        prefix = "module.denoise_fn."

        for key, value in data["model"].items():
            if not key.startswith(prefix):
                continue

            weights[key.replace(prefix, "")] = jnp.asarray(value, device=device)

        # create equivalent pytree structure to get the treedef
        config = UNetConfig.dummy()
        model_dummy = cls.from_args(**asdict(config))
        values_and_paths, treedef = jax.tree.flatten_with_path(model_dummy)

        # reorder the pytorch weights according to the treedef
        values = [weights[join_path(path)] for path, _ in values_and_paths]
        model = jax.tree.unflatten(treedef, values)

        # the sinusoidal embedding is not part of the weights
        model.time_pos_emb = SinusoidalPosEmb.from_dim(model.mlp[0].weight.shape[1])
        return model

    @jax.jit
    def __call__(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []

        for resnet, resnet_2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet_2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet_2, attn, upsample in self.ups:
            val = h.pop()
            x = jnp.concatenate((x, val), axis=1)
            x = resnet(x, t)
            x = resnet_2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
