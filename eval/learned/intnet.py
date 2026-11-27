"""The encoder's integer networks: their exact forward pass, file format and overflow bounds.

Where it fits: stop_train.py trains the stop network (crates/nc-encoder/src/stop_value.rs),
stores it in this form and writes it as a Rust table; the encoder runs it with the nc-dsp
network kernels
(`nn_dense_clipped`, `nn_dense_linear`). `forward` here is the reference those kernels must
match bit for bit, written from the arithmetic alone:

- inputs: u8 in 0..=127;
- hidden layer: out[o] = clamp((bias[o] + sum_i w[o][i] * x[i]) >> shift, 0, 127), with
  w int8 in -127..=127, bias int32 and `>>` an arithmetic shift (it rounds down);
- output layer: value = bias + sum_i w[i] * x[i] in int32, no clamp and no shift.

The shift is a constant of each hidden layer. Reading the network in float: activations
a = x / 127, a hidden layer's weights W = w / 2^shift and biases B = bias / (127 * 2^shift);
the output is value / (127 * 2^output_scale_log2) in the units of the training target.

Overflow (checked by `check_bounds` for every stored network):

- One product is at most 127 * 127 = 16 129 in magnitude, and two of them 32 258, inside
  the signed 16-bit pair sums of the AVX2, NEON and SIMD128 kernels (the reason weights stop
  at -127, not -128).
- A layer of n inputs sums to at most n * 16 129 + |bias|. For the padded widths used here,
  n <= 64, that is 1 032 256 + |bias|, so int32 cannot overflow while
  |bias| <= 2^31 - 1 - n * 16 129; `check_bounds` requires the tighter |sum| < 2^24, which
  also makes float32 sums of these integers exact (quantise.py trains through them).
- The reachable bound is tighter: with the stored weights, row o cannot exceed
  |bias[o]| + 127 * sum_i |w[o][i]|; `largest_sums` reports it per layer.
"""

import numpy as np

ACTIVATION_MAX = 127
WEIGHT_MAX = 127
#: The kernels take dense inputs in multiples of this (zeros pad inputs and weight rows).
DENSE_INPUT_ALIGN = 32
#: Bound required of every accumulated sum: exact in float32, and far inside int32.
SUM_LIMIT = 1 << 24
PRODUCT_MAX = ACTIVATION_MAX * WEIGHT_MAX
assert 2 * PRODUCT_MAX <= np.iinfo(np.int16).max


def padded(count):
    """`count` rounded up to the kernels' input alignment."""
    return -(-count // DENSE_INPUT_ALIGN) * DENSE_INPUT_ALIGN


class IntNet:
    """Layers as (int8 weights [outputs, inputs], int32 biases [outputs]); the last is linear."""

    def __init__(self, weights, biases, shifts, output_scale_log2):
        self.weights = [np.asarray(w, dtype=np.int8) for w in weights]
        self.biases = [np.asarray(b, dtype=np.int32) for b in biases]
        self.shifts = [int(shift) for shift in shifts]  # one per hidden layer
        assert len(self.shifts) == len(self.weights) - 1
        self.output_scale_log2 = int(output_scale_log2)
        check_bounds(self)

    @property
    def sizes(self):
        """Layer widths, inputs first: e.g. (11, 32, 8, 1)."""
        return (self.weights[0].shape[1],) + tuple(w.shape[0] for w in self.weights)

    @property
    def output_unit(self):
        """Integer output per unit of the training target."""
        return ACTIVATION_MAX * (1 << self.output_scale_log2)

    def multiply_accumulates(self, pad=False):
        """Weights the forward pass multiplies: real ones, or the padded rows the kernels walk."""
        return sum(w.shape[0] * (padded(w.shape[1]) if pad else w.shape[1]) for w in self.weights)


def largest_sums(net):
    """Per layer, the largest |bias + sum| any input in 0..=127 can reach with these weights."""
    return [int((np.abs(b.astype(np.int64)) + ACTIVATION_MAX * np.abs(w.astype(np.int64)).sum(1)).max())
            for w, b in zip(net.weights, net.biases)]


def check_bounds(net):
    """Weights in -127..=127 and no reachable sum at or beyond SUM_LIMIT (so none overflows int32)."""
    for w in net.weights:
        assert np.abs(w.astype(np.int16)).max() <= WEIGHT_MAX, "weight outside -127..=127"
        assert padded(w.shape[1]) * PRODUCT_MAX < SUM_LIMIT, "layer too wide for the sum bound"
    assert max(largest_sums(net)) < SUM_LIMIT, f"sum bound exceeded: {largest_sums(net)}"


def forward(net, x, activations=False):
    """The exact integer outputs (int32, [n]) for u8 inputs [n, inputs]; optionally each hidden layer too."""
    assert x.dtype == np.uint8 and x.max(initial=0) <= ACTIVATION_MAX
    hidden, layer_outputs = x.astype(np.int64), []
    for w, b, shift in zip(net.weights, net.biases, net.shifts):
        sums = hidden @ w.astype(np.int64).T + b.astype(np.int64)
        hidden = np.clip(sums >> shift, 0, ACTIVATION_MAX)  # numpy's >> is arithmetic
        layer_outputs.append(hidden.astype(np.uint8))
    value = hidden @ net.weights[-1].astype(np.int64).T + net.biases[-1].astype(np.int64)
    assert np.abs(value).max(initial=0) < SUM_LIMIT
    value = value[:, 0].astype(np.int32)
    return (value, layer_outputs) if activations else value


def save(path, net):
    arrays = {"shifts": np.asarray(net.shifts, dtype=np.int32), "output_scale_log2": net.output_scale_log2}
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
    np.savez(path, **arrays)


def load(path):
    with np.load(path) as arrays:
        count = sum(name.startswith("w") for name in arrays.files)
        return IntNet([arrays[f"w{i}"] for i in range(count)], [arrays[f"b{i}"] for i in range(count)],
                      arrays["shifts"], arrays["output_scale_log2"])
