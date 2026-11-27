"""Fit the stop network's hidden layers on the on-policy stop dump, and write its Rust table.

Where it fits: the post-trial stop (crates/nc-encoder/src/rd_split.rs `FirstTrialStop`)
skips a planned node's other candidates where its score is above 0; the stop network
(stop_value.rs) adds `weight * (offset - value) + sum_j readout_j * h_j` to that score. The
output terms are tuned on real encodes (nceval.tune); this script supplies the rest: the
hidden layers and the value output, fitted to what the alternatives' coding gained and cost
at each stop point (stop_export.py's table, the encoder's own input bytes).

Targets per stop point, from every admitted alternative coded in full: the relative gain
r = max(0, planned cost - best alternative's cost) / planned cost, and the work ratio
w = the alternatives' summed work / the planned candidate's. Three heads share the hidden
layers (so the readouts see gain, chance and cost): the value ln(1 + VALUE_SCALE * r /
(1 + w)), the gain's indicator r > GAIN_FLOOR (logistic), and ln(1 + w). Only the value head
is exported. Rows are weighted so every clip counts the same.

Integer from the start (intnet.py): byte inputs, int8 weights at scale 2^6, clipped ReLU in
0..=127; the first layer trains on standardised inputs folded into byte weights; a float
fit, then quantisation-aware fine-tuning through the exact integer forward.

Reported on the held-out families only (indicative, never adoption evidence: offline fronts
did not transfer to encodes five times on 2026-11-22): the value's correlation, and the
share of the alternatives' gain kept when the lowest-valued stop points are stopped, against
the share of their work saved, beside the linear stop's own point (`stopped`) and a random
order.

usage: stop_train.py STOP.npz OUT_DIR [--seed N] [--epochs N] [--qat-epochs N]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch import nn

import intnet

HIDDEN = (32, 8)
SHIFTS = (6, 6)
OUTPUT_SCALE_LOG2 = 6
HEADS = 3
VALUE_SCALE = 1000.0
GAIN_FLOOR = 0.002
BATCH = 4096
LEARNING_RATE = 3e-3
QAT_LEARNING_RATE = 5e-4
#: Loss weights of the indicator and cost heads beside the value head.
HEAD_WEIGHTS = (1.0, 0.5, 0.5)
SHARES = (0.1, 0.25, 0.5)
TEST_VECTORS = 16
VECTOR_SEED = 2026
RUST_FILE = "stop_value_weights.rs"
DOC_WIDTH = 88


def targets(t):
    gain = np.clip(t["planned_cost"] - t["alt_cost"], 0.0, None)
    gain = np.where(np.isfinite(gain), gain, 0.0)
    r = gain / np.maximum(t["planned_cost"], 1e-9)
    w = t["alt_work"] / np.maximum(t["planned_work"], 1.0)
    return np.stack([np.log1p(VALUE_SCALE * r / (1.0 + w)), (r > GAIN_FLOOR).astype(np.float64),
                     np.log1p(w)], 1).astype(np.float32), gain, t["alt_work"]


def clip_weights(clips):
    _, inverse, counts = np.unique(clips, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return (w / w.mean()).astype(np.float32)


def straight_through(value, quantised):
    return value + (quantised - value).detach()


class QuantNet(nn.Module):
    """The integer network with float latent parameters (as eval/learned/quantise.py on the
    nn-lead branch): integer-scale forward, the first layer on standardised inputs folded
    into byte weights, weights clamped to int8 after each step."""

    def __init__(self, sizes, mean, std):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.scales = [float(1 << s) for s in (*SHIFTS, OUTPUT_SCALE_LOG2)]
        varies = std > 0
        self.register_buffer("input_mean", mean / intnet.ACTIVATION_MAX)
        self.register_buffer("input_spread", torch.where(varies, std / intnet.ACTIVATION_MAX, 1.0))
        with torch.no_grad():
            self.layers[0].weight.mul_(varies.float())

    def float_parameters(self, index):
        layer = self.layers[index]
        if index > 0:
            return layer.weight, layer.bias
        weight = layer.weight / self.input_spread
        return weight, layer.bias - weight @ self.input_mean

    def integer_parameters(self, index, quantised):
        scale = self.scales[index]
        weight, bias = self.float_parameters(index)
        weight, bias = weight * scale, bias * (scale * intnet.ACTIVATION_MAX)
        if quantised:
            weight, bias = straight_through(weight, weight.round()), straight_through(bias, bias.round())
        return weight, bias

    def forward(self, x, quantised):
        hidden, last = x, len(self.layers) - 1
        for index in range(last):
            weight, bias = self.integer_parameters(index, quantised)
            shifted = (hidden @ weight.T + bias) / self.scales[index]
            if quantised:
                shifted = straight_through(shifted, shifted.floor())
            hidden = shifted.clamp(0.0, float(intnet.ACTIVATION_MAX))
        weight, bias = self.integer_parameters(last, quantised)
        return (hidden @ weight.T + bias) / (self.scales[last] * intnet.ACTIVATION_MAX)

    def clamp_to_integer_range(self):
        with torch.no_grad():
            for index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
                limit = intnet.WEIGHT_MAX / scale * (self.input_spread if index == 0 else 1.0)
                layer.weight.copy_(layer.weight.clamp(-limit, limit))

    def to_intnet(self):
        """The rounded network with the value head only."""
        with torch.no_grad():
            rounded = [self.integer_parameters(i, True) for i in range(len(self.layers))]
        weights = [w.cpu().numpy() for w, _ in rounded]
        biases = [b.cpu().numpy() for _, b in rounded]
        weights[-1], biases[-1] = weights[-1][:1], biases[-1][:1]
        return intnet.IntNet(weights, biases, SHIFTS, OUTPUT_SCALE_LOG2)


def loss_of(out, y, w):
    value = (out[:, 0] - y[:, 0]) ** 2
    chance = nn.functional.binary_cross_entropy_with_logits(out[:, 1], y[:, 1], reduction="none")
    cost = (out[:, 2] - y[:, 2]) ** 2
    parts = HEAD_WEIGHTS[0] * value + HEAD_WEIGHTS[1] * chance + HEAD_WEIGHTS[2] * cost
    return (parts * w).mean()


def train(model, x, y, w, epochs, rate, quantised):
    optimiser = torch.optim.Adam(model.parameters(), lr=rate)
    steps = epochs * (len(x) // BATCH + 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, rate, total_steps=steps)
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), BATCH):
            rows = order[start:start + BATCH]
            loss = loss_of(model(x[rows], quantised), y[rows], w[rows])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            model.clamp_to_integer_range()


def front(order_key, gain, work):
    """Stopping the lowest `order_key` first: share of gain kept at each share of work saved."""
    order = np.argsort(order_key, kind="stable")
    saved = np.cumsum(work[order]) / work.sum()
    lost = np.cumsum(gain[order]) / gain.sum()
    return {f"saved_{s}": float(1.0 - np.interp(s, saved, lost)) for s in SHARES}, (saved, lost)


def linear_point(stopped, gain, work):
    return {"work_saved": float(work[stopped].sum() / work.sum()),
            "gain_kept": float(1.0 - gain[stopped].sum() / gain.sum())}


def at_share(curve, share):
    saved, lost = curve
    return float(1.0 - np.interp(share, saved, lost))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def wrap(text, indent):
    """`text` in lines of at most DOC_WIDTH; a list item's continuation lines are indented."""
    lines, line = [], indent
    for word in text.split():
        if len(line) + len(word) + 1 > DOC_WIDTH and line.strip():
            lines.append(line.rstrip())
            line = indent
        line += word + " "
    lines.append(line.rstrip())
    if lines[0].startswith("- ") and not indent:
        lines = lines[:1] + ["  " + l for l in lines[1:]]
    return lines


def rust_table(name, rust_type, length, rows, pad_to):
    body = "\n".join("    " + ", ".join(str(int(v)) for v in list(row) + [0] * (pad_to - len(row))) + ","
                     for row in rows)
    return f"#[rustfmt::skip]\npub(super) const {name}: [{rust_type}; {length}] = [\n{body}\n];\n"


def stop_doc(net, record):
    """The stop network table's header lines."""
    held = record["held_out"]
    return [
        "Weights of the stop network (GENERATED by `eval/learned/stop_train.py`: do not edit;",
        "regenerate).",
        "",
        *wrap(f"Network {' -> '.join(map(str, net.sizes))}, clipped `ReLU` in 0..=127, int8 weights, run by "
              "the nc-dsp network kernels (`nn_dense_clipped`, then `nn_dense_linear`); inputs in the "
              "order of `stop_inputs.rs`. `OUTPUT_ONE` output units are 1.0 of the value head's target, "
              f"ln(1 + {VALUE_SCALE:g} r / (1 + w)) (r: the alternatives' relative gain, w: their work "
              "over the planned candidate's); the indicator and cost heads were trained beside it and "
              "are not exported.", ""),
        "",
        *record_lines(record),
        *wrap(f"- Held-out families ({held['rows']} stop points): value correlation "
              f"{held['value_corr']:.3f}; gain kept at "
              + " / ".join(f"{100 * s:.0f}" for s in SHARES) + " % of the work saved: network "
              + " / ".join(f"{100 * held['network'][f'saved_{s}']:.1f}" for s in SHARES) + ", random "
              + " / ".join(f"{100 * s_:.1f}" for s_ in [1 - s for s in SHARES]) + "; the linear stop's "
              f"point: {100 * held['linear']['gain_kept']:.1f} % kept at "
              f"{100 * held['linear']['work_saved']:.1f} % saved (network there "
              f"{100 * held['network_at_linear']:.1f} %).", ""),
        *bounds_lines(net),
    ]


def record_lines(record):
    """The training run and data lines of a table's header."""
    return [
        *wrap(f"- Run: `{' '.join(record['argv'])}`, seed {record['seed']}, torch {record['torch']}.", ""),
        f"- Data: `{os.path.basename(record['data'])}`, sha256",
        f"  `{record['data_sha256']}`:",
        *wrap(record["dataset"], "  "),
    ]


def bounds_lines(net):
    return wrap(f"- Bounds: the largest reachable |sum| per layer is "
                f"{', '.join(map(str, intnet.largest_sums(net)))} (`eval/learned/intnet.py`).", "")


def rust_source(net, doc, vectors, outputs):
    """The Rust table: `doc` (header lines), sizes, shifts, weights and test vectors."""
    sizes = net.sizes
    pad = intnet.padded(max(sizes[:-1]))
    assert all(intnet.padded(width) == pad for width in sizes[:-1])
    parts = ["\n".join(("//! " + line).rstrip() for line in doc), "",
             f"pub(super) const INPUT_COUNT: usize = {sizes[0]};",
             f"pub(super) const PADDED: usize = {pad};"]
    hidden = range(1, len(sizes) - 1)
    parts += [f"pub(super) const HIDDEN_{k}: usize = {sizes[k]};" for k in hidden]
    parts += [f"pub(super) const SHIFT_{k}: u32 = {net.shifts[k - 1]};" for k in hidden]
    parts += [f"pub(super) const OUTPUT_ONE: i32 = {net.output_unit};", ""]
    for k in hidden:
        parts.append(rust_table(f"WEIGHTS_{k}", "i8", f"HIDDEN_{k} * PADDED", net.weights[k - 1], pad))
        parts.append(rust_table(f"BIAS_{k}", "i32", f"HIDDEN_{k}", [net.biases[k - 1]], 0))
    parts.append(rust_table("WEIGHTS_OUT", "i8", "PADDED", net.weights[-1], pad))
    parts.append(f"pub(super) const BIAS_OUT: i32 = {int(net.biases[-1][0])};\n")
    rows = "\n".join(f"    ([{', '.join(map(str, x))}], {int(y)})," for x, y in zip(vectors, outputs))
    parts.append(f"#[cfg(test)]\n#[rustfmt::skip]\npub(super) const TEST_VECTORS: [([u8; INPUT_COUNT], i32); "
                 f"{len(vectors)}] = [\n{rows}\n];")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--dataset", default="the on-policy stop dump")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    t = dict(np.load(args.data))
    y, gain, work = targets(t)
    fit, test = t["side"] == "train", t["side"] == "heldout"
    x = torch.tensor(t["inputs"][fit], dtype=torch.float32)
    sizes = (x.shape[1],) + HIDDEN + (HEADS,)
    model = QuantNet(sizes, x.mean(0), x.std(0))
    yt, wt = torch.tensor(y[fit]), torch.tensor(clip_weights(t["clip"][fit]))
    train(model, x, yt, wt, args.epochs, LEARNING_RATE, False)
    train(model, x, yt, wt, args.qat_epochs, QAT_LEARNING_RATE, True)
    net = model.to_intnet()
    bytes_test = t["inputs"][test]
    value = intnet.forward(net, bytes_test) / net.output_unit
    with torch.no_grad():
        in_torch = model(torch.tensor(bytes_test, dtype=torch.float32), True)[:, 0].numpy()
    assert np.array_equal(np.rint(in_torch.astype(np.float64) * net.output_unit),
                          intnet.forward(net, bytes_test)), "training forward != integer forward"
    g, wk = gain[test], work[test]
    curve_net = front(value, g, wk)
    linear = linear_point(t["stopped"][test], g, wk)
    held = {"rows": int(test.sum()), "value_corr": float(np.corrcoef(value, y[test, 0])[0, 1]),
            "network": curve_net[0], "linear": linear,
            "network_at_linear": at_share(curve_net[1], linear["work_saved"]),
            "per_class": {}}
    for c in np.unique(t["class"][test]):
        rows = t["class"][test] == c
        if g[rows].sum() > 0:
            held["per_class"][str(c)] = front(value[rows], g[rows], wk[rows])[0]
    record = {"argv": ["eval/learned/stop_train.py", *sys.argv[1:]], "seed": args.seed,
              "torch": torch.__version__, "data": args.data, "data_sha256": sha256(args.data),
              "dataset": args.dataset, "sizes": list(net.sizes), "held_out": held,
              "fit_rows": int(fit.sum())}
    rng = np.random.default_rng(VECTOR_SEED)
    vectors = np.concatenate([np.zeros((1, sizes[0]), np.uint8),
                              np.full((1, sizes[0]), intnet.ACTIVATION_MAX, np.uint8),
                              rng.integers(0, 128, (4, sizes[0]), dtype=np.uint8),
                              bytes_test[np.sort(rng.choice(len(bytes_test), TEST_VECTORS - 6, replace=False))]])
    outputs, hidden = intnet.forward(net, vectors, activations=True)
    intnet.save(os.path.join(args.out, "stop_net.npz"), net)
    np.save(os.path.join(args.out, "hidden_test.npy"), hidden[-1])
    with open(os.path.join(args.out, RUST_FILE), "w") as f:
        f.write(rust_source(net, stop_doc(net, record), vectors, outputs))
    with open(os.path.join(args.out, "stop_training.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps(held, indent=1))


if __name__ == "__main__":
    main()
