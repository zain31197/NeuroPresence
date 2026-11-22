"""Some layers of the two networks that draw the picture, run in 8 bit.

The card is at its power limit while it draws (130 W on the RTX 5050), so every millisecond the networks take
off a frame is a frame a meeting's own work can have. Measured on 22 November 2026, one network at a time:

    generator, its SPADE layers but up_0's (45 convolutions)  24.5 -> 21.6 ms
    warping network, the hourglass (11 convolutions)          14.4 -> 11.6 ms

The layers that decide where things move, and the residual blocks' own convolutions, stay in 16 bit: in 8 bit they
broke the picture (worst frames at 23 dB and 17 dB). What was kept is judged by the full benchmark, not by those
figures (results/benchmark_int8_windows_rtx5050.json against benchmark_half_windows_rtx5050.json, 22 November 2026):
drawing one frame's two networks went from 35.4 to 30.9 ms; identity 0.896 to 0.894, mouth agreement 0.948 to 0.945,
pose and expression error unchanged; flicker 0.81 to 0.87 of real video (at most 1 allowed) and tremble 1.03 to 1.06
(at most 1 allowed: it was missed before this too). The speed is a trade for that small rise, not free.

The largest value that reaches each convolution is kept in int8_scales.json. It is measured by
scripts/calibrate_int8.py on LivePortrait's sample clips and pictures only, so it has no one's face in it, and it
depends only on the networks' weights, so every machine builds the same 8-bit networks. Anything that does not fit
(a missing file, a layer whose name is not found) raises, and the engine then uses its 16-bit networks.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCALES_FILE = Path(os.environ.get("NEUROPRESENCE_INT8_SCALES") or Path(__file__).with_name("int8_scales.json"))
LEAST = 1e-2  # a smaller largest value than this makes a step too fine for a 16-bit scale


def load_scales():
    """The layers to run in 8 bit and the largest value each receives: {"warping": {...}, "generator": {...}}."""
    found = json.loads(SCALES_FILE.read_text(encoding="utf-8"))
    scales = {network: dict(found.get(network, {})) for network in ("warping", "generator")}
    if not any(scales.values()) or not all(v > 0 for layers in scales.values() for v in layers.values()):
        raise ValueError(f"{SCALES_FILE.name} has no ranges")
    return scales


def key():
    """Changes with the ranges, so that networks built with other ranges are not taken for these."""
    return hashlib.md5(SCALES_FILE.read_bytes()).hexdigest()[:8]


def wanted(network, name):
    """Is the convolution called `name` in the network one that runs in 8 bit?

    The generator's SPADE layers except those of up_0, which brought the flicker (the benchmark's warping error against
    real video: 0.81 in 16 bit, 0.85 without up_0's, 0.99 with them, at most 1 allowed); and the warping network's hourglass.
    """
    if network == "generator":
        return ".norm" in name and not name.startswith("up_0.")
    return "hourglass" in name


def layer_name(node_name):
    """The PyTorch name of an exported convolution. The exporter writes /up_1/norm_s/mlp_shared/mlp_shared.0/Conv for
    up_1.norm_s.mlp_shared.0: a Sequential's own level and its item's are the same place."""
    parts = []
    for part in node_name.strip("/").removesuffix("/Conv").split("/"):
        if parts and part.startswith(parts[-1] + "."):
            parts[-1] = part
        else:
            parts.append(part)
    return ".".join(parts)


def fold_spectral_norm(network):
    """A copy of the network with each spectral norm computed once, so its weights are plain numbers to quantize."""
    from torch.nn import Conv2d
    from torch.nn.utils import remove_spectral_norm

    folded = copy.deepcopy(network)
    for module in folded.modules():
        if isinstance(module, Conv2d) and hasattr(module, "weight_orig"):
            remove_spectral_norm(module)
    return folded


def quantize_onnx(path, out, scales):
    """Wrap each convolution named in `scales` in an 8-bit quantization, weights per channel, and write the result.

    The model stays in 16 bit everywhere else. Returns how many convolutions were wrapped; every one in `scales` has
    to be found, because a network that came out differently from the one that was measured must not be trusted.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    nodes, used, done = [], set(), set()
    for node in graph.node:
        name = layer_name(node.name) if node.op_type == "Conv" else None
        if name not in scales:
            nodes.append(node)
            continue
        weight = initializers.get(node.input[1])
        if weight is None:
            raise ValueError(f"the weights of {name} are not plain numbers")
        amount = np.float16(max(float(scales[name]), LEAST) / 127.0)
        values = numpy_helper.to_array(weight).astype(np.float32)
        steps = np.maximum(np.abs(values).reshape(values.shape[0], -1).max(axis=1), 1e-6) / 127.0
        shape = (-1,) + (1,) * (values.ndim - 1)
        whole = np.clip(np.round(values / steps.reshape(shape)), -127, 127).astype(np.int8)
        base = f"int8_{len(done)}_{name.replace('.', '_')}"
        for suffix, data in (("s", np.array(amount, dtype=np.float16)), ("z", np.array(0, dtype=np.int8)), ("w8", whole),
                             ("ws", steps.astype(np.float16)), ("wz", np.zeros(values.shape[0], dtype=np.int8))):
            graph.initializer.append(numpy_helper.from_array(data, name=f"{base}_{suffix}"))
        nodes += [
            helper.make_node("QuantizeLinear", [node.input[0], f"{base}_s", f"{base}_z"], [f"{base}_xq"], name=f"{base}_q"),
            helper.make_node("DequantizeLinear", [f"{base}_xq", f"{base}_s", f"{base}_z"], [f"{base}_x"], name=f"{base}_dq"),
            helper.make_node("DequantizeLinear", [f"{base}_w8", f"{base}_ws", f"{base}_wz"], [f"{base}_w"], name=f"{base}_dqw", axis=0),
            helper.make_node("Conv", [f"{base}_x", f"{base}_w", *node.input[2:]], list(node.output), name=node.name,
                             **{attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}),
        ]
        used.add(weight.name)
        done.add(name)
    missing = set(scales) - done
    if missing:
        raise ValueError(f"{len(missing)} of the layers to run in 8 bit were not found, such as {sorted(missing)[0]}")
    del graph.node[:]
    graph.node.extend(nodes)
    still = {name for node in graph.node for name in node.input}
    for tensor in list(graph.initializer):
        if tensor.name in used and tensor.name not in still:
            graph.initializer.remove(tensor)
    onnx.save(model, str(out))
    return len(done)
