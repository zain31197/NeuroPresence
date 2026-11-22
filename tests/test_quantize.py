"""The 8-bit layers of the networks that draw: the ranges file, the names of the layers, and the ONNX they are put into.
No GPU, TensorRT or LivePortrait weights are needed."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from neuropresence.reenactment import quantize  # noqa: E402


def test_the_ranges_file_names_only_the_layers_that_run_in_8_bit():
    scales = quantize.load_scales()
    assert set(scales) == {"warping", "generator"}
    assert len(scales["generator"]) == 45 and len(scales["warping"]) == 11
    for network, layers in scales.items():
        assert all(quantize.wanted(network, name) for name in layers)
        assert all(0.0 < value < 1e4 for value in layers.values())


def test_the_key_changes_with_the_ranges(monkeypatch, tmp_path):
    before = quantize.key()
    changed = tmp_path / "other.json"
    changed.write_bytes(quantize.SCALES_FILE.read_bytes() + b"\n")
    monkeypatch.setattr(quantize, "SCALES_FILE", changed)
    assert quantize.key() != before


def test_a_damaged_ranges_file_is_refused(monkeypatch, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"warping": {}, "generator": {}}', encoding="utf-8")
    monkeypatch.setattr(quantize, "SCALES_FILE", empty)
    with pytest.raises(ValueError):
        quantize.load_scales()
    monkeypatch.setattr(quantize, "SCALES_FILE", tmp_path / "missing.json")
    with pytest.raises(OSError):
        quantize.load_scales()


@pytest.mark.parametrize("exported, name", [
    ("/up_1/norm_s/mlp_shared/mlp_shared.0/Conv", "up_1.norm_s.mlp_shared.0"),
    ("/up_0/norm_0/mlp_gamma/Conv", "up_0.norm_0.mlp_gamma"),
    ("/network/dense_motion_network/hourglass/encoder/down_blocks.0/conv/Conv",
     "network.dense_motion_network.hourglass.encoder.down_blocks.0.conv"),
    ("/fc/Conv", "fc"),
])
def test_an_exported_convolution_is_found_by_its_name_in_the_network(exported, name):
    assert quantize.layer_name(exported) == name


def test_only_the_spade_layers_and_the_hourglass_run_in_8_bit():
    assert quantize.wanted("generator", "up_1.norm_0.mlp_beta") and not quantize.wanted("generator", "up_1.conv_0")
    assert not quantize.wanted("generator", "up_0.norm_0.mlp_beta")  # in 8 bit these brought flicker
    assert quantize.wanted("warping", "network.dense_motion_network.hourglass.decoder.conv")
    assert not quantize.wanted("warping", "network.dense_motion_network.mask")  # it decides where things move


def network_file(path, names=("/up_0/norm_0/mlp_gamma/Conv", "/up_0/conv_0/Conv")):
    """A model of two convolutions in 16 bit, with plain weights, named as an exporter would name them."""
    rng = np.random.default_rng(3)
    nodes, initializers, previous = [], [], "x"
    for index, name in enumerate(names):
        weight = f"w{index}"
        initializers.append(numpy_helper.from_array(rng.normal(size=(4, 4, 3, 3)).astype(np.float16), name=weight))
        nodes.append(helper.make_node("Conv", [previous, weight], [f"y{index}"], name=name, pads=[1, 1, 1, 1]))
        previous = f"y{index}"
    graph = helper.make_graph(nodes, "two", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4, 8, 8])],
                              [helper.make_tensor_value_info(previous, TensorProto.FLOAT16, [1, 4, 8, 8])], initializers)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), str(path))
    return path


def test_the_named_convolution_is_put_in_8_bit_and_the_other_is_left(tmp_path):
    source, out = network_file(tmp_path / "in.onnx"), tmp_path / "out.onnx"
    assert quantize.quantize_onnx(source, out, {"up_0.norm_0.mlp_gamma": 2.0}) == 1
    model = onnx.load(str(out))
    kinds = [node.op_type for node in model.graph.node]
    assert kinds.count("Conv") == 2 and kinds.count("QuantizeLinear") == 1 and kinds.count("DequantizeLinear") == 2
    quantized = next(node for node in model.graph.node if node.op_type == "Conv" and node.name == "/up_0/norm_0/mlp_gamma/Conv")
    assert "_x" in quantized.input[0] and "_w" in quantized.input[1]
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    assert any(name.endswith("_w8") and values.dtype == np.int8 and np.abs(values).max() <= 127 for name, values in initializers.items())
    assert any(name.endswith("_s") and float(values) == pytest.approx(2.0 / 127.0, rel=0.01) for name, values in initializers.items())
    assert "w0" not in initializers and "w1" in initializers  # the quantized weights replaced the plain ones


def test_a_network_that_came_out_differently_is_refused(tmp_path):
    source = network_file(tmp_path / "in.onnx")
    with pytest.raises(ValueError, match="were not found"):
        quantize.quantize_onnx(source, tmp_path / "out.onnx", {"up_0.norm_0.mlp_gamma": 2.0, "up_9.norm_0.mlp_gamma": 2.0})


def test_a_tiny_range_is_not_made_finer_than_16_bit_can_hold(tmp_path):
    source, out = network_file(tmp_path / "in.onnx"), tmp_path / "out.onnx"
    quantize.quantize_onnx(source, out, {"up_0.norm_0.mlp_gamma": 1e-9})
    scale = next(numpy_helper.to_array(t) for t in onnx.load(str(out)).graph.initializer if t.name.endswith("_s"))
    assert float(scale) >= quantize.LEAST / 127.0 * 0.99


def test_the_spectral_norm_is_folded_into_the_weights():
    torch = pytest.importorskip("torch")
    from torch.nn.utils import spectral_norm

    network = torch.nn.Sequential(spectral_norm(torch.nn.Conv2d(3, 4, 3, padding=1)), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 1)).eval()
    with torch.no_grad():
        network(torch.rand(1, 3, 8, 8))  # settles the norm
    folded = quantize.fold_spectral_norm(network)
    assert not any("weight_orig" in name for name, _ in folded.named_parameters())
    x = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(folded(x), network(x), atol=1e-5)
    assert any("weight_orig" in name for name, _ in network.named_parameters())  # the original is not touched
