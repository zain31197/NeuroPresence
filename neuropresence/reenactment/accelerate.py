"""The two networks that draw the picture, run through TensorRT.

Measured on the RTX 5050 (8 October 2026), one network at a time, half precision in both cases:

    generator         PyTorch 55 ms   TensorRT 25 ms
    warping network   PyTorch 29 ms   TensorRT 14 ms

Converting takes about a minute and a half, so the converted engines are kept in
models/tensorrt and loaded from there the next time. An engine only fits the GPU
model and the TensorRT version it was built with, so both are part of its file name.

TensorRT is optional. If it is not installed, or a conversion fails, the caller
keeps the PyTorch networks.
"""

import hashlib
import logging
import time
from pathlib import Path

import torch

log = logging.getLogger("neuropresence.reenactment")
WORKSPACE_BYTES = 3 << 30
OPTIMIZATION_LEVEL = 3  # TensorRT's default. Level 5 was tried: the same speed (37.5 ms against 37.8) for a slower build


class TensorRTNetwork(torch.nn.Module):
    """A TensorRT engine that takes and gives PyTorch tensors, which never leave the GPU."""

    def __init__(self, plan, as_dict=None):
        """as_dict: None to return the one output as a tensor; a name to return {name: tensor};
        or True to return every output in a dict under its own name."""
        super().__init__()
        import tensorrt as trt

        self._types = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
        self._engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        if self._engine is None:
            raise RuntimeError("TensorRT could not load the engine")
        self._context = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(index) for index in range(self._engine.num_io_tensors)]
        self.inputs = [name for name in names if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self._outs = {}
        for name in names:
            if name not in self.inputs:
                self._outs[name] = torch.empty(tuple(self._engine.get_tensor_shape(name)),
                                               dtype=self._types[self._engine.get_tensor_dtype(name)], device="cuda")
                self._context.set_tensor_address(name, self._outs[name].data_ptr())
        self._as_dict = as_dict  # the warping network's callers expect {"out": tensor}

    def forward(self, *args, **kwargs):
        given = dict(zip(self.inputs, args), **kwargs)
        held = []  # the tensors must stay alive until the engine has run
        for name in self.inputs:
            tensor = given[name].to(self._types[self._engine.get_tensor_dtype(name)]).contiguous()
            held.append(tensor)
            self._context.set_tensor_address(name, tensor.data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT failed to run the network")
        # Copies: the engine writes over its own outputs on the next call.
        copies = {name: out.to(torch.float32, copy=True) for name, out in self._outs.items()}
        if self._as_dict is True:
            return copies
        out = next(iter(copies.values()))
        return {self._as_dict: out} if self._as_dict else out


class _WarpedOnly(torch.nn.Module):
    """The warping network with only the output the generator needs, so it converts as one tensor out."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, feature, kp_source, kp_driving):
        return self.network(feature, kp_source=kp_source, kp_driving=kp_driving)["out"]


class _AsTuple(torch.nn.Module):
    """A network that returns a dict, made to return its values in a fixed order so each converts as a named output."""

    def __init__(self, network, keys):
        super().__init__()
        self.network, self.keys = network, keys

    def forward(self, image):
        out = self.network(image)
        return tuple(out[key] for key in self.keys)


def _build(module, example, input_names, opset, onnx_path, output_names=("out",)):
    """Export a network to ONNX and build a TensorRT engine that keeps the precision it was exported in."""
    import tensorrt as trt

    torch.onnx.export(module, example, str(onnx_path), input_names=input_names, output_names=list(output_names),
                      opset_version=opset, dynamo=False)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))[:300])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.builder_optimization_level = OPTIMIZATION_LEVEL
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT could not build the engine")
    return bytes(plan)


def _plan(name, make, cache_dir, key):
    """The engine for one network: from the cache if it was built for this GPU and TensorRT, else built now."""
    path = Path(cache_dir) / f"{name}-{key}.plan"
    if path.exists():
        return path.read_bytes()
    started = time.perf_counter()
    plan = make(path.with_suffix(".onnx"))
    path.write_bytes(plan)
    path.with_suffix(".onnx").unlink(missing_ok=True)
    log.info("Converted the %s to TensorRT in %.0f s", name, time.perf_counter() - started)
    return plan


def to_tensorrt(warping, generator, cache_dir):
    """TensorRT versions of the half-precision warping network and generator, as (warping, generator).

    Raises if TensorRT is missing or a conversion fails.
    """
    import tensorrt as trt

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in warping.parameters())
    key = hashlib.md5(f"{trt.__version__}|{torch.cuda.get_device_name(0)}|{weights}|{OPTIMIZATION_LEVEL}".encode()).hexdigest()[:10]

    feature = torch.zeros(1, 32, 16, 64, 64, dtype=torch.float16, device="cuda")
    kp_source = torch.rand(1, 21, 3, dtype=torch.float16, device="cuda") * 0.2
    kp_driving = kp_source * 1.01
    with torch.no_grad():
        warp_only = _WarpedOnly(warping).eval()
        warped = warp_only(feature, kp_source, kp_driving)
        # Opset 20 is the first with the 3-D grid sampling the warping network uses.
        warp_plan = _plan("warping", lambda onnx: _build(warp_only, (feature, kp_source, kp_driving),
                                                         ["feature", "kp_source", "kp_driving"], 20, onnx), cache_dir, key)
        generator_plan = _plan("generator", lambda onnx: _build(generator.eval(), (warped,), ["feature"], 17, onnx),
                               cache_dir, key)
    return TensorRTNetwork(warp_plan, as_dict="out"), TensorRTNetwork(generator_plan)


def motion_to_tensorrt(network, cache_dir):
    """A TensorRT version of the network that reads the movement, in full precision.

    Full precision on purpose: in half precision its readings were noisier and the head
    trembled again (see engine.py). Raises if TensorRT is missing or the conversion fails.
    """
    import tensorrt as trt

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights = sum(p.numel() for p in network.parameters())
    key = hashlib.md5(f"{trt.__version__}|{torch.cuda.get_device_name(0)}|{weights}|fp32|{OPTIMIZATION_LEVEL}".encode()).hexdigest()[:10]
    image = torch.rand(1, 3, 256, 256, device="cuda")
    with torch.no_grad():
        keys = list(network(image).keys())
        plan = _plan("motion", lambda onnx: _build(_AsTuple(network, keys).eval(), (image,), ["image"], 17, onnx, keys),
                     cache_dir, key)
    return TensorRTNetwork(plan, as_dict=True)
