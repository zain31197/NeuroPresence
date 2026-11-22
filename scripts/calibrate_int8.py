"""Measure the ranges the 8-bit layers need (neuropresence/reenactment/int8_scales.json).

The generator's SPADE layers and the warping network's hourglass run in 8 bit (neuropresence/reenactment/quantize.py).
For that each of their convolutions needs to know the largest value that reaches it. This drives a handful of
LivePortrait's sample pictures with its sample clips, through the PyTorch networks, and keeps for each convolution
the largest value but the most extreme one in ten thousand, over every picture and clip. Only those public samples
are used: nothing of anyone who uses this system is in the file. It needs running again only if the networks change.

Usage:
    python scripts/calibrate_int8.py         # about two minutes, needs the GPU
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker
from neuropresence.capture.crop import square_face_crop
from neuropresence.reenactment import ReenactmentEngine, quantize
from neuropresence.reenactment.accelerate import _WarpedOnly

EXAMPLES = ROOT / "third_party" / "LivePortrait" / "assets" / "examples"
SOURCES = ("s1.jpg", "s9.jpg", "s10.jpg", "s11.jpg", "s12.jpg", "s22.jpg")
CLIPS = ("d0", "d3", "d6", "d9", "d10", "d11", "d12", "d13", "d18")
FRAMES_PER_PAIR = 24
EXTREME_SHARE = 1e-4  # of the values that reach a layer, the most extreme ones that are left out


def faces_of(clip):
    """Every few face crops of a driving clip, as the engine is given them."""
    cap, found, index = cv2.VideoCapture(str(clip)), [], 0
    with FaceTracker(video=True) as tracker:
        while len(found) < FRAMES_PER_PAIR:
            ok, frame = cap.read()
            if not ok:
                break
            track = tracker.process(frame)
            index += 1
            if track.ok and index % 3 == 0:
                found.append((square_face_crop(frame, track.bbox, scale=2.0), index / 30))
    cap.release()
    return found


def main():
    engine = ReenactmentEngine(tensorrt=False)
    wrapper = engine._wrapper
    networks = {"warping": _WarpedOnly(wrapper.warping_module.network).eval(),
                "generator": wrapper.spade_generator.network.eval()}
    largest, handles = {"warping": {}, "generator": {}}, []

    def watch(network, name):
        def see(_, inputs):
            values = inputs[0].detach().abs().flatten().float()
            kept = max(1, int(values.numel() * EXTREME_SHARE))
            value = float(torch.topk(values, kept).values[-1])
            largest[network][name] = max(value, largest[network].get(name, 0.0))
        return see

    for network, module in networks.items():
        for name, layer in module.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)) and quantize.wanted(network, name):
                handles.append(layer.register_forward_pre_hook(watch(network, name)))
    print({network: sum(1 for n, m in module.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)) and quantize.wanted(network, n))
           for network, module in networks.items()}, "layers to measure", flush=True)

    driven = 0
    for source in SOURCES:
        picture = cv2.imread(str(EXAMPLES / "source" / source))
        if picture is None:
            raise SystemExit(f"{source} is missing from LivePortrait's examples: run scripts/setup_liveportrait.py")
        try:
            engine.set_source(picture)
        except Exception as err:  # a picture without a clear face: the others are enough
            print(f"  {source}: skipped ({err})", flush=True)
            continue
        for clip in CLIPS:
            engine.reset_reference()
            for face, at in faces_of(EXAMPLES / "driving" / f"{clip}.mp4"):
                engine.drive(face, at=at, steady=False, limit=True)
                driven += 1
        print(f"  {source}: {driven} frames driven so far", flush=True)
    for handle in handles:
        handle.remove()
    if any(len(layers) < 5 for layers in largest.values()):
        raise SystemExit("Some layers were never reached.")

    report = {
        "note": "The largest value that reaches each layer run in 8 bit (neuropresence/reenactment/quantize.py). Measured by "
                "scripts/calibrate_int8.py on LivePortrait's sample pictures and clips only.",
        "made": datetime.now().isoformat(timespec="seconds"),
        "frames": driven,
        **{network: {name: float(f"{value:.4g}") for name, value in sorted(layers.items())} for network, layers in largest.items()},
    }
    quantize.SCALES_FILE.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"Written to {quantize.SCALES_FILE}: {len(largest['warping'])} warping and {len(largest['generator'])} generator layers, {driven} frames")


if __name__ == "__main__":
    main()
