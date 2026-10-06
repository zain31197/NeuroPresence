"""The measurable targets the project committed to (proposal section 4.3)."""

TARGETS = {
    "fps": 24.0,  # sustained output frame rate, at least
    "render_ms": 42.0,  # GPU time of the reenactment stage per frame, at most
    "end_to_end_ms": 150.0,  # captured frame to output frame, at most
    "vram_gb": 8.0,  # peak GPU memory, at most
    "csim": 0.80,  # identity similarity between source and output, at least
}
