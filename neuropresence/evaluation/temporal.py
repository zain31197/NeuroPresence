"""Temporal stability measurements on a sequence of frames.

Two measurements, because they catch different faults:

* warping error catches flicker - colour or texture that changes from frame
  to frame in a way no movement explains;
* landmark jitter catches wobble - the face itself trembling by a pixel or
  two, which is real movement and therefore invisible to the warping error.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def backward_warp(image, flow):
    """Sample image at every pixel's own position plus flow.

    image: (N, C, H, W). flow: (N, 2, H, W) in pixels, channel 0 = x, 1 = y.
    If flow tells, for each pixel of frame B, where that point was in frame A,
    then backward_warp(A, flow) rebuilds B. Samples that fall outside the
    image come back as zero.
    """
    _, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    x, y = xs + flow[:, 0], ys + flow[:, 1]
    grid = torch.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def visibility_mask(flow_to_prev, flow_from_prev):
    """True for pixels of the current frame that are also visible in the previous one.

    flow_to_prev is defined on the current frame (current -> previous) and
    flow_from_prev on the previous frame (previous -> current). Where a point
    is visible in both, going there and back returns to the start, so the two
    flows cancel. Where they do not cancel the pixel is hidden in one frame,
    or the flow is wrong, and it is left out (forward-backward check of
    Sundaram et al., 2010).
    """
    _, _, h, w = flow_to_prev.shape
    back = backward_warp(flow_from_prev, flow_to_prev)
    mismatch = ((flow_to_prev + back) ** 2).sum(dim=1)
    size = (flow_to_prev**2).sum(dim=1) + (back**2).sum(dim=1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow_to_prev.device, dtype=flow_to_prev.dtype),
        torch.arange(w, device=flow_to_prev.device, dtype=flow_to_prev.dtype),
        indexing="ij",
    )
    x, y = xs + flow_to_prev[:, 0], ys + flow_to_prev[:, 1]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return (mismatch < 0.01 * size + 0.5) & inside


def masked_mse(a, b, mask):
    """Mean squared difference per image over the pixels where mask is True."""
    error = ((a - b) ** 2).mean(dim=1)
    count = mask.sum(dim=(1, 2)).clamp(min=1)
    return (error * mask).sum(dim=(1, 2)) / count


class WarpingError:
    """Flow-compensated difference between consecutive frames (Lai et al., 2018).

    For each pair of consecutive frames the previous frame is moved onto the
    current one along the optical flow (RAFT, from torchvision) and the mean
    squared difference is taken over the pixels visible in both, with pixel
    values in 0..1. Zero means every change between frames is explained by
    movement; flicker raises it.
    """

    def __init__(self, device=None):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self._raft = self._raft.eval().to(self.device)

    def _to_tensor(self, frames_bgr):
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        return torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float() / 255.0

    def _flow(self, a, b):
        return self._raft(2 * a - 1, 2 * b - 1)[-1]  # RAFT expects -1..1; last refinement

    @torch.no_grad()
    def pair_errors(self, previous_bgr, current_bgr, batch_size=8):
        """One error per (previous, current) frame pair. Frames: HxWx3 uint8, same size."""
        if len(previous_bgr) != len(current_bgr):
            raise ValueError("previous and current must have the same number of frames")
        errors = []
        for i in range(0, len(previous_bgr), batch_size):
            prev = self._to_tensor(previous_bgr[i:i + batch_size])
            cur = self._to_tensor(current_bgr[i:i + batch_size])
            if prev.shape[-2] % 8 or prev.shape[-1] % 8:
                raise ValueError(f"Frame height and width must be multiples of 8, got {tuple(prev.shape[-2:])}")
            to_prev, from_prev = self._flow(cur, prev), self._flow(prev, cur)
            mask = visibility_mask(to_prev, from_prev)
            errors.append(masked_mse(cur, backward_warp(prev, to_prev), mask).cpu().numpy())
        return np.concatenate(errors) if errors else np.zeros(0)

    def __call__(self, frames_bgr):
        """Mean warping error over a clip of consecutive frames."""
        if len(frames_bgr) < 2:
            raise ValueError("Need at least two frames.")
        return float(self.pair_errors(frames_bgr[:-1], frames_bgr[1:]).mean())


def landmark_jitter(points, scale):
    """Mean frame-to-frame acceleration of tracked points, as a fraction of scale.

    points: (T, K, 2) positions of K points over T consecutive frames, with
    NaN rows for frames where the face was not found. scale: a reference
    length in the same units (for example the distance between the outer eye
    corners), one value or one per frame.

    Acceleration is p[t+1] - 2 p[t] + p[t-1]. Natural head movement is smooth
    from one video frame to the next, so its acceleration is close to zero;
    what remains is jitter. Frames next to a gap are skipped.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(f"points must have shape (T, K, 2), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("Need at least three frames.")
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), (points.shape[0],))
    acceleration = points[2:] - 2 * points[1:-1] + points[:-2]
    per_frame = np.linalg.norm(acceleration, axis=2).mean(axis=1) / scale[1:-1]
    per_frame = per_frame[np.isfinite(per_frame)]
    if per_frame.size == 0:
        raise ValueError("No three consecutive frames with a tracked face.")
    return float(per_frame.mean())
