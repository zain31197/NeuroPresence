"""Switches for the parts of the pipeline that can be turned on and off while it runs.

A new feature adds one entry to DEFAULTS. The web app lists whatever is
registered here, so the feature gets its switch with no front-end work, and
can be tested live with it on and off.
"""

import threading
from dataclasses import asdict, dataclass, replace


@dataclass
class Feature:
    key: str
    label: str
    description: str
    enabled: bool


DEFAULTS = (
    Feature(
        "reenactment",
        "Reenactment",
        "Animate the enrolled picture with the live face. When off, the output is the enrolled picture, unchanged.",
        True,
    ),
    Feature(
        "steady_crop",
        "Steady crop",
        "Hold the face crop still while the head is still, so the tracker's small errors are not read as movement.",
        True,
    ),
    Feature(
        "steady_keypoints",
        "Steady head",
        "Smooth the animation's keypoints while the face is nearly still, so a still head is drawn still. "
        "Movement passes straight through.",
        True,
    ),
    Feature(
        "natural_range",
        "Natural head range",
        "Keep the head within the range that looks right on a body that stays still. Small movements are followed "
        "exactly; a head thrown far back or turned far to the side eases to a stop, and a posture held for a few "
        "seconds becomes the new rest position.",
        True,
    ),
    Feature(
        "identity_guard",
        "Identity fallback",
        "Act when the output stops looking like your picture. A short dip is ignored; a lasting one takes a fresh "
        "neutral pose; and if that does not help, the still picture is shown until you resume.",
        True,
    ),
    Feature(
        "delay_watchdog",
        "Delay watchdog",
        "Show the still picture while the delay from camera to output stays above its limit, so that lips that "
        "move late are never shown.",
        True,
    ),
    Feature(
        "tracking_overlay",
        "Tracking overlay",
        "Draw the tracked face landmarks, and the window the face crop is cut from, on the camera preview.",
        False,
    ),
)


class FeatureSet:
    def __init__(self, features=DEFAULTS):
        self._features = {feature.key: replace(feature) for feature in features}
        self._lock = threading.Lock()

    def enabled(self, key):
        with self._lock:
            return self._features[key].enabled

    def set(self, key, enabled):
        """Turn one feature on or off. Raises KeyError for an unknown key."""
        with self._lock:
            self._features[key].enabled = bool(enabled)
            return asdict(self._features[key])

    def as_list(self):
        with self._lock:
            return [asdict(feature) for feature in self._features.values()]
