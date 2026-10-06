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
        "tracking_overlay",
        "Tracking overlay",
        "Draw the tracked face landmarks on the camera preview.",
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
