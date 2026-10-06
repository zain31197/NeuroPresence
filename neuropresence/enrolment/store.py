"""Where the enrolled picture is kept on disk.

    picture.png      the enrolled picture
    neutral.png      the face crop of that picture; it sets the neutral pose
    signature.npy    the face signature (ArcFace embedding), when it could be computed
    record.json      when it was enrolled, where from, and what the checks found

Plain files and one enrolment at a time: there is one user and one picture.
The record is written last, so a folder without it is an unfinished save and
is treated as empty.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PICTURE, NEUTRAL, SIGNATURE, RECORD = "picture.png", "neutral.png", "signature.npy", "record.json"


@dataclass
class Enrolment:
    id: str  # changes with every new enrolment
    enrolled_at: str  # local time, ISO format
    origin: str  # "camera" or "upload"
    checks: list  # what each check found, as dicts
    picture: np.ndarray
    neutral_face: np.ndarray
    signature: np.ndarray | None

    def summary(self):
        """Everything about the enrolment except the pictures themselves."""
        height, width = self.picture.shape[:2]
        return {
            "id": self.id,
            "enrolled_at": self.enrolled_at,
            "origin": self.origin,
            "width": width,
            "height": height,
            "checks": self.checks,
            "has_signature": self.signature is not None,
        }


class EnrolmentStore:
    def __init__(self, folder):
        self.folder = Path(folder)

    def load(self):
        """The stored enrolment, or None if there is none or it cannot be read."""
        try:
            record = json.loads((self.folder / RECORD).read_text())
            picture = cv2.imread(str(self.folder / PICTURE))
            neutral = cv2.imread(str(self.folder / NEUTRAL))
            if picture is None or neutral is None:
                return None
            signature = self.folder / SIGNATURE
            return Enrolment(
                id=record["id"],
                enrolled_at=record["enrolled_at"],
                origin=record["origin"],
                checks=record["checks"],
                picture=picture,
                neutral_face=neutral,
                signature=np.load(signature) if signature.exists() else None,
            )
        except (OSError, ValueError, KeyError):
            return None

    def save(self, picture, neutral_face, signature, origin, checks):
        """Replace whatever is stored with a new enrolment and return it."""
        self.remove()
        self.folder.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        enrolment = Enrolment(
            id=now.strftime("%Y%m%dT%H%M%S%f"),
            enrolled_at=now.isoformat(timespec="seconds"),
            origin=origin,
            checks=checks,
            picture=picture,
            neutral_face=neutral_face,
            signature=None if signature is None else np.asarray(signature, dtype=np.float32),
        )
        cv2.imwrite(str(self.folder / PICTURE), picture)
        cv2.imwrite(str(self.folder / NEUTRAL), neutral_face)
        if enrolment.signature is not None:
            np.save(self.folder / SIGNATURE, enrolment.signature)
        record = {"id": enrolment.id, "enrolled_at": enrolment.enrolled_at, "origin": origin, "checks": checks}
        temporary = self.folder / (RECORD + ".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        os.replace(temporary, self.folder / RECORD)
        return enrolment

    def remove(self):
        """Delete the stored enrolment. The record goes first, so a half-removed folder reads as empty."""
        for name in (RECORD, PICTURE, NEUTRAL, SIGNATURE):
            try:
                (self.folder / name).unlink()
            except FileNotFoundError:
                pass
