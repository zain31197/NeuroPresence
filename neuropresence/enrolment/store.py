"""Where the enrolment is kept on disk. It has two parts.

The face, taken live from the camera. Only its signature is kept, never the picture:

    identity.npy     the face signature (ArcFace embedding) of the person in front of the camera
    identity.json    when it was taken and what the checks found

The meeting picture, uploaded, and accepted only if its face matches that signature:

    picture.png      the picture that is animated
    neutral.png      the face crop of that picture
    signature.npy    the picture's own face signature, for scoring the output against it
    record.json      when it was enrolled, what the checks found, and how well it matched the face

Plain files and one of each: there is one user. A record is written last, so a
folder without it is an unfinished save and is treated as empty.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .checks import POSE_KEYS

PICTURE, NEUTRAL, SIGNATURE, RECORD = "picture.png", "neutral.png", "signature.npy", "record.json"
IDENTITY, IDENTITY_RECORD = "identity.npy", "identity.json"
POSES_RECORD = "poses.json"  # which poses are registered and what each measured, beside their own pose_<key>.npy


@dataclass
class FaceIdentity:
    """Who the user is: the signature of the face that sat in front of the camera, and,
    once registered, how they look turned to each side and tilted up and down.

    poses maps "left"/"right"/"up"/"down" to {"signature", "angle_deg", "captured_at", "checks"}.
    A picture (live or uploaded) is compared against whichever of these looks most like it, not
    only the front-on signature; and angle_deg across the poses is what calibrates the
    reenactment engine's own range of motion to this person (see reenactment/engine.py).
    """

    id: str  # changes every time the face is verified again
    verified_at: str  # local time, ISO format
    checks: list  # what each check found on the frame it was taken from, as dicts
    signature: np.ndarray
    poses: dict = field(default_factory=dict)

    def summary(self):
        """Everything about it except the signatures themselves."""
        return {
            "id": self.id,
            "verified_at": self.verified_at,
            "checks": self.checks,
            "poses": {pose: {"captured_at": p["captured_at"], "angle_deg": p["angle_deg"]} for pose, p in self.poses.items()},
        }


@dataclass
class Enrolment:
    id: str  # changes with every new enrolment
    enrolled_at: str  # local time, ISO format
    origin: str  # "upload"; "camera" for a picture enrolled before the face and the picture were separated
    checks: list  # what each check found, as dicts
    picture: np.ndarray
    neutral_face: np.ndarray
    signature: np.ndarray | None
    match: float | None = None  # how alike its face is to the verified face (CSIM); None for an older record

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
            "match": self.match,
        }


class EnrolmentStore:
    def __init__(self, folder):
        self.folder = Path(folder)

    # ------------------------------------------------- the meeting picture

    def load(self):
        """The stored meeting picture, or None if there is none or it cannot be read."""
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
                match=record.get("match"),
            )
        except (OSError, ValueError, KeyError):
            return None

    def save(self, picture, neutral_face, signature, origin, checks, match=None):
        """Replace the stored meeting picture with a new one and return it."""
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
            match=match,
        )
        cv2.imwrite(str(self.folder / PICTURE), picture)
        cv2.imwrite(str(self.folder / NEUTRAL), neutral_face)
        if enrolment.signature is not None:
            np.save(self.folder / SIGNATURE, enrolment.signature)
        record = {"id": enrolment.id, "enrolled_at": enrolment.enrolled_at, "origin": origin, "checks": checks,
                  "match": match}
        self._write_json(RECORD, record)
        return enrolment

    def remove(self):
        """Delete the stored meeting picture. The record goes first, so a half-removed folder reads as empty."""
        self._unlink(RECORD, PICTURE, NEUTRAL, SIGNATURE)

    # ------------------------------------------------------------ the face

    def load_identity(self):
        """The stored face signature, or None.

        A picture enrolled from the camera before the two were separated carries
        its own signature. That signature was taken live, so it becomes the face.
        """
        try:
            record = json.loads((self.folder / IDENTITY_RECORD).read_text())
            face = FaceIdentity(record["id"], record["verified_at"], record["checks"], np.load(self.folder / IDENTITY))
            face.poses = self._load_poses()
            return face
        except (OSError, ValueError, KeyError):
            pass
        older = self.load()
        if older is None or older.origin != "camera" or older.signature is None:
            return None
        return self._write_identity(FaceIdentity(older.id, older.enrolled_at, older.checks, older.signature))

    def save_identity(self, signature, checks):
        """Replace the stored face signature and return it. Any registered poses are
        for this face specifically, so a newly verified face starts with none."""
        self._unlink(POSES_RECORD, *(self._pose_file(pose) for pose in POSE_KEYS))
        now = datetime.now()
        return self._write_identity(FaceIdentity(now.strftime("%Y%m%dT%H%M%S%f"), now.isoformat(timespec="seconds"),
                                                 checks, np.asarray(signature, dtype=np.float32)))

    def remove_identity(self):
        self._unlink(IDENTITY_RECORD, IDENTITY, POSES_RECORD, *(self._pose_file(pose) for pose in POSE_KEYS))

    # ------------------------------------------------------------- poses

    def save_pose(self, pose, signature, angle_deg, checks):
        """Add or replace one registered pose ("left", "right", "up" or "down") for the
        current face. Raises ValueError if no face has been verified yet."""
        if pose not in POSE_KEYS:
            raise ValueError(f"Unknown pose: {pose}")
        if not (self.folder / IDENTITY).exists():
            raise ValueError("Verify your face before registering a pose.")
        now = datetime.now()
        np.save(self.folder / self._pose_file(pose), np.asarray(signature, dtype=np.float32))
        record = self._read_json(POSES_RECORD) or {}
        record[pose] = {"captured_at": now.isoformat(timespec="seconds"), "angle_deg": float(angle_deg), "checks": checks}
        self._write_json(POSES_RECORD, record)
        return record[pose]

    def _load_poses(self):
        record = self._read_json(POSES_RECORD) or {}
        poses = {}
        for pose, meta in record.items():
            path = self.folder / self._pose_file(pose)
            if pose in POSE_KEYS and path.exists():
                poses[pose] = {**meta, "signature": np.load(path)}
        return poses

    @staticmethod
    def _pose_file(pose):
        return f"pose_{pose}.npy"

    # ------------------------------------------------------------- helpers

    def _write_identity(self, face):
        self.folder.mkdir(parents=True, exist_ok=True)
        np.save(self.folder / IDENTITY, face.signature)
        self._write_json(IDENTITY_RECORD, face.summary())
        return face

    def _write_json(self, name, content):
        temporary = self.folder / (name + ".tmp")
        temporary.write_text(json.dumps(content, indent=2))
        os.replace(temporary, self.folder / name)

    def _read_json(self, name):
        try:
            return json.loads((self.folder / name).read_text())
        except (OSError, ValueError):
            return None

    def _unlink(self, *names):
        for name in names:
            try:
                (self.folder / name).unlink()
            except FileNotFoundError:
                pass
