"""Build the face mesh drawn on the landing page.

The landing page shows a face made of the 478 landmarks the tracker follows.
So that it is nobody's face in particular, this script averages the landmarks
of front-facing frames from several of LivePortrait's sample videos and writes
the result, with the mesh edges, to web/src/data/faceMesh.json.

Usage:
    python scripts/make_face_mesh.py
"""

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture.tracker import DEFAULT_MODEL, rotation_to_euler_deg

VIDEOS = ROOT / "third_party" / "LivePortrait" / "assets" / "examples" / "driving"
OUT = ROOT / "web" / "src" / "data" / "faceMesh.json"
CLIPS = ["d0", "d3", "d6", "d9", "d10", "d13", "d18"]
MAX_ANGLE_DEG = 6.0
FRAME_STEP = 5


def normalise(points):
    points = points - points.mean(axis=0)
    return points / np.sqrt((points**2).sum(axis=1).mean())


def align(points, template):
    """Rotate points onto template (best rotation, no mirroring)."""
    u, _, vt = np.linalg.svd(points.T @ template)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return points @ (u @ vt)


def frontal_faces(landmarker):
    for clip in CLIPS:
        capture = cv2.VideoCapture(str(VIDEOS / f"{clip}.mp4"))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if index % FRAME_STEP:
                continue
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = landmarker.detect(image)
            if len(found.face_landmarks) != 1:
                continue
            matrix = np.asarray(found.facial_transformation_matrixes[0])
            if max(abs(a) for a in rotation_to_euler_deg(matrix[:3, :3])) > MAX_ANGLE_DEG:
                continue
            h, w = frame.shape[:2]
            # x and z are in units of the image width, y in units of its height.
            yield clip, np.array([(p.x * w, p.y * h, p.z * w) for p in found.face_landmarks[0]])
        capture.release()


def unique_edges(connections):
    return sorted({tuple(sorted((c.start, c.end))) for c in connections})


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        output_facial_transformation_matrixes=True,
    )
    per_clip = {}
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for clip, points in frontal_faces(landmarker):
            per_clip.setdefault(clip, []).append(normalise(points))
    if not per_clip:
        sys.exit("No front-facing frames found. Run scripts/setup_liveportrait.py first.")

    # One mean face per person first, so a long clip does not outweigh a short one.
    people = [normalise(np.mean(faces, axis=0)) for faces in per_clip.values()]
    mean = people[0]
    for _ in range(5):
        mean = normalise(np.mean([align(person, mean) for person in people], axis=0))

    edges = vision.FaceLandmarksConnections
    mesh = {
        "about": "Average of front-facing faces from sample videos. x right, y down, z negative toward the viewer.",
        "people": len(people),
        "points": np.round(mean, 4).tolist(),
        "tesselation": unique_edges(edges.FACE_LANDMARKS_TESSELATION),
        "contours": unique_edges(edges.FACE_LANDMARKS_CONTOURS),
        "irises": unique_edges(edges.FACE_LANDMARKS_LEFT_IRIS + edges.FACE_LANDMARKS_RIGHT_IRIS),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mesh, separators=(",", ":")))
    print(f"Wrote {OUT}: {len(mesh['points'])} points from {len(people)} people "
          f"({sum(len(f) for f in per_clip.values())} frames), {len(mesh['tesselation'])} mesh edges, "
          f"{len(mesh['contours'])} contour edges, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
