"""How alike must two faces be to count as the same person? A measurement for the upload check.

An uploaded meeting picture is accepted only if its face matches the face signature
taken live from the camera. This script measures the similarity (cosine of ArcFace
embeddings, the project's CSIM) for pairs that are the same person and pairs that are
not, on the sample material, and reports how many of each a given limit lets through.

Same person: frames of one sample clip, far apart in time (pose and expression differ).
Different people: one frame from each of two different clips or sample photographs.

Usage:
    python scripts/study_same_person.py
"""

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker
from neuropresence.identity import IdentityScorer
from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR

SAMPLES = LIVEPORTRAIT_DIR / "assets" / "examples"
FRAMES_PER_CLIP = 10
LIMITS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# Looked at by eye after a first run scored them as alike: these sample pictures are dogs and a monkey,
# not people, and the photograph s11 shows the child of clip d11, so the two are one person.
NOT_PEOPLE = {"s22", "s25", "s30", "s31", "s32"}
SAME_PERSON = {"s11": "d11"}
OUT = ROOT / "results" / "same_person_study.json"


def clip_faces(path, tracker, scorer):
    """Embeddings of frames spread over a clip, where exactly one face is tracked."""
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    found = []
    for index in np.linspace(0, max(total - 1, 0), FRAMES_PER_CLIP).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        track = tracker.process(frame) if ok else None
        if track is not None and track.ok:
            found.append(scorer.embed(frame, track.landmarks))
    capture.release()
    return found


def main():
    people = {}  # name -> embeddings of that one person
    with FaceTracker(video=False) as tracker:
        scorer = IdentityScorer(tracker=tracker)
        for path in sorted((SAMPLES / "driving").glob("*.mp4")):
            faces = clip_faces(path, tracker, scorer)
            if len(faces) >= 2:
                people[path.stem] = faces
        for path in sorted((SAMPLES / "source").glob("*")):
            image = cv2.imread(str(path)) if path.suffix.lower() in (".jpg", ".jpeg", ".png") else None
            track = tracker.process(image) if image is not None else None
            if track is None or not track.ok or path.stem in NOT_PEOPLE:
                continue
            # A second picture of someone already seen joins that person: a same-person pair from another camera.
            people.setdefault(SAME_PERSON.get(path.stem, path.stem), []).append(scorer.embed(image, track.landmarks))

    same = [float(a @ b) for faces in people.values() for a, b in itertools.combinations(faces, 2)]
    # One face per person, so each pair of people counts once.
    different = sorted(((float(people[a][0] @ people[b][0]), a, b) for a, b in itertools.combinations(people, 2)), reverse=True)
    scores = np.array([score for score, _, _ in different])

    print(f"{len(people)} people; {len(same)} same-person pairs; {len(scores)} different-person pairs")
    print(f"same person:      lowest {min(same):.3f}  5th percentile {np.percentile(same, 5):.3f}  median {np.median(same):.3f}")
    print(f"different people: median {np.median(scores):.3f}  95th {np.percentile(scores, 95):.3f}  99th {np.percentile(scores, 99):.3f}  highest {scores.max():.3f}")
    print("most alike different-person pairs (check these by eye: two clips can show one person):")
    for score, a, b in different[:8]:
        print(f"  {score:.3f}  {a} and {b}")
    table = []
    for limit in LIMITS:
        row = {"limit": limit, "same_person_accepted": round(float(np.mean(np.array(same) >= limit)), 4),
               "different_people_accepted": round(float(np.mean(scores >= limit)), 4)}
        table.append(row)
        print(f"  limit {limit:.2f}: accepts {row['same_person_accepted']:.1%} of same-person pairs, "
              f"{row['different_people_accepted']:.2%} of different-person pairs")

    OUT.write_text(json.dumps({
        "about": "Similarity of ArcFace embeddings (CSIM) for same-person and different-person pairs on the sample "
                 "clips and photographs. Same-person pairs are frames of one clip, so they share camera and light: "
                 "a picture from another day will score lower than these.",
        "people": len(people),
        "left_out_as_not_people": sorted(NOT_PEOPLE),
        "same_person": {"pairs": len(same), "lowest": round(min(same), 3), "p5": round(float(np.percentile(same, 5)), 3),
                        "median": round(float(np.median(same)), 3)},
        "different_people": {"pairs": int(len(scores)), "median": round(float(np.median(scores)), 3),
                             "p95": round(float(np.percentile(scores, 95)), 3), "p99": round(float(np.percentile(scores, 99)), 3),
                             "highest": round(float(scores.max()), 3),
                             "most_alike": [{"csim": round(s, 3), "a": a, "b": b} for s, a, b in different[:8]]},
        "by_limit": table,
    }, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
