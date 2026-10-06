"""Check our face alignment against InsightFace's standard one.

CSIM is normally computed by aligning the face with the five keypoints of
InsightFace's own detector (SCRFD). This project aligns with five MediaPipe
landmarks instead, so the identity module needs no second detector. This
script measures how much that choice changes the similarity scores, using
LivePortrait's sample images and frames of its sample videos.

Usage:
    python scripts/validate_alignment.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker
from neuropresence.identity import ArcFaceEmbedder
from neuropresence.identity.align import align_face, five_points
from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR

sys.path.insert(0, str(LIVEPORTRAIT_DIR))
from src.utils.face_analysis_diy import FaceAnalysisDIY  # LivePortrait's copy of InsightFace

SAMPLES = LIVEPORTRAIT_DIR / "assets" / "examples"
VIDEOS = ["d0", "d3", "d6", "d9", "d10", "d18"]
SAME_FACE_AGREEMENT = 0.5  # below this the two detectors picked different faces in the image


def sample_images():
    images = {p.name: cv2.imread(str(p)) for p in sorted((SAMPLES / "source").glob("*.jpg"))}
    for name in VIDEOS:
        capture = cv2.VideoCapture(str(SAMPLES / "driving" / f"{name}.mp4"))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in (0, count // 2, count - 5):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                images[f"{name}_frame{index}"] = frame
        capture.release()
    return images


def main():
    detector = FaceAnalysisDIY(name="buffalo_l", root=str(LIVEPORTRAIT_DIR / "pretrained_weights" / "insightface"),
                               providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=(512, 512), det_thresh=0.1)
    embedder = ArcFaceEmbedder()

    standard, ours, offsets = {}, {}, []
    with FaceTracker(video=False) as tracker:
        for name, image in sample_images().items():
            faces = detector.get(image, flag_do_landmark_2d_106=False)
            track = tracker.process(image)
            if not faces or not track.ok:
                continue  # drawings, animals and group photos that one of the detectors rejects
            keypoints, points = faces[0].kps.astype(np.float64), five_points(track.landmarks)
            a, b = embedder.embed(align_face(image, keypoints)), embedder.embed(align_face(image, points))
            if float(a @ b) < SAME_FACE_AGREEMENT:
                continue
            standard[name], ours[name] = a, b
            eye_distance = np.linalg.norm(keypoints[0] - keypoints[1])
            offsets.append(np.linalg.norm(points - keypoints, axis=1) / eye_distance)

    names = list(standard)
    upper = np.triu_indices(len(names), 1)
    sim_standard = np.array([[standard[a] @ standard[b] for b in names] for a in names])[upper]
    sim_ours = np.array([[ours[a] @ ours[b] for b in names] for a in names])[upper]
    difference = sim_ours - sim_standard
    report = {
        "images_compared": len(names),
        "image_pairs_compared": int(difference.size),
        "point_offset_fraction_of_eye_distance_median": round(float(np.median(offsets)), 3),
        "same_image_similarity_between_alignments_median": round(
            float(np.median([standard[n] @ ours[n] for n in names])), 3),
        "csim_difference_mean": round(float(difference.mean()), 4),
        "csim_difference_mean_abs": round(float(np.abs(difference).mean()), 4),
        "csim_difference_max_abs": round(float(np.abs(difference).max()), 3),
        "csim_correlation": round(float(np.corrcoef(sim_ours, sim_standard)[0, 1]), 4),
    }
    print(json.dumps(report, indent=2))
    out = ROOT / "results" / "alignment_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
