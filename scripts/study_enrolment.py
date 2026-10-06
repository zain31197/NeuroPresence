"""Does the enrolled picture's quality change the output? A measurement, not an opinion.

Part A, source faults. For several sample clips, the same clip is reenacted
from different source frames of that clip: a neutral, front-facing one, and
ones with an open mouth, closed eyes, a turned head or a broad smile, plus
blurred, darkened and low-resolution copies of the neutral one. Because the
source comes from the clip, the real frame is the right answer for every
output frame, so each output is compared with the real frame of the same
instant: how alike the two faces are (CSIM), and how far apart their
expression, pose and pixels are.

Part B, face size. The generator makes a 512-pixel picture of the face region.
A sharp, high-resolution picture is enrolled at several sizes and reproduced
without any movement, and the detail in the face is measured three ways: how
sharp the output face is when shown at one standard size (what a viewer gets),
how much of the source's detail that is, and how much is kept at the picture's
own resolution (whether the face is as sharp as the frame around it). This
finds how small the face can be before the output goes soft, and how large
before the face is softer than its surroundings.

Usage:
    python scripts/study_enrolment.py              # both parts, about 15 minutes
    python scripts/study_enrolment.py --only b     # one part; the other is kept from the saved file
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuropresence.capture import FaceTracker
from neuropresence.capture.crop import square_face_crop
from neuropresence.enrolment import checks
from neuropresence.identity import IdentityScorer
from neuropresence.identity.align import align_face, five_points
from neuropresence.reenactment import ReenactmentEngine, SourceError
from neuropresence.reenactment.engine import LIVEPORTRAIT_DIR, OUTPUT_SIZE

SAMPLES = LIVEPORTRAIT_DIR / "assets" / "examples"
CLIPS = ["d0", "d3", "d6", "d9", "d10", "d18"]
FRAME_STEP, FRAMES_PER_CLIP = 2, 90
SIZE_PICTURES = ["source/s7.jpg", "source/s1.jpg", "driving/d13.mp4"]
FACE_HEIGHTS = [460, 380, 300, 260, 200, 150, 110]
DRAWN_FACE_HEIGHT = 260  # how tall the generator draws the face in its 512-pixel picture
OUT = ROOT / "results" / "enrolment_study.json"


def read_frames(clip):
    capture = cv2.VideoCapture(str(SAMPLES / "driving" / f"{clip}.mp4"))
    frames, index = [], 0
    while len(frames) < FRAMES_PER_CLIP:
        ok, frame = capture.read()
        if not ok:
            break
        if index % FRAME_STEP == 0:
            frames.append(frame)
        index += 1
    capture.release()
    return frames


def describe(image, track, scorer):
    """What is compared between a real frame and an output frame."""
    return {
        "embedding": scorer.embed(image, track.landmarks),
        "expression": np.array([score for name, score in sorted(track.blendshapes.items()) if name != "_neutral"]),
        "pose": np.array(track.pose_deg),
        "face": align_face(image, five_points(track.landmarks)).astype(np.float32),
        "sharpness": checks.sharpness(checks.face_region(image, track.landmarks)[1]),
    }


def source_variants(frames, tracks):
    """The source frames to compare for one clip: {name: (image, what was changed)}."""
    measured = {i: checks.measure(frames[i], t) for i, t in enumerate(tracks) if t.ok}

    def unrest(m):  # how far a frame is from a neutral, front-facing face
        return (m["lip_gap"] / checks.MAX_LIP_GAP + m["eye_closed"] / checks.MAX_EYE_CLOSED + m["smile"] / checks.MAX_SMILE
                + abs(m["yaw_deg"]) / checks.MAX_YAW_DEG + abs(m["pitch_deg"]) / checks.MAX_PITCH_DEG)

    neutral = min(measured, key=lambda i: unrest(measured[i]))
    base = frames[neutral]
    variants = {"neutral": (base, measured[neutral])}

    def worst(key, limit, others):
        """The frame that breaks one limit the most while staying within the others."""
        ok = [i for i in measured if all(abs(measured[i][k]) <= v for k, v in others.items())]
        if not ok:
            return
        i = max(ok, key=lambda j: abs(measured[j][key]))
        if abs(measured[i][key]) > limit:
            return i

    posed = {"yaw_deg": checks.MAX_YAW_DEG, "pitch_deg": checks.MAX_PITCH_DEG}
    faults = {
        "mouth open": worst("lip_gap", checks.MAX_LIP_GAP, posed),
        "eyes closed": worst("eye_closed", checks.MAX_EYE_CLOSED, posed | {"lip_gap": checks.MAX_LIP_GAP}),
        "broad smile": worst("smile", checks.MAX_SMILE, posed | {"lip_gap": checks.MAX_LIP_GAP}),
        "head turned": worst("yaw_deg", checks.MAX_YAW_DEG, {"lip_gap": checks.MAX_LIP_GAP * 2}),
    }
    for name, index in faults.items():
        if index is not None:
            variants[name] = (frames[index], measured[index])
    small = cv2.resize(base, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    variants["blurred"] = (cv2.GaussianBlur(base, (0, 0), 3.0), None)
    variants["dark"] = ((base.astype(np.float32) * 0.4).astype(np.uint8), None)
    variants["low resolution"] = (cv2.resize(small, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR), None)
    return variants


def reenact(engine, tracker, source):
    """Enrol a source the way the app does: the picture, and its own face as the neutral pose."""
    track = tracker.process(source)
    if not track.ok:
        raise SourceError("no single face found in the source")
    engine.set_source(source)
    engine.set_reference(square_face_crop(source, track.bbox))
    return track


def part_a(engine, tracker, scorer):
    rows = []
    for clip in CLIPS:
        frames = read_frames(clip)
        tracks = [tracker.process(frame) for frame in frames]
        real = [describe(f, t, scorer) if t.ok else None for f, t in zip(frames, tracks)]
        for name, (source, measured) in source_variants(frames, tracks).items():
            try:
                source_track = reenact(engine, tracker, source)
            except SourceError as err:
                rows.append({"clip": clip, "source": name, "error": str(err)})
                continue
            csim, expression, pose, pixels, detail, missing = [], [], [], [], [], 0
            for frame, track, truth in zip(frames, tracks, real):
                if truth is None:
                    continue
                output = engine.drive(square_face_crop(frame, track.bbox))
                out_track = tracker.process(output)
                if not out_track.ok:
                    missing += 1
                    continue
                got = describe(output, out_track, scorer)
                csim.append(float(got["embedding"] @ truth["embedding"]))
                expression.append(float(np.abs(got["expression"] - truth["expression"]).mean()))
                pose.append(float(np.abs(got["pose"] - truth["pose"]).mean()))
                pixels.append(float(np.abs(got["face"] - truth["face"]).mean()))
                detail.append(got["sharpness"] / max(truth["sharpness"], 1e-6))
            passed = checks.all_passed(checks.evaluate(source, source_track))
            row = {
                "clip": clip, "source": name, "frames": len(csim), "frames_without_a_face": missing,
                "source_passes_every_check": passed,
                "csim_to_real": round(float(np.mean(csim)), 3),
                "expression_error": round(float(np.mean(expression)), 4),
                "pose_error_deg": round(float(np.mean(pose)), 2),
                "face_pixel_error": round(float(np.mean(pixels)), 2),
                "detail_kept": round(float(np.median(detail)), 3),
            }
            if measured:
                row["source_measurements"] = {k: round(v, 3) for k, v in measured.items()
                                              if k in ("lip_gap", "eye_closed", "smile", "yaw_deg", "pitch_deg")}
            rows.append(row)
            print(f"  {clip:4s} {name:15s} CSIM {row['csim_to_real']:.3f}  expression {row['expression_error']:.4f}  "
                  f"pose {row['pose_error_deg']:.2f}  pixels {row['face_pixel_error']:.2f}  detail {row['detail_kept']:.2f}", flush=True)
    return rows


def summarise_a(rows):
    """Average each kind of source over the clips, and its change from the neutral source of the same clip."""
    neutral = {r["clip"]: r for r in rows if r["source"] == "neutral" and "error" not in r}
    summary = {}
    for name in dict.fromkeys(r["source"] for r in rows):
        mine = [r for r in rows if r["source"] == name and "error" not in r and r["clip"] in neutral]
        if not mine:
            continue
        entry = {"clips": len(mine)}
        for key in ("csim_to_real", "expression_error", "pose_error_deg", "face_pixel_error", "detail_kept"):
            entry[key] = round(float(np.mean([r[key] for r in mine])), 4)
            entry[f"{key}_change"] = round(float(np.mean([r[key] - neutral[r["clip"]][key] for r in mine])), 4)
        summary[name] = entry
    return summary


def load_picture(name):
    path = SAMPLES / name
    if path.suffix == ".mp4":
        capture = cv2.VideoCapture(str(path))
        ok, image = capture.read()
        capture.release()
        return image if ok else None
    return cv2.imread(str(path))


def detail_at_own_size(image, landmarks):
    """Fine detail inside the face at the picture's own resolution: variance of the Laplacian, not resized."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(landmarks.astype(np.int32)), 255)
    return float(cv2.Laplacian(grey, cv2.CV_64F)[mask > 0].var())


def part_b(engine, tracker):
    rows = []
    for name in SIZE_PICTURES:
        original = load_picture(name)
        track = tracker.process(original)
        if original is None or not track.ok:
            continue
        for target in FACE_HEIGHTS:
            scale = target / track.bbox[3]
            if scale > 1.0:
                continue  # only ever shrink: enlarging adds no real detail
            picture = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            try:
                source_track = reenact(engine, tracker, picture)
            except SourceError:
                continue
            output = engine.drive(square_face_crop(picture, source_track.bbox))  # the neutral pose itself: no movement
            source = engine.source_frame
            placed = tracker.process(source)
            drawn = tracker.process(engine.source_crop)
            if not placed.ok or not drawn.ok:
                continue
            before = checks.sharpness(checks.face_region(source, placed.landmarks)[1])
            after = checks.sharpness(checks.face_region(output, placed.landmarks)[1])
            own = detail_at_own_size(output, placed.landmarks) / max(detail_at_own_size(source, placed.landmarks), 1e-6)
            # The generator's own picture, before it is resized into the frame: nothing but the
            # source picture differs between sizes here, so this is the effect of face size alone.
            generated = detail_at_own_size(engine.last_crop, drawn.landmarks)
            # How much the generator's 512-pixel picture is stretched to fill its place in the frame.
            stretch = float(np.linalg.norm(engine._paste["to_roi"][:, 0]))
            row = {"picture": name, "face_height_px": int(placed.bbox[3]), "stretch": round(stretch, 2),
                   "source_sharpness": round(before, 1), "output_sharpness": round(after, 1),
                   "generator_output_detail": round(generated, 1),
                   "detail_kept": round(after / max(before, 1e-6), 3), "detail_kept_at_own_size": round(own, 3)}
            rows.append(row)
            print(f"  {name:16s} face {row['face_height_px']:4d} px  stretch {row['stretch']:.2f}  "
                  f"sharpness {row['source_sharpness']:6.1f} -> {row['output_sharpness']:5.1f}  "
                  f"generator output {row['generator_output_detail']:6.1f}  "
                  f"kept {row['detail_kept']:.2f}  kept at own size {row['detail_kept_at_own_size']:.2f}", flush=True)
    return rows


def summarise_b(rows):
    """Detail in the generator's output at each face size, as a multiple of what the same picture gives at 260 pixels."""
    summary = {}
    for name in dict.fromkeys(r["picture"] for r in rows):
        mine = [r for r in rows if r["picture"] == name]
        base = min(mine, key=lambda r: abs(r["face_height_px"] - DRAWN_FACE_HEIGHT))
        if abs(base["face_height_px"] - DRAWN_FACE_HEIGHT) > 15:
            continue  # this picture was never tried near that size
        summary[name] = {str(r["face_height_px"]): round(r["generator_output_detail"] / base["generator_output_detail"], 2)
                         for r in mine}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["a", "b"], help="run one part and keep the other from the saved file")
    args = parser.parse_args()
    saved = json.loads(OUT.read_text()) if args.only and OUT.exists() else {}

    engine = ReenactmentEngine()
    with FaceTracker(video=False) as tracker:
        scorer = IdentityScorer(tracker=tracker)
        rows_a, rows_b = saved.get("source_faults", []), saved.get("face_size", [])
        if args.only != "b":
            print("Part A: faults in the source picture")
            rows_a = part_a(engine, tracker, scorer)
        if args.only != "a":
            print("Part B: size of the face in the picture")
            rows_b = part_b(engine, tracker)
    summary = summarise_a(rows_a)
    sizes = summarise_b(rows_b)
    print("\nDetail in the generator's output by face height, against the same picture at 260 pixels")
    for name, by_height in sizes.items():
        print(f"  {name:16s} " + "  ".join(f"{height} px: {ratio:.2f}" for height, ratio in by_height.items()))
    print(f"\n{'source':15s} {'clips':>5s} {'CSIM':>7s} {'change':>8s} {'expr':>8s} {'change':>8s} {'pixels':>7s} {'change':>7s} {'detail':>7s}")
    for name, s in summary.items():
        print(f"{name:15s} {s['clips']:5d} {s['csim_to_real']:7.3f} {s['csim_to_real_change']:+8.3f} "
              f"{s['expression_error']:8.4f} {s['expression_error_change']:+8.4f} "
              f"{s['face_pixel_error']:7.2f} {s['face_pixel_error_change']:+7.2f} {s['detail_kept']:7.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "about": "Effect of the enrolled picture on output quality. Part A compares each output frame with the real "
                 "frame of the same instant in self-reenactment; changes are against the neutral source of the same "
                 "clip. Part B reproduces a still picture at several face sizes. Sharpness is the variance of the "
                 "Laplacian on the face brought to a standard height of 256 pixels, the measure the Sharp check uses; "
                 "detail_kept is output over source by that measure, and detail_kept_at_own_size is the same ratio "
                 "at the picture's own resolution, where the frame around the face keeps all of its detail. "
                 "generator_output_detail is the variance of the Laplacian inside the face of the generator's own "
                 "512-pixel picture, before it is resized into the frame; face_size_summary gives it as a multiple "
                 "of what the same picture yields with a 260-pixel face, the size the generator draws.",
        "generator_output_px": OUTPUT_SIZE,
        "source_faults_summary": summary,
        "face_size_reference_px": DRAWN_FACE_HEIGHT,
        "face_size_summary": sizes,
        "source_faults": rows_a,
        "face_size": rows_b,
    }, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
