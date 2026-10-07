"""What the Enrolment screen explains: the limits the checks apply, and the measurements behind them."""

import json
from pathlib import Path

from ..capture.crop import MAX_PICTURE_DIM
from ..enrolment import checks
from ..identity import SAME_PERSON_CSIM

STUDY_FILE = "enrolment_study.json"


def limits():
    """The numbers the checks compare against, so the app can show them beside its readings."""
    return {
        "max_picture_px": MAX_PICTURE_DIM,
        "min_face_height_px": checks.MIN_FACE_HEIGHT_PX,
        "good_face_height_px": checks.GOOD_FACE_HEIGHT_PX,
        "max_face_height_share": checks.MAX_FACE_HEIGHT_SHARE,
        "max_yaw_deg": checks.MAX_YAW_DEG,
        "max_pitch_deg": checks.MAX_PITCH_DEG,
        "max_roll_deg": checks.MAX_ROLL_DEG,
        "min_sharpness": checks.MIN_SHARPNESS,
        "good_sharpness": checks.GOOD_SHARPNESS,
        "min_bright_level": checks.MIN_BRIGHT_LEVEL,
        "same_person_csim": SAME_PERSON_CSIM,
    }


def study(results_dir):
    """The enrolment study (scripts/study_enrolment.py) in the shape the app draws, or None if it has not been run."""
    try:
        data = json.loads((Path(results_dir) / STUDY_FILE).read_text())
        by_source = data["source_faults_summary"]
        good = by_source["neutral"]
        faults = [{"name": name, "clips": entry["clips"], "csim_change": entry["csim_to_real_change"],
                   "detail_kept": entry["detail_kept"]}
                  for name, entry in by_source.items() if name != "neutral"]
        pictures = [{"name": Path(name).stem,
                     "points": sorted(({"face_height_px": int(height), "detail": ratio} for height, ratio in sizes.items()),
                                      key=lambda point: point["face_height_px"])}
                    for name, sizes in data["face_size_summary"].items()]
        return {
            "clips": good["clips"],
            "good_picture": {"csim": good["csim_to_real"], "detail_kept": good["detail_kept"]},
            "faults": sorted(faults, key=lambda fault: fault["csim_change"]),  # the costliest first
            "face_size": {"reference_px": data["face_size_reference_px"], "pictures": pictures},
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None  # not run yet, or written by an older version of the script


def enrolment_guide(results_dir):
    return {"limits": limits(), "study": study(results_dir)}
