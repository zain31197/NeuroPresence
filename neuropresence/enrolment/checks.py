"""Is this picture good enough to enrol?

The enrolled picture is what every output frame is made from, so its faults
show in all of them. Each check here measures one thing about a picture and
its tracked face, compares it with a limit, and says in plain words what is
wrong and what to change. The limits were set from measurements (see
scripts/study_enrolment.py and results/enrolment_study.json).

Two of the checks are built so that they do not judge a person's looks:
light is read from the brightest parts of the face and from blown-out pixels,
never from the average brightness, so skin tone does not decide it; and the
eyes count as closed only at a score no open eye reached in the sample faces,
because the score for an open eye differs from person to person.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..capture import TrackStatus

# ------------------------------------------------------------------ limits

MAX_YAW_DEG = 12.0
MAX_PITCH_DEG = 15.0
MAX_ROLL_DEG = 10.0

# The more pixels the picture has on the face, the sharper the output, over the
# whole range measured (110 to 460 pixels). On the generator's own output, and
# against the 260-pixel face it draws, a 150-pixel face in the picture gives
# 0.33 to 0.58 of the detail, a 200-pixel face 0.61 to 0.79, a 300-pixel face
# 1.10 to 1.21 and a 460-pixel face 1.7 (face_size_summary in the results file).
# So the stored picture is never shrunk to suit the generator, the floor is set
# where the loss is still moderate, and below the "good" size the person is told
# that coming closer helps, while there is room to.
MIN_FACE_HEIGHT_PX = 200  # within reach of a 640 x 480 camera: 42% of its frame height
GOOD_FACE_HEIGHT_PX = 300
ROOM_TO_COME_CLOSER = 0.50  # share of the frame height below which coming closer is suggested
MAX_FACE_HEIGHT_SHARE = 0.62  # of the frame height; closer than this the head is cropped

# Room around the tracked face, as a share of the face's own height or width.
# The tracked face ends at the upper forehead, so the hair needs the room above.
MIN_ROOM_ABOVE = 0.30
MIN_ROOM_BELOW = 0.12
MIN_ROOM_BESIDE = 0.25
MAX_OFF_CENTRE = 0.22  # face centre this far from the middle, as a share of the frame width

# Where the outline on the Enrolment screen asks for the face. A face that fills the
# outline passes the size and framing checks with room to spare on every side, and is
# as large as that allows, because a larger face gives a sharper output. The outline
# is worked out here, beside the limits, so the two cannot drift apart.
TARGET_FACE_HEIGHT_SHARE = 0.54  # of the frame height; the limit is MAX_FACE_HEIGHT_SHARE
TARGET_ROOM_ABOVE = 0.42  # of the face height; the limit is MIN_ROOM_ABOVE
TARGET_ROOM_BESIDE = 0.35  # of the face width, in a narrow frame; the limit is MIN_ROOM_BESIDE
FACE_WIDTH_OVER_HEIGHT = 0.85  # the tracked face of 41 front-facing sample faces: 0.75 to 0.97, median 0.85
LARGER_THAN_OUTLINE = 1.08  # a face this many times the outline's height is told to move back, not to shift

MIN_BRIGHT_LEVEL = 95.0  # the 95th percentile of face brightness, 0..255
MIN_LIGHT_RANGE = 25.0  # 95th minus 5th percentile: below this the face is flat and murky
MAX_BLOWN_OUT_SHARE = 0.06  # share of face pixels at or near pure white

MIN_SHARPNESS = 18.0  # variance of the Laplacian on the face, scaled to a standard size
# The output face keeps about 0.41 of the picture's sharpness by this measure. At
# this level the output face itself still passes MIN_SHARPNESS; below it, it would not.
GOOD_SHARPNESS = 45.0
SHARPNESS_SIZE = 256

MAX_LIP_GAP = 0.07  # gap between the lips over mouth width
MAX_EYE_CLOSED = 0.50  # MediaPipe eyeBlink score, either eye; open eyes scored up to 0.43
MAX_SMILE = 0.55  # MediaPipe mouthSmile score, either side

# Landmark numbers (MediaPipe FaceMesh).
UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291


@dataclass
class Check:
    key: str
    label: str
    passed: bool
    hint: str  # what is wrong and what to change; empty when the check passes
    value: float | None = None  # the measurement the verdict rests on
    tip: str = ""  # for a check that passes: what would make the result better still


# ----------------------------------------------------------- measurements


def sharpness(face_grey):
    """How much fine detail the face holds: variance of the Laplacian at a standard size."""
    if face_grey.size == 0:
        return 0.0
    scale = SHARPNESS_SIZE / face_grey.shape[0]
    resized = cv2.resize(face_grey, (max(1, int(round(face_grey.shape[1] * scale))), SHARPNESS_SIZE),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def face_region(image_bgr, landmarks):
    """Brightness (0..255) of the pixels inside the face outline, and the face as a grey crop."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros(grey.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    x, y, w, h = cv2.boundingRect(hull)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, grey.shape[1]), min(y + h, grey.shape[0])
    return grey[mask > 0], grey[y0:y1, x0:x1]


def target_face(frame_w, frame_h):
    """The face box the outline asks for in a frame of this size, as (x, y, w, h) in pixels.

    Its height is a set share of the frame's, unless the frame is so narrow (a phone held
    upright) that a face that tall would leave too little room beside it.
    """
    h = min(TARGET_FACE_HEIGHT_SHARE * frame_h, frame_w / (1 + 2 * TARGET_ROOM_BESIDE) / FACE_WIDTH_OVER_HEIGHT)
    w = h * FACE_WIDTH_OVER_HEIGHT
    return (frame_w - w) / 2, TARGET_ROOM_ABOVE * h, w, h


def outline_for(frame_w, frame_h):
    """The same box as shares of the frame's width and height, for the app to draw."""
    x, y, w, h = target_face(frame_w, frame_h)
    return {"x": round(x / frame_w, 4), "y": round(y / frame_h, 4), "w": round(w / frame_w, 4), "h": round(h / frame_h, 4)}


def placement(bbox, frame_w, frame_h):
    """How large the face is and where it sits in the frame: what the size and framing checks look at."""
    x, y, w, h = bbox
    return {
        "face_height_px": float(h),
        "face_height_share": h / frame_h,
        "face_over_outline": h / target_face(frame_w, frame_h)[3],
        "room_above": y / h,
        "room_below": (frame_h - (y + h)) / h,
        "room_left": x / w,
        "room_right": (frame_w - (x + w)) / w,
        "off_centre": ((x + w / 2) - frame_w / 2) / frame_w,
    }


def measure(image_bgr, track):
    """Everything the checks look at, as plain numbers. Needs a track with exactly one face."""
    frame_h, frame_w = image_bgr.shape[:2]
    points, scores = track.landmarks, track.blendshapes
    pixels, face_grey = face_region(image_bgr, points)
    dark, bright = (float(v) for v in np.percentile(pixels, [5, 95])) if pixels.size else (0.0, 0.0)
    mouth_width = float(np.linalg.norm(points[MOUTH_LEFT] - points[MOUTH_RIGHT]))
    yaw, pitch, roll = (float(angle) for angle in track.pose_deg)
    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        **placement(track.bbox, frame_w, frame_h),
        "bright_level": bright,
        "light_range": bright - dark,
        "blown_out_share": float((pixels >= 250).mean()) if pixels.size else 0.0,
        "sharpness": sharpness(face_grey),
        "lip_gap": float(np.linalg.norm(points[UPPER_LIP] - points[LOWER_LIP])) / max(mouth_width, 1e-6),
        "eye_closed": float(max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))),
        "smile": float(max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0))),
    }


# ----------------------------------------------------------------- checks


def _facing(m, upload):
    worst = max(abs(m["yaw_deg"]) / MAX_YAW_DEG, abs(m["pitch_deg"]) / MAX_PITCH_DEG, abs(m["roll_deg"]) / MAX_ROLL_DEG)
    hint = ""
    if abs(m["yaw_deg"]) > MAX_YAW_DEG:
        hint = "Choose a photo where you are facing the camera." if upload else "The head is turned to one side. Face the camera."
    elif abs(m["pitch_deg"]) > MAX_PITCH_DEG:  # a positive pitch is a lowered chin
        if upload:
            hint = "Choose a photo with your chin level, not tipped up or down."
        else:
            hint = "The chin is too low. Raise it a little." if m["pitch_deg"] > 0 else "The chin is too high. Lower it a little."
    elif abs(m["roll_deg"]) > MAX_ROLL_DEG:
        hint = "Choose a photo where your head is level, not tilted." if upload else "The head is tilted. Straighten it."
    return Check("facing", "Facing the camera", not hint, hint, round(worst, 2))


def _size(m, upload):
    hint = tip = ""
    if m["face_height_px"] < MIN_FACE_HEIGHT_PX:
        hint = ("The face is too small in the picture. Choose a photo where your face is larger, or crop in closer."
                if upload else "The face is too small in the picture. Move closer to the camera.")
    elif m["face_height_share"] > MAX_FACE_HEIGHT_SHARE:
        hint = ("The face fills too much of the picture. Choose a photo with a bit more room around your face."
                if upload else "The face fills too much of the picture. Move back from the camera.")
    elif m["face_height_px"] < GOOD_FACE_HEIGHT_PX and m["face_height_share"] < ROOM_TO_COME_CLOSER:
        tip = ("A larger face gives a sharper result. A closer, more tightly framed photo would help."
               if upload else "A larger face gives a sharper result. Move a little closer if you can.")
    return Check("size", "Face large enough", not hint, hint, round(m["face_height_px"]), tip)


def _framing(m, upload):
    hint = ""
    # A face larger than the outline has no room because it is too close; one that fits
    # the outline's size is only in the wrong place, and moving back would cost sharpness.
    too_close = m["face_over_outline"] > LARGER_THAN_OUTLINE
    if m["room_above"] < MIN_ROOM_ABOVE:
        if upload:
            hint = ("Choose a photo with a bit more room around your face." if too_close
                    else "Choose a photo with more room above your head.")
        else:
            hint = ("The top of the head is cut off. Move back a little." if too_close
                    else "The top of the head is cut off. Tilt the camera up a little, or sit lower.")
    elif m["room_below"] < MIN_ROOM_BELOW:
        if upload:
            hint = ("Choose a photo with a bit more room around your face." if too_close
                    else "Choose a photo with more room below your chin.")
        else:
            hint = ("The chin is at the bottom edge. Move back a little." if too_close
                    else "The chin is at the bottom edge. Tilt the camera down a little, or sit higher.")
    elif min(m["room_left"], m["room_right"]) < MIN_ROOM_BESIDE or abs(m["off_centre"]) > MAX_OFF_CENTRE:
        if upload:
            hint = "Choose a photo where your face is closer to the middle of the picture."
        else:
            # Directions are the person's own: a face at the left of the camera's picture
            # belongs to someone sitting too far to their right.
            side = "left" if m["off_centre"] < 0 else "right"
            hint = f"The face is too far to one side. Move to your {side}, toward the middle."
    room = min(m["room_above"], m["room_below"], m["room_left"], m["room_right"])
    return Check("framing", "Whole head in frame", not hint, hint, round(room, 2))


def _light(m, upload):
    hint = ""
    if m["blown_out_share"] > MAX_BLOWN_OUT_SHARE:
        hint = ("There is too much light on the face. Choose a photo with less light on it, or out of direct sun."
                if upload else "There is too much light on the face. Turn away from the lamp or window.")
    elif m["bright_level"] < MIN_BRIGHT_LEVEL or m["light_range"] < MIN_LIGHT_RANGE:
        hint = ("There is not enough light on the face. Choose a better-lit photo."
                if upload else "There is not enough light on the face. Face a lamp or a window.")
    return Check("light", "Enough light", not hint, hint, round(m["bright_level"], 1))


def _sharp(m, upload):
    hint = tip = ""
    if m["sharpness"] < MIN_SHARPNESS:
        hint = "The picture is blurred. Choose a sharper photo." if upload else "The picture is blurred. Hold still and check the camera's focus."
    elif m["sharpness"] < GOOD_SHARPNESS:
        tip = ("The picture is a little soft. A sharper, better-lit photo would help."
               if upload else "The picture is a little soft. More light and a clean lens make it sharper.")
    return Check("sharp", "Sharp", not hint, hint, round(m["sharpness"], 1), tip)


def _expression(m, upload):
    hint = ""
    if m["lip_gap"] > MAX_LIP_GAP:
        hint = "Choose a photo with your mouth closed." if upload else "The mouth is open. Close it."
    elif m["eye_closed"] > MAX_EYE_CLOSED:
        hint = "Choose a photo with your eyes open." if upload else "The eyes are closed. Open them."
    elif m["smile"] > MAX_SMILE:
        hint = ("Choose a photo with a relaxed face. A slight smile is fine, a broad one is not."
                if upload else "The face is not relaxed. A slight smile is fine, a broad one is not.")
    return Check("expression", "Relaxed face, eyes open", not hint, hint, round(m["lip_gap"], 3))


UNMEASURED = [("facing", "Facing the camera"), ("size", "Face large enough"), ("framing", "Whole head in frame"),
              ("light", "Enough light"), ("sharp", "Sharp"), ("expression", "Relaxed face, eyes open")]


def judge(measurements, origin="camera"):
    """Turn the measurements of one face (from measure) into the list of Checks.

    origin is "camera" (default, live capture: the hints are instructions you can
    act on right now) or "upload": a picture already taken cannot be told to move
    closer or tilt up, so the hints there ask for a different photo instead.
    """
    m, upload = measurements, origin == "upload"
    return [Check("face", "One face in view", True, ""),
            _facing(m, upload), _size(m, upload), _framing(m, upload), _light(m, upload),
            _sharp(m, upload), _expression(m, upload)]


def evaluate(image_bgr, track, origin="camera"):
    """Run every check on a picture and its tracked face. Returns the list of Checks.

    When no single face is found, the other checks cannot be measured and are
    reported as not passed, without a hint of their own.
    """
    upload = origin == "upload"
    if track.status is TrackStatus.NO_FACE:
        hint = "No face was found. Choose a different photo." if upload else "No face was found. Sit in front of the camera."
        face = Check("face", "One face in view", False, hint)
    elif track.status is TrackStatus.MULTIPLE_FACES:
        hint = ("More than one face was found. Choose a photo with only you in it." if upload
                else "More than one face was found. Only you should be in the picture.")
        face = Check("face", "One face in view", False, hint)
    else:
        return judge(measure(image_bgr, track), origin)
    return [face] + [Check(key, label, False, "") for key, label in UNMEASURED]


# ------------------------------------------------------- pose capture
#
# After the face is verified (front-on, step one), the person registers five more pictures of
# themselves: facing the camera again, turned to each side, and tilted up and down. Two things
# come of it: a richer face signature (an uploaded meeting picture, or a live face, is compared
# against whichever of these looks most like it, not only the step-one capture) weighted toward
# the pose people are actually in for most of a meeting, since facing the camera is registered
# again here on its own; and a measured range for how far this particular person's head actually
# turns, from the four turned/tilted ones, which replaces the fixed, one-person guess the
# reenactment engine otherwise clamps motion to (see reenactment/engine.py, calibrate_pose_range).
#
# The turned/tilted ones are not front-on, so the facing/size/framing/expression checks do not
# apply to them: only that one face is in view, it is turned the way asked, and the picture is
# lit and sharp enough to embed. "front" reuses the same facing limits as step one.

MIN_PROFILE_YAW_DEG = 35.0  # a side capture must turn at least this far for the angle to be worth measuring ...
MAX_PROFILE_YAW_DEG = 85.0  # ... and not so far the face is edge-on and barely visible
MIN_TILT_PITCH_DEG = 12.0  # an up/down capture must tilt at least this far ...
MAX_TILT_PITCH_DEG = 40.0  # ... and not so far the chin or forehead hides the face

# "front" first: it is the pose a person is actually in for most of a meeting, so it is the one
# most worth a second, fresh reference sample. The others calibrate the range of motion; this one
# does not (there is nothing to measure an extreme of), it only strengthens the identity signature.
POSE_KEYS = ("front", "left", "right", "up", "down")


def _front(m):
    """Facing the camera, the same limits as the step-one capture (see _facing)."""
    worst = max(abs(m["yaw_deg"]) / MAX_YAW_DEG, abs(m["pitch_deg"]) / MAX_PITCH_DEG, abs(m["roll_deg"]) / MAX_ROLL_DEG)
    hint = ""
    if abs(m["yaw_deg"]) > MAX_YAW_DEG:
        hint = "Turn to face the camera directly."
    elif abs(m["pitch_deg"]) > MAX_PITCH_DEG:
        hint = "Level your chin: not tipped up or down."
    elif abs(m["roll_deg"]) > MAX_ROLL_DEG:
        hint = "Straighten your head: it's tilted."
    return Check("pose", "Facing the camera", not hint, hint, round(worst, 2))


def _profile(m):
    """Turned to one side. Which side is not checked here: left and right use the same window,
    on whichever sign of yaw the camera measures; the caller tells them apart by the sign."""
    yaw = m["yaw_deg"]
    hint = ""
    if abs(yaw) < MIN_PROFILE_YAW_DEG:
        hint = "Turn your head further to the side."
    elif abs(yaw) > MAX_PROFILE_YAW_DEG:
        hint = "That's turned too far: ease back a little so the face stays in view."
    return Check("pose", "Turned to the side", not hint, hint, round(yaw, 1))


def _tilt_up(m):
    # A positive pitch is a lowered chin (see _facing); "up" is the other way.
    up = -m["pitch_deg"]
    hint = ""
    if up < MIN_TILT_PITCH_DEG:
        hint = "Tilt your head back a little further, looking up."
    elif up > MAX_TILT_PITCH_DEG:
        hint = "That's tilted too far back: ease down a little."
    return Check("pose", "Tilted up", not hint, hint, round(up, 1))


def _tilt_down(m):
    down = m["pitch_deg"]
    hint = ""
    if down < MIN_TILT_PITCH_DEG:
        hint = "Tilt your head down a little further, chin toward your chest."
    elif down > MAX_TILT_PITCH_DEG:
        hint = "That's tilted too far down: ease up a little."
    return Check("pose", "Tilted down", not hint, hint, round(down, 1))


_POSE_CHECK = {"front": _front, "left": _profile, "right": _profile, "up": _tilt_up, "down": _tilt_down}
POSE_LABEL = {"front": "Facing the camera", "left": "Turned to the side", "right": "Turned to the side",
              "up": "Tilted up", "down": "Tilted down"}


def judge_pose(measurements, pose):
    """The checks for one pose-capture frame: the angle itself, then light and sharp, reusing
    the camera wording (a pose capture is always live; there is no uploaded equivalent)."""
    m = measurements
    return [_POSE_CHECK[pose](m), _light(m, upload=False), _sharp(m, upload=False)]


def evaluate_pose(image_bgr, track, pose):
    """Run the pose-capture checks on a picture and its tracked face. Returns the list of Checks."""
    if track.status is TrackStatus.NO_FACE:
        face = Check("face", "One face in view", False, "No face was found. Sit in front of the camera.")
    elif track.status is TrackStatus.MULTIPLE_FACES:
        face = Check("face", "One face in view", False, "More than one face was found. Only you should be in the picture.")
    else:
        return [Check("face", "One face in view", True, "")] + judge_pose(measure(image_bgr, track), pose)
    return [face, Check("pose", POSE_LABEL[pose], False, ""), Check("light", "Enough light", False, ""),
            Check("sharp", "Sharp", False, "")]


def all_passed(checks):
    return all(check.passed for check in checks)


def first_hint(checks):
    """The one thing to fix first, or an empty string if everything passes."""
    return next((check.hint for check in checks if not check.passed and check.hint), "")


def first_tip(checks):
    """Once everything passes: the one change that would improve the picture most, or an empty string."""
    if not all_passed(checks):
        return ""
    return next((check.tip for check in checks if check.tip), "")


def as_dicts(checks):
    return [asdict(check) for check in checks]
