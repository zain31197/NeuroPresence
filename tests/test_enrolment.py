"""The enrolment checks and the enrolment store."""

import cv2
import numpy as np
import pytest

from neuropresence.capture import TrackResult, TrackStatus
from neuropresence.enrolment import (EnrolmentStore, all_passed, evaluate, first_hint, first_tip, make_candidate,
                                     make_pose_candidate, prepare)
from neuropresence.enrolment import checks as c

# ------------------------------------------------------------ a good face

# What measure() returns for a well-taken picture. Each test changes one thing.
GOOD = {
    "yaw_deg": 2.0, "pitch_deg": -3.0, "roll_deg": 1.0,
    "face_height_px": 320.0, "face_height_share": 0.44, "face_over_outline": 0.82,
    "room_above": 0.6, "room_below": 0.8, "room_left": 1.5, "room_right": 1.5, "off_centre": 0.0,
    "bright_level": 170.0, "light_range": 90.0, "blown_out_share": 0.0,
    "sharpness": 80.0, "lip_gap": 0.01, "eye_closed": 0.1, "smile": 0.1,
}


def verdicts(**changes):
    return {check.key: check for check in c.judge({**GOOD, **changes})}


def test_a_good_picture_passes_every_check():
    checks = c.judge(GOOD)
    assert [check.key for check in checks] == ["face", "facing", "size", "framing", "light", "sharp", "expression"]
    assert all_passed(checks)
    assert first_hint(checks) == "" and first_tip(checks) == ""
    assert all(check.hint == "" and check.tip == "" for check in checks)


@pytest.mark.parametrize("changes, key, hint", [
    ({"yaw_deg": 20.0}, "facing", "The head is turned to one side. Face the camera."),
    ({"pitch_deg": 22.0}, "facing", "The chin is too low. Raise it a little."),
    ({"pitch_deg": -22.0}, "facing", "The chin is too high. Lower it a little."),
    ({"roll_deg": -14.0}, "facing", "The head is tilted. Straighten it."),
    ({"face_height_px": 120.0}, "size", "The face is too small in the picture. Move closer to the camera."),
    ({"face_height_share": 0.7}, "size", "The face fills too much of the picture. Move back from the camera."),
    ({"room_above": 0.1}, "framing", "The top of the head is cut off. Tilt the camera up a little, or sit lower."),
    ({"room_above": 0.1, "face_over_outline": 1.12}, "framing", "The top of the head is cut off. Move back a little."),
    ({"room_below": 0.05}, "framing", "The chin is at the bottom edge. Tilt the camera down a little, or sit higher."),
    ({"room_below": 0.05, "face_over_outline": 1.12}, "framing", "The chin is at the bottom edge. Move back a little."),
    ({"off_centre": -0.3}, "framing", "The face is too far to one side. Move to your left, toward the middle."),
    ({"room_right": 0.1, "off_centre": 0.2}, "framing", "The face is too far to one side. Move to your right, toward the middle."),
    ({"bright_level": 60.0}, "light", "There is not enough light on the face. Face a lamp or a window."),
    ({"light_range": 10.0}, "light", "There is not enough light on the face. Face a lamp or a window."),
    ({"blown_out_share": 0.3}, "light", "There is too much light on the face. Turn away from the lamp or window."),
    ({"sharpness": 6.0}, "sharp", "The picture is blurred. Hold still and check the camera's focus."),
    ({"lip_gap": 0.2}, "expression", "The mouth is open. Close it."),
    ({"eye_closed": 0.7}, "expression", "The eyes are closed. Open them."),
    ({"smile": 0.8}, "expression", "The face is not relaxed. A slight smile is fine, a broad one is not."),
])
def test_each_fault_fails_its_own_check_and_says_what_to_do(changes, key, hint):
    found = verdicts(**changes)
    assert not found[key].passed
    assert found[key].hint == hint
    assert all(check.passed for other, check in found.items() if other != key)  # and no other check


@pytest.mark.parametrize("changes, key, camera_hint, upload_hint", [
    ({"face_height_px": 120.0}, "size",
     "The face is too small in the picture. Move closer to the camera.",
     "The face is too small in the picture. Choose a photo where your face is larger, or crop in closer."),
    ({"yaw_deg": 20.0}, "facing",
     "The head is turned to one side. Face the camera.",
     "Choose a photo where you are facing the camera."),
    ({"bright_level": 60.0}, "light",
     "There is not enough light on the face. Face a lamp or a window.",
     "There is not enough light on the face. Choose a better-lit photo."),
    ({"sharpness": 6.0}, "sharp",
     "The picture is blurred. Hold still and check the camera's focus.",
     "The picture is blurred. Choose a sharper photo."),
    ({"lip_gap": 0.2}, "expression",
     "The mouth is open. Close it.",
     "Choose a photo with your mouth closed."),
])
def test_an_uploaded_picture_is_told_to_choose_another_photo_not_to_move(changes, key, camera_hint, upload_hint):
    # A picture already taken cannot be acted on live: an upload gets its own wording,
    # a live camera frame keeps the instruction it can actually be followed by.
    m = {**GOOD, **changes}
    camera_found = {check.key: check for check in c.judge(m)}  # origin defaults to "camera"
    upload_found = {check.key: check for check in c.judge(m, origin="upload")}
    assert camera_found[key].hint == camera_hint
    assert upload_found[key].hint == upload_hint


def test_limits_are_inclusive():
    at_the_limit = verdicts(yaw_deg=c.MAX_YAW_DEG, face_height_px=c.MIN_FACE_HEIGHT_PX, room_above=c.MIN_ROOM_ABOVE,
                            bright_level=c.MIN_BRIGHT_LEVEL, sharpness=c.MIN_SHARPNESS, lip_gap=c.MAX_LIP_GAP,
                            eye_closed=c.MAX_EYE_CLOSED)
    assert all(check.passed for check in at_the_limit.values())


def test_a_face_below_the_measured_floor_is_too_small():
    # Below about 200 pixels the output loses more than a third of the detail it has at 260 (the enrolment study).
    assert c.MIN_FACE_HEIGHT_PX == 200
    assert not verdicts(face_height_px=199.0)["size"].passed
    assert verdicts(face_height_px=200.0)["size"].passed


@pytest.mark.parametrize("changes, key, tip", [
    ({"face_height_px": 240.0, "face_height_share": 0.33}, "size",
     "A larger face gives a sharper result. Move a little closer if you can."),
    ({"sharpness": 30.0}, "sharp", "The picture is a little soft. More light and a clean lens make it sharper."),
])
def test_a_picture_that_passes_can_still_be_told_how_to_be_better(changes, key, tip):
    checks = c.judge({**GOOD, **changes})
    found = {check.key: check for check in checks}
    assert all_passed(checks) and first_hint(checks) == ""  # advice never stops a picture from being taken
    assert found[key].tip == tip and first_tip(checks) == tip
    assert all(check.tip == "" for other, check in found.items() if other != key)


def test_coming_closer_is_not_suggested_when_there_is_no_room_to():
    # 240 pixels is all a small camera frame allows here: any closer and the head would be cropped.
    assert verdicts(face_height_px=240.0, face_height_share=0.55)["size"].tip == ""
    assert verdicts(face_height_px=c.GOOD_FACE_HEIGHT_PX, face_height_share=0.42)["size"].tip == ""


def test_advice_waits_until_every_check_passes():
    checks = c.judge({**GOOD, "sharpness": 30.0, "yaw_deg": 20.0})
    assert first_hint(checks) == "The head is turned to one side. Face the camera."
    assert first_tip(checks) == ""  # one thing at a time: the fault first
    both = c.judge({**GOOD, "face_height_px": 240.0, "face_height_share": 0.33, "sharpness": 30.0})
    assert first_tip(both) == "A larger face gives a sharper result. Move a little closer if you can."


def test_eyes_that_score_as_open_for_some_people_are_not_called_closed():
    # The highest score an open eye reached in the sample faces was 0.43.
    assert verdicts(eye_closed=0.43)["expression"].passed


# ------------------------------------------------------------- the outline

# Camera pictures of several shapes: wide, the older 4:3, square, a phone held upright, and full HD.
FRAMES = [(1280, 720), (640, 480), (512, 512), (720, 1280), (1920, 1080)]


def placed(box, frame_w, frame_h):
    """The verdicts for an otherwise good face whose tracked box is here in a frame of this size."""
    return {check.key: check for check in c.judge({**GOOD, **c.placement(box, frame_w, frame_h)})}


@pytest.mark.parametrize("frame_w, frame_h", FRAMES)
def test_a_face_that_fills_the_outline_passes_with_room_to_spare(frame_w, frame_h):
    x, y, w, h = c.target_face(frame_w, frame_h)
    assert all(check.passed for check in placed((x, y, w, h), frame_w, frame_h).values())
    for scale in (0.9, 1.1):  # a tenth smaller or larger than the outline
        resized = (x - w * (scale - 1) / 2, y - h * (scale - 1) / 2, w * scale, h * scale)
        assert all(check.passed for check in placed(resized, frame_w, frame_h).values()), scale
    for dx, dy in ((-0.04, 0), (0.04, 0), (0, -0.04), (0, 0.04)):  # or a little off in any direction
        moved = (x + dx * frame_w, y + dy * frame_h, w, h)
        assert all(check.passed for check in placed(moved, frame_w, frame_h).values()), (dx, dy)


def test_the_outline_asks_for_a_large_face_because_that_is_sharper():
    x, y, w, h = c.target_face(1280, 720)
    assert h / 720 == pytest.approx(c.TARGET_FACE_HEIGHT_SHARE) and h >= c.GOOD_FACE_HEIGHT_PX
    assert w / h == pytest.approx(c.FACE_WIDTH_OVER_HEIGHT)
    assert x + w / 2 == pytest.approx(640)  # in the middle
    assert c.placement((x, y, w, h), 1280, 720)["face_over_outline"] == pytest.approx(1.0)
    # Whoever fills the outline is not then told to come closer, in a small camera picture either.
    assert placed((x, y, w, h), 1280, 720)["size"].tip == ""
    assert placed(c.target_face(640, 480), 640, 480)["size"].tip == ""
    assert c.target_face(640, 480)[3] > c.MIN_FACE_HEIGHT_PX * 1.2


def test_in_an_upright_picture_the_outline_is_limited_by_the_width():
    x, y, w, h = c.target_face(720, 1280)
    assert h / 1280 < c.TARGET_FACE_HEIGHT_SHARE  # a face that tall would touch the sides
    assert x / w == pytest.approx(c.TARGET_ROOM_BESIDE)


def test_the_outline_is_given_to_the_app_as_shares_of_the_frame():
    box = c.outline_for(1280, 720)
    assert box["h"] == pytest.approx(0.54) and box["y"] == pytest.approx(0.42 * 0.54, abs=1e-3)
    assert box["w"] == pytest.approx(0.54 * 720 * 0.85 / 1280, abs=1e-3)
    assert box["x"] + box["w"] / 2 == pytest.approx(0.5, abs=1e-3)
    assert c.outline_for(2560, 1440) == box  # only the shape of the picture matters, not its size


def test_a_well_sized_face_too_high_in_the_picture_is_told_to_shift_not_to_move_back():
    # As first seen on a real camera (7 October 2026): a face of a good size, 396 px of 720,
    # with the hair at the top edge. Moving back would have cost sharpness for nothing.
    found = placed((472, 93, 336, 396), 1280, 720)
    assert found["size"].passed and not found["framing"].passed
    assert found["framing"].hint == "The top of the head is cut off. Tilt the camera up a little, or sit lower."
    # A face clearly larger than the outline has no room because it is too close.
    closer = placed((440, 60, 400, 440), 1280, 720)
    assert closer["size"].passed and closer["framing"].hint == "The top of the head is cut off. Move back a little."


# ------------------------------------------------------- measured pictures

FACE_BOX = (220, 75, 200, 215)  # x, y, w, h in a 640 x 360 picture


def textured(shape=(360, 640, 3), seed=0):
    """A stand-in for a well-lit face: broad light and shade, with fine detail on top."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    shade = 45 * np.sin(xs / 10.0) * np.cos(ys / 12.0)
    detail = np.random.default_rng(seed).integers(-20, 21, shape[:2])
    grey = np.clip(128 + shade + detail, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def tracked(yaw=0.0, pitch=0.0, lip_gap_px=2.0, blink=0.1, box=FACE_BOX):
    """A tracker result for one face filling the box, mouth nearly closed."""
    x, y, w, h = box
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(x, x + w, 478), rng.uniform(y, y + h, 478)]).astype(np.float32)
    points[:4] = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]  # the outline spans the whole box
    middle = x + w / 2
    points[c.UPPER_LIP], points[c.LOWER_LIP] = (middle, y + 120), (middle, y + 120 + lip_gap_px)
    points[c.MOUTH_LEFT], points[c.MOUTH_RIGHT] = (middle - 30, y + 121), (middle + 30, y + 121)
    return TrackResult(TrackStatus.OK, 1.0, landmarks=points, bbox=box, pose_deg=(yaw, pitch, 0.0),
                       blendshapes={"eyeBlinkLeft": blink, "eyeBlinkRight": blink, "mouthSmileLeft": 0.1,
                                    "mouthSmileRight": 0.1, "jawOpen": 0.05})


def test_a_bright_detailed_face_passes_and_reports_its_measurements():
    picture = textured()
    measured = c.measure(picture, tracked())
    assert measured["face_height_px"] == 215
    assert measured["room_above"] == pytest.approx(75 / 215)
    assert measured["off_centre"] == pytest.approx(0.0)
    assert measured["bright_level"] > 150 and measured["blown_out_share"] == 0.0
    assert measured["sharpness"] > 10 * c.MIN_SHARPNESS
    assert measured["lip_gap"] == pytest.approx(2 / 60, abs=1e-3)
    assert all_passed(evaluate(picture, tracked()))


def test_light_is_measured_on_the_face_not_the_whole_picture():
    picture = textured()
    x, y, w, h = FACE_BOX
    picture[y:y + h, x:x + w] //= 5  # only the face is dark
    found = {check.key: check for check in evaluate(picture, tracked())}
    assert not found["light"].passed
    bright_face = np.zeros((360, 640, 3), dtype=np.uint8)  # dark room, face lit
    bright_face[y:y + h, x:x + w] = textured()[y:y + h, x:x + w]
    assert {check.key: check for check in evaluate(bright_face, tracked())}["light"].passed


def test_a_dark_skinned_face_in_good_light_passes_the_light_check():
    """Low average brightness with bright highlights is good light on darker skin, not a dark picture."""
    rng = np.random.default_rng(3)
    picture = rng.integers(25, 75, (360, 640, 3), dtype=np.uint8)  # mostly dark tones
    highlights = rng.random((360, 640)) < 0.12  # with highlights on brow, nose and cheeks
    picture[highlights] = rng.integers(110, 170, (int(highlights.sum()), 3), dtype=np.uint8)
    measured = c.measure(picture, tracked())
    assert cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY).mean() < 70  # darker on average than the old-style limit
    assert {check.key: check for check in evaluate(picture, tracked())}["light"].passed, measured


def test_blown_out_and_blurred_pictures_fail():
    found = {check.key: check for check in evaluate(np.full((360, 640, 3), 255, np.uint8), tracked())}
    assert found["light"].hint.startswith("There is too much light")
    blurred = cv2.GaussianBlur(textured(), (0, 0), 4.0)
    found = {check.key: check for check in evaluate(blurred, tracked())}
    assert not found["sharp"].passed and found["light"].passed


def test_sharpness_does_not_depend_on_the_size_of_the_picture():
    face = textured(shape=(400, 300, 3))[:, :, 0]
    blurred = cv2.GaussianBlur(face, (0, 0), 6.0)
    assert c.sharpness(face) > 20 * c.sharpness(blurred)
    twice = cv2.resize(blurred, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    assert c.sharpness(twice) == pytest.approx(c.sharpness(blurred), rel=0.35)
    assert c.sharpness(np.zeros((0, 0), np.uint8)) == 0.0


@pytest.mark.parametrize("status, hint", [
    (TrackStatus.NO_FACE, "No face was found. Sit in front of the camera."),
    (TrackStatus.MULTIPLE_FACES, "More than one face was found. Only you should be in the picture."),
])
def test_without_exactly_one_face_nothing_else_is_judged(status, hint):
    checks = evaluate(textured(), TrackResult(status, 1.0))
    assert len(checks) == 7 and not any(check.passed for check in checks)
    assert first_hint(checks) == hint
    assert [check.hint for check in checks[1:]] == [""] * 6


# -------------------------------------------------------------- pose capture


def test_a_side_capture_passes_within_the_profile_window():
    found = {ch.key: ch for ch in c.judge_pose({**GOOD, "yaw_deg": 60.0}, "left")}
    assert found["pose"].passed and found["pose"].value == 60.0


@pytest.mark.parametrize("yaw, hint", [
    (20.0, "Turn your head further to the side."),
    (90.0, "That's turned too far: ease back a little so the face stays in view."),
])
def test_a_side_capture_outside_the_window_says_which_way_to_correct(yaw, hint):
    found = {ch.key: ch for ch in c.judge_pose({**GOOD, "yaw_deg": yaw}, "left")}
    assert not found["pose"].passed and found["pose"].hint == hint


def test_left_and_right_use_the_same_window_distinguished_by_sign():
    # Which side is not built into the check: a capture passes "left" or "right" the same way,
    # on whichever sign of yaw the camera measures. The caller (runtime.py) tells them apart and
    # requires the two captures to be opposite signs, so the same turn cannot be used for both.
    left = {ch.key: ch for ch in c.judge_pose({**GOOD, "yaw_deg": -60.0}, "left")}
    right = {ch.key: ch for ch in c.judge_pose({**GOOD, "yaw_deg": 60.0}, "right")}
    assert left["pose"].passed and right["pose"].passed


@pytest.mark.parametrize("pitch, pose, hint", [
    (5.0, "up", "Tilt your head back a little further, looking up."),
    (-50.0, "up", "That's tilted too far back: ease down a little."),
    (-25.0, "up", ""),
    (5.0, "down", "Tilt your head down a little further, chin toward your chest."),
    (50.0, "down", "That's tilted too far down: ease up a little."),
    (25.0, "down", ""),
])
def test_tilt_capture_checks_the_right_direction(pitch, pose, hint):
    found = {ch.key: ch for ch in c.judge_pose({**GOOD, "pitch_deg": pitch}, pose)}
    assert found["pose"].hint == hint
    assert found["pose"].passed == (hint == "")


def test_an_up_tilt_mistakenly_done_as_a_down_tilt_fails_the_down_check():
    # Looking up (negative pitch) should not quietly pass as a "down" capture.
    found = {ch.key: ch for ch in c.judge_pose({**GOOD, "pitch_deg": -25.0}, "down")}
    assert not found["pose"].passed


def test_pose_capture_still_checks_light_and_sharpness():
    found = {ch.key: ch for ch in c.judge_pose({**GOOD, "yaw_deg": 60.0, "sharpness": 6.0}, "left")}
    assert found["pose"].passed and not found["sharp"].passed


@pytest.mark.parametrize("status, hint", [
    (TrackStatus.NO_FACE, "No face was found. Sit in front of the camera."),
    (TrackStatus.MULTIPLE_FACES, "More than one face was found. Only you should be in the picture."),
])
def test_pose_capture_without_one_face_is_not_judged(status, hint):
    checks = c.evaluate_pose(textured(), TrackResult(status, 1.0), "left")
    assert len(checks) == 4 and not any(check.passed for check in checks)
    assert checks[0].hint == hint


def test_a_good_side_capture_passes_every_check():
    checks = c.evaluate_pose(textured(), tracked(yaw=60.0), "left")
    assert [ch.key for ch in checks] == ["face", "pose", "light", "sharp"]
    assert c.all_passed(checks)


def test_a_good_tilt_capture_passes_every_check():
    checks = c.evaluate_pose(textured(), tracked(pitch=-25.0), "up")
    assert c.all_passed(checks)
    checks = c.evaluate_pose(textured(), tracked(pitch=25.0), "down")
    assert c.all_passed(checks)


def test_a_pose_candidate_carries_its_pose_specific_origin_and_checks():
    candidate = make_pose_candidate(textured(), tracked(yaw=60.0), "left")
    assert candidate.passed and candidate.origin == "pose:left"
    assert [ch.key for ch in candidate.checks] == ["face", "pose", "light", "sharp"]
    assert candidate.neutral_face.shape == (256, 256, 3)


def test_a_pose_candidate_without_a_face_cannot_be_used():
    candidate = make_pose_candidate(textured(), TrackResult(TrackStatus.NO_FACE, 1.0), "up")
    assert not candidate.passed and candidate.landmarks is None and candidate.neutral_face is None


# -------------------------------------------------------------- candidates


def test_a_candidate_carries_its_checks_and_its_neutral_pose():
    candidate = make_candidate(textured(), tracked(), "camera")
    assert candidate.passed
    assert candidate.neutral_face.shape == (256, 256, 3)
    summary = candidate.summary()
    assert summary["origin"] == "camera" and summary["passed"] and summary["hint"] == "" and summary["tip"] == ""
    assert (summary["width"], summary["height"]) == (640, 360)
    assert [check["key"] for check in summary["checks"]] == ["face", "facing", "size", "framing", "light", "sharp", "expression"]


def test_a_candidate_without_a_face_cannot_be_used():
    candidate = make_candidate(textured(), TrackResult(TrackStatus.NO_FACE, 1.0), "upload")
    assert not candidate.passed and candidate.neutral_face is None and candidate.landmarks is None
    # An uploaded picture cannot be told to sit in front of the camera: it asks for a different photo instead.
    assert candidate.summary()["hint"] == "No face was found. Choose a different photo."


def test_the_sharper_frame_with_more_open_eyes_scores_higher():
    sharp, soft = textured(), cv2.GaussianBlur(textured(), (0, 0), 1.2)
    assert make_candidate(sharp, tracked(), "camera").score > make_candidate(soft, tracked(), "camera").score
    assert make_candidate(sharp, tracked(blink=0.05), "camera").score > make_candidate(sharp, tracked(blink=0.4), "camera").score


def test_large_pictures_are_brought_down_to_the_stored_size():
    assert prepare(np.zeros((3000, 4000, 3), np.uint8)).shape == (960, 1280, 3)
    small = np.zeros((360, 640, 3), np.uint8)
    assert prepare(small) is small


# ------------------------------------------------------------------- store


def enrol(store, value=90, origin="camera", signature=True):
    picture = np.full((360, 640, 3), value, dtype=np.uint8)
    neutral = np.full((256, 256, 3), value // 2, dtype=np.uint8)
    vector = np.linspace(0, 1, 512, dtype=np.float32) if signature else None
    return store.save(picture, neutral, vector, origin, [{"key": "face", "passed": True}])


def test_store_round_trip(tmp_path):
    store = EnrolmentStore(tmp_path / "enrolment")
    assert store.load() is None
    saved = enrol(store)
    loaded = store.load()
    assert loaded.id == saved.id and loaded.origin == "camera"
    assert loaded.enrolled_at == saved.enrolled_at
    assert np.array_equal(loaded.picture, saved.picture)
    assert np.array_equal(loaded.neutral_face, saved.neutral_face)
    assert np.allclose(loaded.signature, saved.signature)
    assert loaded.summary() == {"id": saved.id, "enrolled_at": saved.enrolled_at, "origin": "camera", "width": 640,
                                "height": 360, "checks": [{"key": "face", "passed": True}], "has_signature": True,
                                "match": None}


def test_store_keeps_one_enrolment_and_replaces_it(tmp_path):
    store = EnrolmentStore(tmp_path)
    first = enrol(store, value=90)
    second = enrol(store, value=180, origin="upload", signature=False)
    loaded = store.load()
    assert loaded.id == second.id != first.id
    assert loaded.picture.mean() == 180 and loaded.origin == "upload"
    assert loaded.signature is None  # the old signature must not outlive its picture
    assert not (tmp_path / "signature.npy").exists()


def test_store_remove_and_unfinished_saves(tmp_path):
    store = EnrolmentStore(tmp_path)
    enrol(store)
    (tmp_path / "record.json").unlink()  # as if the save had been cut short
    assert store.load() is None
    enrol(store)
    store.remove()
    assert store.load() is None
    assert list(tmp_path.iterdir()) == []
    store.remove()  # removing nothing is not an error


def test_the_face_signature_is_kept_apart_from_the_meeting_picture(tmp_path):
    store = EnrolmentStore(tmp_path)
    assert store.load_identity() is None
    vector = np.linspace(0, 1, 512, dtype=np.float32)
    face = store.save_identity(vector, [{"key": "face", "passed": True}])
    loaded = store.load_identity()
    assert loaded.id == face.id and loaded.verified_at == face.verified_at
    assert np.allclose(loaded.signature, vector)
    assert loaded.summary() == {"id": face.id, "verified_at": face.verified_at, "checks": [{"key": "face", "passed": True}], "poses": {}}
    # Only the signature and its record are on disk: no picture of the face is kept.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["identity.json", "identity.npy"]

    picture = store.save(np.full((360, 640, 3), 90, np.uint8), np.full((256, 256, 3), 40, np.uint8), vector, "upload",
                         [{"key": "identity", "passed": True}], match=0.62)
    assert store.load().match == 0.62 and store.load().summary()["match"] == 0.62
    store.remove()  # removing the meeting picture leaves the face
    assert store.load() is None and store.load_identity().id == face.id
    store.remove_identity()
    assert store.load_identity() is None and list(tmp_path.iterdir()) == []
    assert picture.match == 0.62


def test_a_picture_enrolled_from_the_camera_before_the_change_becomes_the_face(tmp_path):
    store = EnrolmentStore(tmp_path)
    older = enrol(store, origin="camera")  # taken live, with its own signature, as enrolment first worked
    face = store.load_identity()
    assert face.id == older.id and face.verified_at == older.enrolled_at
    assert np.allclose(face.signature, older.signature)
    store.remove()  # ... and it no longer depends on that picture
    assert store.load_identity().id == older.id


def test_an_uploaded_picture_from_before_the_change_is_not_taken_for_the_face(tmp_path):
    store = EnrolmentStore(tmp_path)
    enrol(store, origin="upload")  # nobody sat in front of the camera for this one
    assert store.load_identity() is None


# ----------------------------------------------------------- registered poses


def test_a_pose_cannot_be_registered_before_the_face_is(tmp_path):
    store = EnrolmentStore(tmp_path)
    with pytest.raises(ValueError, match="Verify your face"):
        store.save_pose("left", np.zeros(512, np.float32), 60.0, [])


def test_registered_poses_round_trip_with_the_face(tmp_path):
    store = EnrolmentStore(tmp_path)
    store.save_identity(np.linspace(0, 1, 512, dtype=np.float32), [{"key": "face", "passed": True}])
    left = np.full(512, 0.1, dtype=np.float32)
    up = np.full(512, 0.2, dtype=np.float32)
    store.save_pose("left", left, -60.0, [{"key": "pose", "passed": True}])
    store.save_pose("up", up, -25.0, [{"key": "pose", "passed": True}])

    face = store.load_identity()
    assert set(face.poses) == {"left", "up"}
    assert np.allclose(face.poses["left"]["signature"], left)
    assert face.poses["left"]["angle_deg"] == -60.0
    assert np.allclose(face.poses["up"]["signature"], up)
    assert face.summary()["poses"]["left"] == {"captured_at": face.poses["left"]["captured_at"], "angle_deg": -60.0}


def test_registering_a_pose_again_replaces_it(tmp_path):
    store = EnrolmentStore(tmp_path)
    store.save_identity(np.zeros(512, np.float32), [])
    store.save_pose("left", np.full(512, 0.1, np.float32), 50.0, [])
    store.save_pose("left", np.full(512, 0.9, np.float32), 70.0, [])
    face = store.load_identity()
    assert len(face.poses) == 1
    assert face.poses["left"]["angle_deg"] == 70.0
    assert np.allclose(face.poses["left"]["signature"], 0.9)


def test_verifying_the_face_again_clears_poses_registered_for_the_old_one(tmp_path):
    store = EnrolmentStore(tmp_path)
    store.save_identity(np.zeros(512, np.float32), [])
    store.save_pose("left", np.full(512, 0.1, np.float32), 50.0, [])
    store.save_identity(np.ones(512, np.float32), [])  # the face is verified again, a new signature
    assert store.load_identity().poses == {}


def test_forgetting_the_face_removes_its_poses_too(tmp_path):
    store = EnrolmentStore(tmp_path)
    store.save_identity(np.zeros(512, np.float32), [])
    store.save_pose("left", np.full(512, 0.1, np.float32), 50.0, [])
    store.remove_identity()
    assert store.load_identity() is None
    assert list(tmp_path.iterdir()) == []  # nothing of the face or its poses left behind


def test_an_unknown_pose_key_is_rejected(tmp_path):
    store = EnrolmentStore(tmp_path)
    store.save_identity(np.zeros(512, np.float32), [])
    with pytest.raises(ValueError, match="Unknown pose"):
        store.save_pose("sideways", np.zeros(512, np.float32), 60.0, [])
