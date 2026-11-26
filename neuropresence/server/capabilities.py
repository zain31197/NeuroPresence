"""What each pipeline stage can do today.

The landing page and the app read this list, so what they claim is built
cannot drift away from the code. Update the status here when a stage lands.

status: "working" - built and running; "partial" - part of it is built;
"planned" - designed, not built.
"""

# Done once, before any session: not one of the per-frame stages below.
ENROLMENT = {
    "key": "enrolment",
    "name": "Enrolment",
    "summary": "Your face is verified live with the camera, and only its signature is kept. Then you upload the "
               "picture you want people to see: it is checked for quality, and accepted only if it shows the same face.",
    "status": "working",
    "note": None,
}

STAGES = [
    {
        "key": "capture",
        "name": "Capture",
        "summary": "Reads the camera and finds the face and its 478 landmarks in every frame.",
        "status": "working",
        "note": None,
    },
    {
        "key": "motion",
        "name": "Motion encoding",
        "summary": "Turns the tracked face into head pose, lip and expression signals.",
        "status": "working",
        "note": None,
    },
    {
        "key": "reenactment",
        "name": "Reenactment",
        "summary": "Animates the enrolled picture with those signals and blends it back into the frame.",
        "status": "working",
        "note": "Runs at 20 to 22 frames a second in a live session on the RTX 5050, against a target of 24. The render "
                "and delay targets are met.",
    },
    {
        "key": "identity",
        "name": "Identity",
        "summary": "Checks that the output still looks like the enrolled person.",
        "status": "working",
        "note": "Similarity is measured twice a second. A lasting drop takes a fresh neutral pose, and if that does "
                "not help the still picture is shown until you sit at rest again, when it resumes by itself.",
    },
    {
        "key": "consent",
        "name": "Consent and disclosure",
        "summary": "Animates only the enrolled user's own face and marks the output as synthetic.",
        "status": "working",
        "note": "Before a face is verified and before every camera session, two actions chosen at random are asked "
                "for, and the face is matched all the way through; after three that do not pass, the next has to wait. "
                "During the session the camera face is compared twice a second. Every output frame carries the label "
                "\"AI reenacted\".",
    },
    {
        "key": "virtual_camera",
        "name": "Virtual camera",
        "summary": "Delivers the output to meeting apps as an ordinary camera.",
        "status": "working",
        "note": "Switched on in the Live Studio. A meeting app sees the output of a running session, with its mark and "
                "its fallbacks, and a plain card at any other time. A virtual microphone holds the voice back by as "
                "long as the picture takes, so lips and words arrive together. Needs OBS Virtual Camera and "
                "VB-Audio Virtual Cable on Windows, or v4l2loopback and a PipeWire null sink on Linux.",
    },
]

# What drives the picture. Both run through the same stages from reenactment on: the identity check, the mark
# and the virtual camera apply to each.
MODES = [
    {
        "key": "camera",
        "name": "Camera on",
        "summary": "Your face at the camera drives your picture: head pose, lips, eyes and expression, as you move.",
        "status": "working",
        "note": "About 20 to 22 frames a second and 95 ms from camera to picture on the RTX 5050, measured during calls "
                "in Google Meet and Zoom.",
    },
    {
        "key": "voice",
        "name": "Camera off",
        "summary": "Your voice drives your picture: a speech to motion model turns what you say into lips, head and "
                   "blinks, and the meeting hears you in step with it.",
        "status": "working",
        "note": "Starts with the same check at the camera, then the camera is closed. The picture runs about 0.8 s "
                "behind the voice, and the voice for the meeting is held back by as much.",
    },
]

# How fast a live session runs: the one place the app takes these figures from, so every screen says the same.
# Measured on the RTX 5050 (130 W): 21.5 to 22.0 frames a second with no meeting (11 October and 20 November 2026),
# 20.0 to 20.2 as medians during calls in Google Meet and Zoom (23 October 2026), 95 ms from camera to picture.
# A benchmark run is slower (16.6 to 17.6) because it tracks and draws one after the other, where a session does
# both at once; it is shown as what it is.
LIVE = {
    "fps": [20, 22],
    "delay_ms": 95,
    "measured": "Live sessions on an RTX 5050, with no meeting and during calls in Google Meet and Zoom.",
}
