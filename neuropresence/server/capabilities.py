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
    "summary": "You take one picture of yourself. It is checked for sharpness, light, framing and a relaxed, "
               "front-facing face, because every output frame is made from it.",
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
        "note": "Runs below the target frame rate until it is optimized.",
    },
    {
        "key": "identity",
        "name": "Identity",
        "summary": "Checks that the output still looks like the enrolled person.",
        "status": "partial",
        "note": "Similarity is measured live. The automatic fallback on identity loss is not built yet.",
    },
    {
        "key": "consent",
        "name": "Consent and disclosure",
        "summary": "Animates only the enrolled user's own face and marks the output as synthetic.",
        "status": "planned",
        "note": None,
    },
    {
        "key": "virtual_camera",
        "name": "Virtual camera",
        "summary": "Delivers the output to meeting apps as an ordinary camera.",
        "status": "planned",
        "note": None,
    },
]
