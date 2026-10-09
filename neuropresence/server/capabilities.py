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
        "note": "Runs below the target frame rate until it is optimized.",
    },
    {
        "key": "identity",
        "name": "Identity",
        "summary": "Checks that the output still looks like the enrolled person.",
        "status": "working",
        "note": "Similarity is measured twice a second. A lasting drop takes a fresh neutral pose, and if that does "
                "not help the still picture is shown until you resume.",
    },
    {
        "key": "consent",
        "name": "Consent and disclosure",
        "summary": "Animates only the enrolled user's own face and marks the output as synthetic.",
        "status": "working",
        "note": "Before a face is verified and before every camera session, two actions chosen at random are asked "
                "for, and the face is matched all the way through. During the session the camera face is compared "
                "once a second. Every output frame carries the label \"AI reenacted\".",
    },
    {
        "key": "virtual_camera",
        "name": "Virtual camera",
        "summary": "Delivers the output to meeting apps as an ordinary camera.",
        "status": "planned",
        "note": None,
    },
]
