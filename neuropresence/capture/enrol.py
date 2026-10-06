"""Pick the frame that becomes the source image."""


def first_frontal_frame(frames, tracker, max_frames=150, max_angle_deg=20):
    """Return the first frame showing exactly one roughly front-facing face.

    frames is a FrameSource and tracker a FaceTracker. Returns None if no such
    frame arrives within max_frames or the stream ends first.
    """
    for _ in range(max_frames):
        frame = frames.read()
        if frame is None:
            return None
        track = tracker.process(frame)
        if track.ok and all(abs(angle) <= max_angle_deg for angle in track.pose_deg):
            return frame
    return None
