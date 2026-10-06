"""Stage 1 input: frames from a webcam or, for testing, a recorded video file."""

import cv2


class FrameSourceError(RuntimeError):
    """The camera or video file could not be opened."""


class FrameSource:
    """Yields BGR frames from a camera index (e.g. 0) or a video file path."""

    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.is_camera = isinstance(source, int)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            kind = "camera" if self.is_camera else "video file"
            raise FrameSourceError(f"Could not open {kind}: {source!r}")
        if self.is_camera:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Frames per second the source reports; 0 if it does not say.
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self):
        """Return the next frame, or None when the stream ends or a read fails."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
