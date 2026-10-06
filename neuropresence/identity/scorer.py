"""Identity similarity between two faces (CSIM)."""

import numpy as np

from ..capture import FaceTracker
from .align import align_face, five_points
from .arcface import ArcFaceEmbedder


class IdentityScorer:
    """Scores how alike two faces are: the cosine similarity of their ArcFace
    embeddings, from -1 to 1. The same face in the same image scores 1.0.

    This module does not depend on the reenactment core, so the same scorer
    can serve the evaluation, the live identity monitor and the consent gate.
    """

    def __init__(self, embedder=None, tracker=None):
        self._embedder = embedder or ArcFaceEmbedder()
        self._tracker = tracker  # made on first use, when landmarks are not supplied

    def embed(self, image_bgr, landmarks=None):
        """Embed the face in the image. Returns None unless exactly one face is found.

        Pass the 478 MediaPipe landmarks if they are already known for this
        image, to skip a second face search.
        """
        if landmarks is None:
            if self._tracker is None:
                self._tracker = FaceTracker(video=False)
            track = self._tracker.process(image_bgr)
            if not track.ok:
                return None
            landmarks = track.landmarks
        return self._embedder.embed(align_face(image_bgr, five_points(landmarks)))

    @staticmethod
    def similarity(embedding_a, embedding_b):
        return float(np.dot(embedding_a, embedding_b))

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
