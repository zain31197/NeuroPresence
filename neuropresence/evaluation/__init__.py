from .stats import summarise, summarise_scores

__all__ = ["WarpingError", "landmark_jitter", "summarise", "summarise_scores"]


def __getattr__(name):
    # The flicker and jitter measures need PyTorch, which takes seconds to load. They are loaded on
    # first use, so that what needs only numbers (the call measurement, for one) does not wait for it.
    if name in ("WarpingError", "landmark_jitter"):
        from . import temporal
        return getattr(temporal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
