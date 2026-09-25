"""Build-Pipeline und Assoziations-CLI."""

from .build_config import BuildConfig

__all__ = ["BuildConfig", "build"]


def __getattr__(name):
    if name == "build":
        from .build import build

        return build
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
