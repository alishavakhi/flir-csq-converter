"""Error types for the FLIR CSQ converter.

Every error carries a human-readable ``hint`` that the CLI and the GUI print
verbatim, because the most common failures are missing decoder packages, a
colourised export instead of a radiometric recording, and files that are still
being written by the camera.
"""

from __future__ import annotations


class CsqError(Exception):
    """Base class for every handled error in this package."""

    exit_code = 2

    def __init__(self, message: str, *, hint: str | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - trivial
        text = self.message
        if self.detail:
            text += f"\n  detail: {self.detail}"
        if self.hint:
            text += f"\n  hint:   {self.hint}"
        return text


class NotACsqFile(CsqError):
    """The file is not a FLIR FFF/CSQ container, or contains no frames."""


class MalformedFrame(CsqError):
    """A frame header or record directory cannot be interpreted."""


class UnsupportedPayload(CsqError):
    """The raw thermal payload uses a codec that is not available or not known."""


class MissingCalibration(CsqError):
    """Planck constants are absent or invalid: temperature conversion is unsupported."""

    exit_code = 3


class NotRadiometric(CsqError):
    """The frame holds a colourised image, not per-pixel radiometric counts."""


class ConversionCancelled(CsqError):
    """The user cancelled the conversion."""

    exit_code = 130


class OutputError(CsqError):
    """Writing or verifying an output file failed."""
