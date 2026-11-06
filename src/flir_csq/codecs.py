"""Decoders for the compressed 16-bit payloads that FLIR writes into CSQ frames.

What a CSQ frame actually contains
----------------------------------
The raw-thermal-image record of a T-series ``.csq`` frame does **not** hold a
display image. It holds the sensor's 14-bit measurement counts, packed as 16-bit
samples and compressed with **JPEG-LS** (ITU-T T.87, frame header ``FF F7``).
Older cameras and single-shot JPEGs use uncompressed little/big-endian counts,
a PNG, or lossless JPEG (SOF3) instead, so all four are handled here.

Nothing in this module converts counts to temperature, and nothing ever accepts a
colourised preview as a substitute: a 3- or 4-channel decode is rejected.

Codecs, licences and why there is a fallback chain
---------------------------------------------------
===================  ==================  ==========================================
package              licence             handles
===================  ==================  ==========================================
imagecodecs (CharLS) BSD-3-Clause        JPEG-LS (16-bit), lossless JPEG, PNG, TIFF
pylibjpeg-libjpeg    MIT                 JPEG-LS (verified), lossless and plain JPEG
Pillow               HPND                PNG and TIFF payloads only
===================  ==================  ==========================================

No vendor SDK is used. If no installed codec can read a payload the caller gets
:class:`~flir_csq.errors.UnsupportedPayload` with the exact install command,
never a decoded-but-wrong image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import NotRadiometric, PayloadDecodeError, UnsupportedPayload

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
JPEG_SOI = b"\xff\xd8"

SOF0 = 0xC0  # baseline DCT, 8-bit, lossy
SOF3 = 0xC3  # lossless sequential
SOF7 = 0xC7
SOF11 = 0xCB
SOF15 = 0xCF
SOF55 = 0xF7  # JPEG-LS


@dataclass(frozen=True)
class Codec:
    name: str
    jpegls: bool
    available: bool
    version: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "jpegls": self.jpegls, "available": self.available,
                "version": self.version, "note": self.note}


def jpeg_profile(payload: bytes) -> str:
    """Classify a JPEG stream as ``jpegls``, ``lossless jpeg``, ``jpeg`` or ``unknown``.

    Reads the first frame header (SOFn) rather than guessing from the magic
    bytes: JPEG-LS and lossless JPEG both start with the ordinary ``FF D8``.
    """
    if not payload.startswith(JPEG_SOI):
        return "unknown"
    i, n = 2, min(len(payload), 1 << 16)
    while i + 3 < n:
        if payload[i] != 0xFF:
            i += 1
            continue
        marker = payload[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:  # EOI
            break
        seg_len = int.from_bytes(payload[i + 2:i + 4], "big")
        if seg_len < 2:
            break
        if marker == SOF55:
            return "jpegls"
        if marker in (SOF3, SOF7, SOF11, SOF15):
            return "lossless jpeg"
        if SOF0 <= marker <= SOF15:
            return "jpeg"
        i += 2 + seg_len
    return "jpeg"


@dataclass(frozen=True)
class JpegFrameHeader:
    """Dimensions declared by a JPEG/JPEG-LS frame header.

    Read before decoding on purpose: a corrupt stream can claim enormous
    dimensions, and a decoder that trusts them allocates the frame before it
    discovers the data is garbage (measured: a 60000x60000 header asks numpy for
    6.7 GiB). Validating against the recording's own dimensions removes that
    failure mode and turns it into a normal per-frame error.
    """

    kind: str
    precision: int | None = None
    height: int | None = None
    width: int | None = None
    components: int | None = None

    @property
    def pixels(self) -> int | None:
        if self.height is None or self.width is None:
            return None
        return int(self.height) * int(self.width)

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "precision": self.precision, "height": self.height,
                "width": self.width, "components": self.components}


def jpeg_has_eoi(payload: bytes, window: int = 64) -> bool:
    """True when a JPEG stream carries an end-of-image marker near its end.

    A truncated frame is the commonest corruption in a recording that was copied
    while the camera was still writing. CharLS rejects such streams, but the
    fallback decoder (pylibjpeg-libjpeg) happily decodes them and returns *wrong*
    pixels (measured: 25 286 pixels off by up to 247 counts on a payload cut by
    5 KB). Checking the marker turns that silent corruption into a clean error.
    """
    return payload.rfind(b"\xff\xd9", max(0, len(payload) - window)) != -1


def jpeg_frame_header(payload: bytes) -> JpegFrameHeader:
    """Parse the first SOFn segment of a JPEG stream (dimensions and precision)."""
    if not payload.startswith(JPEG_SOI):
        return JpegFrameHeader("unknown")
    i, n = 2, min(len(payload), 1 << 16)
    while i + 9 < n:
        if payload[i] != 0xFF:
            i += 1
            continue
        marker = payload[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        seg_len = int.from_bytes(payload[i + 2:i + 4], "big")
        if seg_len < 2:
            break
        is_sof = marker == SOF55 or (SOF0 <= marker <= 0xCF
                                     and marker not in (0xC4, 0xC8, 0xCC))
        if is_sof:
            kind = "jpegls" if marker == SOF55 else (
                "lossless jpeg" if marker in (SOF3, SOF7, SOF11, SOF15) else "jpeg")
            precision = payload[i + 4] if i + 5 <= n else None
            height = int.from_bytes(payload[i + 5:i + 7], "big") if i + 7 <= n else None
            width = int.from_bytes(payload[i + 7:i + 9], "big") if i + 9 <= n else None
            components = payload[i + 9] if i + 10 <= n else None
            return JpegFrameHeader(kind, precision, height, width, components)
        i += 2 + seg_len
    return JpegFrameHeader("unknown")


def available_codecs() -> list[Codec]:
    """Probe what this interpreter can decode, without raising."""
    out: list[Codec] = []
    try:
        import imagecodecs

        out.append(Codec("imagecodecs", hasattr(imagecodecs, "jpegls_decode"), True,
                         str(getattr(imagecodecs, "__version__", "?"))))
    except Exception as exc:  # noqa: BLE001 - probing must never raise
        out.append(Codec("imagecodecs", False, False, note=type(exc).__name__))
    try:
        from libjpeg import decode  # noqa: F401

        out.append(Codec("pylibjpeg-libjpeg", _libjpeg_handles_jpegls(), True))
    except Exception as exc:  # noqa: BLE001
        out.append(Codec("pylibjpeg-libjpeg", False, False, note=type(exc).__name__))
    try:
        import PIL

        out.append(Codec("Pillow", False, True, str(getattr(PIL, "__version__", "?"))))
    except Exception as exc:  # noqa: BLE001
        out.append(Codec("Pillow", False, False, note=type(exc).__name__))
    return out


def _libjpeg_handles_jpegls() -> bool:
    """Probe the installed pylibjpeg-libjpeg build with a 4x4 JPEG-LS stream."""
    try:
        import imagecodecs
        import numpy as np
        from libjpeg import decode

        probe = imagecodecs.jpegls_encode(np.arange(16, dtype=np.uint16).reshape(4, 4))
        return np.asarray(decode(probe)).shape == (4, 4)
    except Exception:  # noqa: BLE001 - probe only
        return False


def can_decode_jpegls() -> bool:
    return any(c.available and c.jpegls for c in available_codecs())


def classify_payload(payload: bytes) -> str:
    if payload.startswith(PNG_MAGIC):
        return "png"
    if payload[:4] in TIFF_MAGIC:
        return "tiff"
    if payload.startswith(JPEG_SOI):
        return jpeg_profile(payload)
    return "raw16"


def decode_compressed(payload: bytes, *, width: int | None = None,
                      height: int | None = None,
                      allow_fallback: bool = False) -> tuple[np.ndarray, str]:
    """Decode a compressed payload into ``uint16`` counts.

    Returns ``(counts, codec_description)``. Raises
    :class:`~flir_csq.errors.UnsupportedPayload` when no installed codec can read
    the stream, and :class:`~flir_csq.errors.NotRadiometric` when it decodes to a
    colour image.
    """
    kind = classify_payload(payload)
    if kind == "png":
        arr, name = _decode_pillow(payload), "PNG (Pillow)"
    elif kind == "tiff":
        arr, name = _decode_pillow(payload), "TIFF (Pillow)"
    elif kind in ("jpegls", "lossless jpeg", "jpeg"):
        header = jpeg_frame_header(payload)
        if width and height and header.height and header.width:
            if (header.width, header.height) != (width, height):
                raise PayloadDecodeError(
                    f"payload header declares {header.width}x{header.height}, but the frame "
                    f"record stores {width}x{height}",
                    hint="The frame is corrupt (or was written by a different mode). It is not "
                         "decoded, because trusting those dimensions can allocate gigabytes.",
                    codec=kind,
                    payload_size=len(payload),
                )
        elif width and height and header.pixels is None:
            raise PayloadDecodeError(
                f"{kind} payload has no readable frame header ({len(payload)} bytes)",
                hint="The frame is corrupt or truncated; nothing was decoded.",
                codec=kind, payload_size=len(payload),
            )
        if kind in ("jpegls", "lossless jpeg", "jpeg") and not jpeg_has_eoi(payload):
            raise PayloadDecodeError(
                f"{kind} payload is truncated: no end-of-image marker in the last 64 bytes "
                f"({len(payload)} bytes)",
                hint="The recording was probably copied or read while the camera was still "
                     "writing. Re-copy the file; recovery mode can skip this frame.",
                codec=kind, payload_size=len(payload),
            )
        arr, name = _decode_jpeg(payload, kind, width=width, height=height,
                                 allow_fallback=allow_fallback)
    else:  # pragma: no cover - only reached for uncompressed payloads
        raise UnsupportedPayload(f"unrecognised payload ({len(payload)} bytes)")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] in (3, 4):
            raise NotRadiometric(
                f"decoded payload is a {arr.shape[2]}-channel image, i.e. a colourised "
                "preview",
                hint="Record with radiometric storage (*.csq) enabled on the camera, or "
                     "export a temperature matrix from FLIR software.",
            )
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise UnsupportedPayload(f"decoded payload has shape {arr.shape}, expected 2-D")
    if width and height and arr.shape != (height, width):
        if arr.size == width * height:
            arr = arr.reshape(height, width)
        else:
            raise UnsupportedPayload(
                f"decoded payload is {arr.shape[1]}x{arr.shape[0]}, but the record header "
                f"says {width}x{height}"
            )
    if arr.dtype != np.uint16:
        if arr.dtype.kind == "f":
            arr = np.rint(arr)
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr, name


def _decode_pillow(payload: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(payload)))


def _decode_jpeg(payload: bytes, kind: str, *, width: int | None = None,
                 height: int | None = None,
                 allow_fallback: bool = False) -> tuple[np.ndarray, str]:
    """Decode a JPEG-family payload, preserving the codec's own error.

    The decoded frame is written into a buffer of the recording's dimensions when
    they are known, so a corrupt header cannot make the codec allocate more than
    one frame.
    """
    attempts: list[tuple[str, str]] = []
    installed: list[str] = []
    cause: BaseException | None = None
    imagecodecs = None

    try:
        import imagecodecs as _imagecodecs
    except Exception as exc:  # noqa: BLE001 - import problems are "not installed"
        attempts.append(("imagecodecs", f"not importable: {type(exc).__name__}: {exc}"))
    else:
        imagecodecs = _imagecodecs

    if imagecodecs is not None:
        installed.append("imagecodecs")
        if kind == "jpegls":
            try:
                out = np.empty((height, width), dtype=np.uint16) if width and height else None
                try:
                    arr = (imagecodecs.jpegls_decode(payload, out=out) if out is not None
                           else imagecodecs.jpegls_decode(payload))
                except TypeError:
                    # older imagecodecs without out=; the caller already bounded the size
                    arr = imagecodecs.jpegls_decode(payload)
                return np.asarray(arr), "imagecodecs/CharLS (JPEG-LS)"
            except Exception as exc:  # noqa: BLE001 - the codec's own failure
                cause = cause or exc
                attempts.append(("imagecodecs/CharLS (JPEG-LS)",
                                 f"{type(exc).__name__}: {exc}"))
                if not allow_fallback:
                    # Do not hand a stream the primary codec rejected to a second
                    # decoder that may accept it and return wrong pixels.
                    raise _payload_error(kind, attempts, cause, installed, len(payload)) from exc
        else:
            for fn_name in ("jpeg16_decode", "jpeg8_decode", "jpegsof3_decode", "jpeg_decode"):
                fn = getattr(imagecodecs, fn_name, None)
                if fn is None:
                    continue
                try:
                    return np.asarray(fn(payload)), f"imagecodecs/{fn_name}"
                except Exception as exc:  # noqa: BLE001 - try the next profile
                    cause = cause or exc
                    attempts.append((f"imagecodecs/{fn_name}", f"{type(exc).__name__}: {exc}"))

    try:
        from libjpeg import decode as libjpeg_decode
    except Exception as exc:  # noqa: BLE001
        attempts.append(("pylibjpeg-libjpeg", f"not importable: {type(exc).__name__}: {exc}"))
    else:
        installed.append("pylibjpeg-libjpeg")
        try:
            return np.asarray(libjpeg_decode(payload)), "pylibjpeg-libjpeg"
        except Exception as exc:  # noqa: BLE001
            cause = cause or exc
            attempts.append(("pylibjpeg-libjpeg", f"{type(exc).__name__}: {exc}"))

    if not installed:
        summary = ("no JPEG-LS decoder is installed" if kind == "jpegls"
                   else "no JPEG decoder is installed")
        raise UnsupportedPayload(
            f"cannot decode the {kind} payload: {summary}",
            hint="pip install imagecodecs   (bundles CharLS) or "
                 "pip install pylibjpeg pylibjpeg-libjpeg",
            detail="; ".join(f"{name}: {text}" for name, text in attempts) or None,
        )

    raise _payload_error(kind, attempts, cause, installed, len(payload))


def _payload_error(kind: str, attempts: list[tuple[str, str]],
                   cause: BaseException | None, installed: list[str],
                   payload_size: int) -> PayloadDecodeError:
    first = attempts[0][1] if attempts else "unknown error"
    return PayloadDecodeError(
        f"the {kind} decoder could not read this payload ({first})",
        hint="The frame is corrupt or uses an unsupported JPEG-LS profile. "
             "Re-run with --on-decode-error recover to keep the rest of the recording, "
             "and --dump-bad-frames DIR to save the payload for analysis.",
        detail="; ".join(f"{name}: {text}" for name, text in attempts),
        codec=", ".join(installed),
        payload_size=payload_size,
        cause=cause,
        attempts=attempts,
    )


