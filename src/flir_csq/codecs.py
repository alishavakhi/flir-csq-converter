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

from .errors import NotRadiometric, UnsupportedPayload

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
                      height: int | None = None) -> tuple[np.ndarray, str]:
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
        arr, name = _decode_jpeg(payload, kind)
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


def _decode_jpeg(payload: bytes, kind: str) -> tuple[np.ndarray, str]:
    problems: list[str] = []

    try:
        import imagecodecs

        if kind == "jpegls":
            return np.asarray(imagecodecs.jpegls_decode(payload)), "imagecodecs/CharLS (JPEG-LS)"
        for fn_name in ("jpeg16_decode", "jpeg8_decode", "jpeg_decode"):
            fn = getattr(imagecodecs, fn_name, None)
            if fn is None:
                continue
            try:
                return np.asarray(fn(payload)), f"imagecodecs/{fn_name}"
            except Exception as exc:  # noqa: BLE001 - 16/8-bit attempt
                problems.append(f"{fn_name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"imagecodecs: {exc}")

    # pylibjpeg-libjpeg: the installed build does decode JPEG-LS (verified: identical
    # counts to CharLS on a real T1020 recording), and is the fallback when imagecodecs
    # cannot be installed.
    try:
        from libjpeg import decode as libjpeg_decode

        return np.asarray(libjpeg_decode(payload)), "pylibjpeg-libjpeg"
    except Exception as exc:  # noqa: BLE001
        problems.append(f"pylibjpeg-libjpeg: {exc}")

    summary = (
        "the frame is JPEG-LS compressed and no JPEG-LS decoder is installed"
        if kind == "jpegls"
        else "no installed decoder could read this JPEG stream"
    )
    raise UnsupportedPayload(
        f"cannot decode the {kind} payload ({summary})",
        hint="pip install imagecodecs   (bundles CharLS, the JPEG-LS codec FLIR uses)",
        detail="; ".join(problems[:3]) or None,
    )
