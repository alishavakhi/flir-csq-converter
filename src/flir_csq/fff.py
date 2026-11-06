"""FLIR FFF (FLIR File Format) blocks: the unit a ``.csq`` recording is made of.

A ``.csq`` is a flat concatenation of complete FFF blocks, one per recorded
frame, written by the camera when "Radiometric recording (*.csq)" is selected.
FLIR's own T5xx manual states that the format "supports full radiometry but is
only supported by FLIR Systems software. The file does not include any visual
image information."

Layout implemented here (offsets from the start of a block), taken from
ExifTool's public ``Image::ExifTool::FLIR`` table and cross-checked against
flirpy's independent reader:

======================  =========================================================
0x00  char[4]           file format ID ``FFF\\0``
0x04  char[16]          creator, e.g. ``CAMCTRL`` / ``RTP``
0x14  uint32            format version (100, 101 seen)
0x18  uint32            offset of the record directory
0x1c  uint32            number of record-directory entries
0x20  uint32            next free index id
0x24  uint16            swap pattern
0x28  uint16[7]         spares
0x34  uint32[2]         reserved
0x3c  uint32            checksum
======================  =========================================================

Record-directory entries are 32 bytes:
``uint16 type, uint16 subtype, uint32 version, uint32 index_id, uint32 offset,
uint32 length, uint32 parent, uint32 object, uint32 checksum``.

Two records matter for a temperature conversion:

* type ``0x01`` RawData - ``uint16 byte-order sentinel`` (0x0002 when the
  record is little-endian), ``uint16 width`` at +0x02, ``uint16 height`` at
  +0x04, payload from +0x20. Subtype 1/2 = raw big/little-endian counts,
  3 = PNG, 4 = compressed (JPEG-LS on every T-series CSQ examined).
* type ``0x20`` CameraInfo - Planck constants, measurement parameters, camera
  identity and the per-frame timestamp (:mod:`flir_csq.metadata`).

Records this module does not interpret (measurement tools 0x21, palette 0x22,
GPS 0x2b, audio 0x24, ...) are listed but never silently dropped from the
report: unknown types are surfaced by ``flir-csq inspect`` so a future camera
firmware that changes the container is visible instead of being mis-decoded.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .codecs import classify_payload, decode_compressed
from .errors import MalformedFrame, NotRadiometric, UnsupportedPayload

MAGIC = (b"FFF\x00", b"AFF\x00")
HEADER = struct.Struct("<4s16sIIIIH7HIII")  # 64 bytes
RECORD = struct.Struct("<HHIIIIIII")  # 32 bytes
HEADER_SIZE = HEADER.size
RECORD_SIZE = RECORD.size

RAW_DATA = 0x01
CAMERA_INFO = 0x20
MEASUREMENT_INFO = 0x21
PALETTE_INFO = 0x22
PICTURE_IN_PICTURE = 0x2A
GPS_INFO = 0x2B
PARAMETER_INFO = 0x2E

RECORD_NAMES = {
    RAW_DATA: "RawData",
    CAMERA_INFO: "CameraInfo",
    MEASUREMENT_INFO: "MeasurementInfo",
    PALETTE_INFO: "PaletteInfo",
    PICTURE_IN_PICTURE: "PiP",
    GPS_INFO: "GPSInfo",
    PARAMETER_INFO: "ParameterInfo",
}


@dataclass(frozen=True)
class FffHeader:
    magic: bytes
    creator: str
    version: int
    directory_offset: int
    record_count: int
    next_index_id: int
    swap_pattern: int
    checksum: int
    big_endian: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "creator": self.creator,
            "format_version": self.version,
            "record_directory_offset": self.directory_offset,
            "record_count": self.record_count,
            "big_endian": self.big_endian,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class FffRecord:
    type: int
    subtype: int
    version: int
    index_id: int
    offset: int
    length: int
    parent: int
    object_number: int
    checksum: int

    @property
    def name(self) -> str:
        return RECORD_NAMES.get(self.type, f"0x{self.type:02x}")


@dataclass
class RawFrame:
    """One decoded radiometric frame."""

    raw: np.ndarray  # uint16, (height, width)
    width: int
    height: int
    subtype: int
    codec: str
    record_length: int
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.raw.shape[0]), int(self.raw.shape[1]))


def detect_big_endian(header_bytes: bytes, base: int = 0) -> bool:
    """The block does not declare its byte order; the format version does.

    Versions are small integers (100-200). Reading one as a huge number means
    the byte order is wrong (same rule ExifTool and flirpy apply).
    """
    le = struct.unpack_from("<I", header_bytes, base + 0x14)[0]
    if 0 < le < 1000:
        return False
    be = struct.unpack_from(">I", header_bytes, base + 0x14)[0]
    return 0 < be < 1000


def parse_header(data: bytes, base: int = 0, *, big_endian: bool | None = None) -> FffHeader:
    if data[base:base + 4] not in MAGIC:
        raise MalformedFrame("block does not start with an FFF header")
    big = detect_big_endian(data, base) if big_endian is None else big_endian
    vals = struct.unpack_from(">4s16sIIIIH7HIII", data, base) if big else HEADER.unpack_from(data, base)
    magic = vals[0]
    creator = vals[1].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    version, dir_off, n_records = int(vals[2]), int(vals[3]), int(vals[4])
    if n_records <= 0 or n_records > 4096:
        raise MalformedFrame(f"implausible record count {n_records}")
    if dir_off < HEADER_SIZE or dir_off > (1 << 26):
        raise MalformedFrame(f"implausible record-directory offset {dir_off}")
    return FffHeader(magic=magic, creator=creator, version=version, directory_offset=dir_off,
                     record_count=n_records, next_index_id=int(vals[5]),
                     swap_pattern=int(vals[6]), checksum=int(vals[16]), big_endian=big)


def unpack_records(raw_directory: bytes, big_endian: bool) -> list[FffRecord]:
    """Unpack already-read directory bytes into records."""
    rec_struct = struct.Struct(">HHIIIIIII") if big_endian else RECORD
    n = len(raw_directory) // RECORD_SIZE
    return [FffRecord(*rec_struct.unpack_from(raw_directory, i * RECORD_SIZE)) for i in range(n)]


def parse_records(data: bytes, base: int = 0, header: FffHeader | None = None) -> list[FffRecord]:
    hdr = header or parse_header(data, base)
    start = base + hdr.directory_offset
    end = start + hdr.record_count * RECORD_SIZE
    if end > len(data):
        raise MalformedFrame("record directory is truncated")
    return unpack_records(data[start:end], hdr.big_endian)


def frame_extent(data: bytes, base: int = 0) -> tuple[int, FffHeader, list[FffRecord]]:
    """Size of the block starting at ``base``, computed from its own records."""
    hdr = parse_header(data, base)
    records = parse_records(data, base, hdr)
    size = hdr.directory_offset + hdr.record_count * RECORD_SIZE
    for rec in records:
        if rec.length > 0:
            size = max(size, rec.offset + rec.length)
    return size, hdr, records


def find_record(records: list[FffRecord], record_type: int) -> FffRecord | None:
    for rec in records:
        if rec.type == record_type and rec.length > 0:
            return rec
    return None


def parse_raw_frame(
    data: bytes,
    base: int = 0,
    *,
    png_byteswap: str = "auto",
) -> RawFrame:
    """Decode the RawData record of the block at ``base`` into sensor counts."""
    hdr = parse_header(data, base)
    records = parse_records(data, base, hdr)
    rec = find_record(records, RAW_DATA)
    if rec is None:
        raise NotRadiometric(
            "frame has no raw-thermal-image record (type 0x01)",
            hint="This looks like a colourised export, not a radiometric CSQ frame.",
        )
    start, end = base + rec.offset, base + rec.offset + rec.length
    if start < base or end > len(data):
        raise MalformedFrame(
            f"raw data record points outside the block (offset {rec.offset}, {rec.length} bytes)"
        )
    blob = data[start:end]
    if len(blob) < 0x20:
        raise MalformedFrame("raw data record is shorter than its 32-byte header")

    sentinel = struct.unpack_from("<H", blob, 0)[0]
    if sentinel == 0x0002:
        endian = "<"
    elif struct.unpack_from(">H", blob, 0)[0] == 0x0002:
        endian = ">"
    else:  # unknown sentinel: every T-series sample examined is little-endian
        endian = "<"
    width = struct.unpack_from(endian + "H", blob, 0x02)[0]
    height = struct.unpack_from(endian + "H", blob, 0x04)[0]
    payload = blob[0x20:]
    warnings: list[str] = []
    kind = classify_payload(payload)

    if kind == "raw16":
        if width <= 0 or height <= 0:
            raise MalformedFrame(f"invalid raw dimensions {width}x{height}")
        need = width * height * 2
        if len(payload) < need:
            raise MalformedFrame(
                f"raw payload truncated: {len(payload)} bytes, {need} needed for {width}x{height}"
            )
        arr = np.frombuffer(payload, dtype=endian + "u2", count=width * height)
        arr = arr.reshape(height, width).astype(np.uint16)
        codec = f"raw {'big' if endian == '>' else 'little'}-endian 16-bit"
    elif kind in ("png", "tiff"):
        arr = _decode_pillow(payload, byteswap=png_byteswap if kind == "png" else "off")
        codec = f"{kind.upper()} (Pillow)"
        if arr.shape != (height, width):
            if arr.size == width * height:
                arr = arr.reshape(height, width)
            else:
                raise UnsupportedPayload(
                    f"{kind.upper()} payload is {arr.shape[1]}x{arr.shape[0]}, header says "
                    f"{width}x{height}"
                )
    else:
        arr, codec = decode_compressed(payload, width=width, height=height)
        if rec.subtype not in (4, 3) and kind in ("jpegls", "lossless jpeg"):
            warnings.append(
                f"record subtype {rec.subtype} but payload is {kind}; decoded by content"
            )

    if arr.dtype != np.uint16:
        arr = np.clip(np.rint(arr.astype(np.float64)), 0, 65535).astype(np.uint16)
    if int(arr.max()) == 0 and int(arr.min()) == 0:
        warnings.append("decoded frame is all zero - the payload may be corrupt")

    return RawFrame(raw=arr, width=width or int(arr.shape[1]), height=height or int(arr.shape[0]),
                    subtype=rec.subtype, codec=codec, record_length=len(blob), warnings=warnings)


def _decode_pillow(payload: bytes, *, byteswap: str = "auto") -> np.ndarray:
    """PNG/TIFF payloads may store the 16-bit counts in the wrong byte order.

    ExifTool documents that "most FLIR cameras using the PNG format seem to
    write the 16-bit raw image data in the wrong byte order". A swapped read
    spans almost the full 16-bit range, which no real scene does, so ``auto``
    swaps only when that signature is present.
    """
    import io as _io

    from PIL import Image

    arr = np.array(Image.open(_io.BytesIO(payload)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.uint16)
    if byteswap != "off" and arr.size:
        span = (int(arr.max()) - int(arr.min())) / 65535.0
        if byteswap == "on" or span > 0.5:
            arr = arr.byteswap()
    return arr
