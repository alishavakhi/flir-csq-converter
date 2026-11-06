"""CSQ container: frame table, random access, and recording-level inspection.

A ``.csq`` recording is a flat concatenation of complete FFF blocks. Framing is
derived from each block's own record directory (record offset + length), never
from a magic-byte search: ``FFF\\x00`` can occur inside compressed pixel data,
so scanning alone would cut frames in the wrong place.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CsqError, MalformedFrame, NotACsqFile, NotRadiometric
from .fff import (
    CAMERA_INFO,
    HEADER_SIZE,
    MAGIC,
    RECORD_SIZE,
    RawFrame,
    find_record,
    parse_header,
    parse_raw_frame,
    unpack_records,
)
from .metadata import CameraInfo, parse_camera_info

SCAN_RECOVERY_WINDOW = 1 << 20  # up to 1 MiB of garbage we are willing to skip


@dataclass(frozen=True)
class FrameExtent:
    offset: int
    size: int
    truncated: bool = False

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class CsqIndex:
    path: Path
    frames: list[FrameExtent] = field(default_factory=list)
    file_size: int = 0
    creator: str = ""
    format_version: int = 0
    big_endian: bool = False
    record_types: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leading_bytes: int = 0
    trailing_bytes: int = 0

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def complete_frames(self) -> int:
        return sum(1 for f in self.frames if not f.truncated)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_size_bytes": self.file_size,
            "n_frames": self.n_frames,
            "complete_frames": self.complete_frames,
            "creator": self.creator,
            "format_version": self.format_version,
            "big_endian": self.big_endian,
            "record_types": {f"0x{k:02x}": v for k, v in sorted(self.record_types.items())},
            "leading_bytes": self.leading_bytes,
            "trailing_bytes": self.trailing_bytes,
            "warnings": list(self.warnings),
        }


def _find_magic(fh, start: int, limit: int) -> int | None:
    pos = start
    while pos < limit:
        fh.seek(pos)
        chunk = fh.read(min(SCAN_RECOVERY_WINDOW, limit - pos))
        if not chunk:
            return None
        hits = [i for i in (chunk.find(m) for m in MAGIC) if i >= 0]
        if hits:
            return pos + min(hits)
        pos += max(1, len(chunk) - 3)
    return None


def _read_block(fh, offset: int, file_size: int) -> tuple[int, Any, list, bytes]:
    """Read header + record directory at ``offset`` and return the block size."""
    fh.seek(offset)
    head = fh.read(HEADER_SIZE)
    hdr = parse_header(head)
    dir_end = hdr.directory_offset + hdr.record_count * RECORD_SIZE
    if offset + dir_end > file_size:
        raise MalformedFrame(
            f"record directory at offset {offset} runs past the end of the file"
        )
    fh.seek(offset + hdr.directory_offset)
    raw_dir = fh.read(hdr.record_count * RECORD_SIZE)
    records = unpack_records(raw_dir, hdr.big_endian)
    size = dir_end
    for rec in records:
        if rec.length > 0:
            if rec.offset < HEADER_SIZE or rec.offset > file_size - offset:
                raise MalformedFrame(f"record 0x{rec.type:02x} points outside the file")
            size = max(size, rec.offset + rec.length)
    return size, hdr, records, raw_dir


def scan_csq(path: str | Path, *, progress=None) -> CsqIndex:
    """Build the frame table of a ``.csq``/``.seq`` file without decoding pixels.

    A truncated final frame is kept and marked: a recording that lost its last
    block because the camera was still writing is still usable, and the caller
    decides whether to convert the complete frames before it.
    """
    p = Path(path)
    if not p.is_file():
        raise NotACsqFile(f"file not found: {p}")
    size = p.stat().st_size
    if size < HEADER_SIZE:
        raise NotACsqFile(f"{p.name} is {size} bytes - too small to hold a CSQ frame")

    index = CsqIndex(path=p, file_size=size)
    with p.open("rb") as fh:
        first = _find_magic(fh, 0, size)
        if first is None:
            raise NotACsqFile(
                f"{p.name} contains no FLIR FFF frame header",
                hint="A .csq recording is a concatenation of FFF blocks. If the file was "
                     "produced by conversion software, export a radiometric sequence from "
                     "FLIR software instead.",
            )
        if first:
            index.leading_bytes = first
            index.warnings.append(f"{first} byte(s) of leading data before the first frame")
        cursor = first
        seen_first = False
        while cursor + HEADER_SIZE <= size:
            fh.seek(cursor)
            if fh.read(4) not in MAGIC:
                nxt = _find_magic(fh, cursor + 4, min(size, cursor + 4 + SCAN_RECOVERY_WINDOW))
                if nxt is None:
                    index.warnings.append(f"no frame header after offset {cursor}; scan stopped")
                    break
                index.warnings.append(
                    f"skipped {nxt - cursor} unparsable byte(s) before frame {len(index.frames)}"
                )
                cursor = nxt
                continue
            try:
                block_size, hdr, records, _ = _read_block(fh, cursor, size)
            except MalformedFrame as exc:
                nxt = _find_magic(fh, cursor + 4, min(size, cursor + 4 + SCAN_RECOVERY_WINDOW))
                if nxt is None:
                    index.warnings.append(f"frame at {cursor} is unreadable ({exc}); scan stopped")
                    break
                index.warnings.append(f"frame at {cursor} is unreadable ({exc}); resynchronised")
                cursor = nxt
                continue

            for rec in records:
                index.record_types[rec.type] = index.record_types.get(rec.type, 0) + 1
            if not seen_first:
                index.creator = hdr.creator
                index.format_version = hdr.version
                index.big_endian = hdr.big_endian
                seen_first = True

            if cursor + block_size > size:
                index.warnings.append(
                    f"last frame at offset {cursor} is incomplete: {size - cursor} of "
                    f"{block_size} bytes (the camera may still have been writing)"
                )
                index.frames.append(FrameExtent(cursor, size - cursor, truncated=True))
                cursor = size
                break
            index.frames.append(FrameExtent(cursor, block_size))
            cursor += block_size
            if progress and len(index.frames) % 512 == 0:
                progress(cursor / max(size, 1), f"indexed {len(index.frames)} frames")

        if not index.frames:
            raise NotACsqFile(
                f"no complete FFF frame found in {p.name}",
                hint="Re-copy the recording from the SD card; the file may be truncated.",
            )
        last_complete = next((f for f in reversed(index.frames) if not f.truncated), None)
        if last_complete is not None:
            index.trailing_bytes = max(0, size - last_complete.end)
            if index.trailing_bytes and not any(f.truncated for f in index.frames):
                index.warnings.append(
                    f"{index.trailing_bytes} trailing byte(s) after the last frame were ignored"
                )
    return index


class CsqReader:
    """Random-access reader over :class:`CsqIndex`.

    Frames are read on demand, so peak memory is one compressed frame plus its
    decoded array regardless of recording length.
    """

    def __init__(self, index: CsqIndex, *, png_byteswap: str = "auto",
                 allow_fallback: bool = False) -> None:
        self.index = index
        self.png_byteswap = png_byteswap
        self.allow_fallback = allow_fallback
        self._fh = index.path.open("rb")

    # -- lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001,S110  # pragma: no cover - double close is harmless
            pass

    def __enter__(self) -> "CsqReader":  # noqa: PYI034,UP037
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.index.n_frames

    # -- access --------------------------------------------------------------------
    def frame_bytes(self, i: int) -> bytes:
        ext = self.index.frames[int(i)]
        self._fh.seek(ext.offset)
        data = self._fh.read(ext.size)
        if len(data) != ext.size:
            raise MalformedFrame(f"frame {i} is shorter on disk than in the index")
        return data

    def read_frame(self, i: int) -> RawFrame:
        """Decode frame ``i`` into 16-bit sensor counts.

        Any failure is tagged with ``frame_index`` so callers can report *which*
        frame is unreadable and recovery mode can skip exactly that one.
        """
        if self.index.frames[int(i)].truncated:
            exc = MalformedFrame(f"frame {i} is incomplete (the recording was cut short)")
            exc.frame_index = int(i)
            raise exc
        try:
            frame = parse_raw_frame(self.frame_bytes(i), 0, png_byteswap=self.png_byteswap,
                                    allow_fallback=self.allow_fallback)
        except CsqError as exc:
            if getattr(exc, "frame_index", None) is None:
                exc.frame_index = int(i)
            raise
        if frame.raw.size == 0:
            exc = NotRadiometric(f"frame {i} decoded to an empty array")
            exc.frame_index = int(i)
            raise exc
        return frame

    def raw_payload(self, i: int) -> bytes | None:
        """The RawData payload bytes of frame ``i`` (for diagnostics / dumping)."""
        from .fff import RAW_DATA, find_record, parse_header, parse_records

        try:
            data = self.frame_bytes(i)
            header = parse_header(data, 0)
            rec = find_record(parse_records(data, 0, header), RAW_DATA)
        except CsqError:
            return None
        if rec is None or rec.length <= 0x20:
            return None
        start = rec.offset + 0x20
        if start >= len(data):
            return None
        return data[start:rec.offset + rec.length]

    def camera_info(self, i: int) -> CameraInfo:
        """Read only the CameraInfo record of frame ``i``.

        Touches a few kilobytes per frame (header + directory + one record), so
        every timestamp in a 20 000-frame recording can be read in seconds
        without decoding a single pixel.
        """
        ext = self.index.frames[int(i)]
        self._fh.seek(ext.offset)
        head = self._fh.read(HEADER_SIZE)
        hdr = parse_header(head)
        self._fh.seek(ext.offset + hdr.directory_offset)
        raw_dir = self._fh.read(hdr.record_count * RECORD_SIZE)
        rec = find_record(unpack_records(raw_dir, hdr.big_endian), CAMERA_INFO)
        if rec is None or rec.length <= 0:
            return CameraInfo()
        blob_len = min(rec.length, max(0, ext.size - rec.offset))
        if blob_len <= 0:
            return CameraInfo()
        self._fh.seek(ext.offset + rec.offset)
        return parse_camera_info(self._fh.read(blob_len))

    def iter_frames(self, indices: Any | None = None) -> Iterator[tuple[int, RawFrame]]:
        for i in range(len(self)) if indices is None else indices:
            yield int(i), self.read_frame(int(i))
