"""Headless batch conversion of a directory of recordings.

Designed for a Linux server with no display: no Qt import, no GUI, one file at
a time, a log file, and per-file error isolation. A corrupt recording fails its
own entry and leaves every other conversion intact.

What "already done" means
-------------------------
Completed work is recorded in ``<output>/.flir_csq_state.json`` together with a
cheap signature of the input (size, mtime, SHA-256 of the first and last 64 KiB)
and the exact conversion settings. A recording is skipped only when

* the state entry says ``ok``,
* the signature of the input is unchanged,
* the conversion settings are unchanged, and
* the output file still exists with the recorded size.

Anything else is converted again. ``--overwrite`` ignores all of that, and
``--dry-run`` shows the plan without writing anything.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .converter import ConversionOptions, ConversionResult, convert
from .errors import ConversionCancelled, CsqError
from .exporter import CONVERTER_VERSION, default_output_path

LOG = logging.getLogger("flir_csq.batch")

STATE_VERSION = 1
STATE_FILENAME = ".flir_csq_state.json"
SUMMARY_FILENAME = "batch_summary.json"
DEFAULT_EXTENSIONS = (".csq", ".seq")


# ----------------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------------
def discover_recordings(input_dir: str | Path, *, recursive: bool = True,
                        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
                        pattern: str | None = None) -> list[Path]:
    """Every recording under ``input_dir``, sorted by path for a stable order."""
    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"input directory not found: {root}")
    wanted = tuple(ext.lower() for ext in extensions)
    found: list[Path] = []

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if pattern and not path.match(pattern):
                    continue
                if path.suffix.lower() in wanted:
                    found.append(path)
    else:
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            if pattern and not path.match(pattern):
                continue
            if path.suffix.lower() in wanted:
                found.append(path)
    return found


# ----------------------------------------------------------------------------------
# state
# ----------------------------------------------------------------------------------
def signature(path: Path) -> dict[str, Any]:
    """Cheap identity of a recording: size, mtime and both ends hashed."""
    stat = path.stat()
    with path.open("rb") as fh:
        head = fh.read(1 << 16)
        tail_size = min(1 << 16, stat.st_size)
        fh.seek(max(0, stat.st_size - tail_size))
        tail = fh.read(tail_size)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256_head": hashlib.sha256(head).hexdigest(),
        "sha256_tail": hashlib.sha256(tail).hexdigest(),
    }


def load_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / STATE_FILENAME
    if not path.is_file():
        return {"version": STATE_VERSION, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOG.warning("state file %s is unreadable; starting from an empty state", path)
        return {"version": STATE_VERSION, "files": {}}
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return {"version": STATE_VERSION, "files": {}}
    data.setdefault("version", STATE_VERSION)
    return data


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    """Atomic write: a crash mid-write must not destroy the progress record."""
    path = output_dir / STATE_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    state["version"] = STATE_VERSION
    state["updated_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    payload = json.dumps(state, indent=2, sort_keys=True, default=str)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------
# options and results
# ----------------------------------------------------------------------------------
@dataclass
class BatchOptions:
    input_dir: Path
    output_dir: Path
    formats: tuple[str, ...] = ("h5",)
    recursive: bool = True
    skip_existing: bool = True
    overwrite: bool = False
    continue_on_error: bool = True
    dry_run: bool = False
    preserve_tree: bool = True
    naming: str = "suffix"  # "suffix" -> name_temperature.h5, "stem" -> name.h5
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    pattern: str | None = None
    limit: int | None = None
    # conversion settings (passed straight through)
    save_raw: bool = True
    save_mask: bool = False
    save_metadata: bool = True
    mode: str = "auto"
    stride: int = 1
    compression_level: int = 4
    mat_version: str = "auto"
    on_decode_error: str = "fail"
    dump_bad_frames: Path | None = None
    allow_fallback_decoder: bool = False
    emissivity: float | None = None
    object_distance_m: float | None = None
    reflected_temperature_c: float | None = None
    atmospheric_temperature_c: float | None = None
    relative_humidity: float | None = None
    window_transmission: float | None = None
    window_temperature_c: float | None = None
    log_file: Path | None = None
    summary_path: Path | None = None
    progress: Callable[[float, str], None] | None = None

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        if self.naming not in ("suffix", "stem"):
            raise ValueError("naming must be 'suffix' or 'stem'")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

    def conversion_key(self) -> str:
        """Settings that change the numbers: a change must force reconversion."""
        payload = {
            "formats": list(self.formats),
            "save_raw": self.save_raw,
            "save_mask": self.save_mask,
            "mode": self.mode,
            "stride": self.stride,
            "compression_level": self.compression_level,
            "mat_version": self.mat_version,
            "on_decode_error": self.on_decode_error,
            "emissivity": self.emissivity,
            "object_distance_m": self.object_distance_m,
            "reflected_temperature_c": self.reflected_temperature_c,
            "atmospheric_temperature_c": self.atmospheric_temperature_c,
            "relative_humidity": self.relative_humidity,
            "window_transmission": self.window_transmission,
            "window_temperature_c": self.window_temperature_c,
            "converter_version": CONVERTER_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class FileOutcome:
    input: Path
    status: str  # ok | ok-with-gaps | skipped | failed | planned
    output: Path | None = None
    frames: int = 0
    recovered_frames: int = 0
    duration_s: float = 0.0
    error: str | None = None
    hint: str | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input),
            "output": str(self.output) if self.output else None,
            "status": self.status,
            "frames": self.frames,
            "recovered_frames": self.recovered_frames,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "hint": self.hint,
            "warnings": self.warnings[:20],
        }


@dataclass
class BatchResult:
    input_dir: Path
    output_dir: Path
    outcomes: list[FileOutcome] = field(default_factory=list)
    started_utc: str = ""
    finished_utc: str = ""
    duration_s: float = 0.0
    log_file: Path | None = None
    summary_path: Path | None = None
    state_path: Path | None = None
    dry_run: bool = False

    @property
    def converted(self) -> int:
        return sum(1 for o in self.outcomes if o.status in ("ok", "ok-with-gaps"))

    @property
    def with_gaps(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "ok-with-gaps")

    @property
    def recovered_frames(self) -> int:
        return sum(o.recovered_frames for o in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    @property
    def planned(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "planned")

    @property
    def frames(self) -> int:
        """Frames actually written in this run (skipped files are not re-counted)."""
        return sum(o.frames for o in self.outcomes if o.status in ("ok", "ok-with-gaps"))

    @property
    def frames_skipped(self) -> int:
        return sum(o.frames for o in self.outcomes if o.status == "skipped")

    @property
    def exit_code(self) -> int:
        if self.failed and (self.converted or self.skipped):
            return 4  # partial success; the successful files are usable
        if self.failed:
            return 2
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "converter": {"name": "flir-csq", "version": CONVERTER_VERSION},
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "dry_run": self.dry_run,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "duration_s": round(self.duration_s, 3),
            "totals": {
                "files_found": len(self.outcomes),
                "converted": self.converted,
                "skipped": self.skipped,
                "failed": self.failed,
                "planned": self.planned,
                "files_with_gaps": self.with_gaps,
                "recovered_frames": self.recovered_frames,
                "frames_written": self.frames,
                "frames_in_skipped_files": self.frames_skipped,
            },
            "files": [o.as_dict() for o in self.outcomes],
            "log_file": str(self.log_file) if self.log_file else None,
            "state_file": str(self.state_path) if self.state_path else None,
        }


# ----------------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------------
def setup_batch_logging(output_dir: Path, *, verbose: bool = False,
                        log_file: Path | None = None) -> Path:
    """One log file per run, plus stderr. Returns the log path."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_file = log_dir / f"batch_{stamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("flir_csq")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return log_file


# ----------------------------------------------------------------------------------
# output paths
# ----------------------------------------------------------------------------------
def _plan_targets(files: list[Path], options: BatchOptions) -> dict[Path, Path]:
    """Output directory for each recording: mirrors the tree, or flattens it.

    Flattening is only safe when file names are unique; a duplicate stem goes into
    a sub-directory named after its parent, and the log says so.
    """
    plan: dict[Path, Path] = {}
    used: dict[str, Path] = {}
    for path in files:
        if options.preserve_tree:
            directory = options.output_dir / path.relative_to(options.input_dir).parent
        else:
            directory = options.output_dir
            name = path.stem.lower()
            previous = used.get(name)
            if previous is not None and previous != path:
                slug = path.parent.name or "root"
                directory = options.output_dir / slug
                LOG.warning("duplicate file name %r for %s and %s: writing %s into %s/",
                            path.stem, previous, path, path.name, slug)
            used[name] = path
        plan[path] = directory
    return plan


_STEM_EXTENSIONS = {"h5": ".h5", "npy": ".npy", "mat": ".mat", "csv": ""}


def _outputs_of(path: Path, directory: Path, options: BatchOptions) -> dict[str, Path]:
    """Planned output paths. ``suffix`` keeps ``_temperature``, ``stem`` drops it."""
    outputs = {fmt: default_output_path(path, directory, fmt) for fmt in options.formats}
    if options.naming == "stem":
        outputs = {fmt: directory / f"{path.stem}{_STEM_EXTENSIONS[fmt]}"
                   for fmt in options.formats}
    return outputs


def _record_key(path: Path, options: BatchOptions) -> str:
    try:
        return str(path.relative_to(options.input_dir))
    except ValueError:
        return str(path)


def _record_key(path: Path, options: BatchOptions) -> str:
    try:
        return str(path.relative_to(options.input_dir))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------------
# the job
# ----------------------------------------------------------------------------------
class _Progress:
    """Throttled per-frame progress into the log (no flooding at 30 Hz)."""

    def __init__(self, logger: logging.Logger, label: str,
                 callback: Callable[[float, str], None] | None) -> None:
        self.logger = logger
        self.label = label
        self.callback = callback
        self.last_time = 0.0
        self.last_pct = -1

    def __call__(self, fraction: float, message: str, done: int, total: int) -> None:
        now = time.monotonic()
        pct = int(fraction * 100)
        if done and (now - self.last_time < 5.0) and pct == self.last_pct and done != total:
            return
        self.last_time = now
        self.last_pct = pct
        if done:
            self.logger.info("%s: frame %d/%d (%d%%)", self.label, done, total, pct)
        else:
            self.logger.info("%s: %s", self.label, message)
        if self.callback:
            self.callback(fraction, f"{self.label}: {message}")


def run_batch(options: BatchOptions) -> BatchResult:
    """Convert every recording under ``options.input_dir`` into ``options.output_dir``."""
    started = time.perf_counter()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_batch_logging(options.output_dir, log_file=options.log_file)
    logger = LOG
    state = load_state(options.output_dir)
    state["converter_version"] = CONVERTER_VERSION
    state.setdefault("files", {})

    result = BatchResult(
        input_dir=options.input_dir,
        output_dir=options.output_dir,
        started_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        log_file=log_file,
        state_path=options.output_dir / STATE_FILENAME,
        dry_run=options.dry_run,
    )
    logger.info("flir-csq %s batch: %s -> %s", CONVERTER_VERSION,
                options.input_dir, options.output_dir)
    logger.info("formats=%s recursive=%s skip_existing=%s dry_run=%s",
                ",".join(options.formats), options.recursive, options.skip_existing,
                options.dry_run)

    try:
        files = discover_recordings(options.input_dir, recursive=options.recursive,
                                    extensions=options.extensions, pattern=options.pattern)
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        result.finished_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - started
        _write_summary(result, options)
        return result

    if options.limit is not None:
        files = files[: options.limit]
    if not files:
        logger.warning("no .csq/.seq recordings found under %s", options.input_dir)
        result.finished_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - started
        _write_summary(result, options)
        return result
    logger.info("%d recording(s) found", len(files))

    key = options.conversion_key()
    targets = _plan_targets(files, options)
    try:
        for index, path in enumerate(files, start=1):
            label = f"[{index}/{len(files)}] {_record_key(path, options)}"
            outputs = _outputs_of(path, targets[path], options)
            entry = state["files"].get(_record_key(path, options), {})
            current = signature(path)

            if _can_skip(entry, current, key, outputs, options):
                logger.info("%s: already converted, skipping (%s)", label, outputs["h5"]
                            if "h5" in outputs else next(iter(outputs.values())))
                result.outcomes.append(FileOutcome(
                    input=path, status="skipped", output=outputs.get("h5"),
                    frames=int(entry.get("frames", 0))))
                continue

            if options.dry_run:
                logger.info("%s: would convert -> %s", label,
                            ", ".join(str(p) for p in outputs.values()))
                result.outcomes.append(FileOutcome(
                    input=path, status="planned", output=outputs.get("h5")))
                continue

            logger.info("%s: converting", label)
            file_started = time.perf_counter()
            try:
                conversion = _convert_one(path, outputs, options, label)
            except ConversionCancelled as exc:
                logger.warning("%s: %s", label, exc)
                result.outcomes.append(FileOutcome(input=path, status="failed",
                                                   error=str(exc)))
                _record(state, options, path, current, key, "failed", error=str(exc),
                        outputs=outputs)
                save_state(options.output_dir, state)
                break
            except CsqError as exc:
                logger.error("%s: %s", label, exc.message)
                if exc.hint:
                    logger.error("%s: hint: %s", label, exc.hint)
                result.outcomes.append(FileOutcome(
                    input=path, status="failed", error=exc.message, hint=exc.hint))
                _record(state, options, path, current, key, "failed", error=exc.message,
                        outputs=outputs)
                save_state(options.output_dir, state)
                if not options.continue_on_error:
                    logger.error("stopping after the first failure (--on-error stop)")
                    break
                continue
            except Exception as exc:  # noqa: BLE001 - one file must not kill the batch
                logger.exception("%s: unexpected error: %s", label, exc)
                result.outcomes.append(FileOutcome(
                    input=path, status="failed",
                    error=f"{type(exc).__name__}: {exc}"))
                _record(state, options, path, current, key, "failed",
                        error=f"{type(exc).__name__}: {exc}", outputs=outputs)
                save_state(options.output_dir, state)
                if not options.continue_on_error:
                    break
                continue

            duration = time.perf_counter() - file_started
            primary = conversion.outputs.get("h5") or next(iter(conversion.outputs.values()))
            logger.info("%s: done, %d frames -> %s (%.1f s, %.1f frames/s)", label,
                        conversion.frames_written, primary, duration,
                        conversion.frames_written / duration if duration > 0 else 0.0)
            if not conversion.validation.get("passed", True):
                logger.warning("%s: output verification failed: %s", label,
                               "; ".join(conversion.validation.get("problems", [])))
            for warning in conversion.warnings[:5]:
                logger.info("%s: warning: %s", label, warning)
            recovered = int(conversion.metadata.get("conversion", {})
                            .get("recovered_frames", 0) or 0)
            outcome_status = "ok-with-gaps" if recovered else "ok"
            if recovered:
                logger.warning("%s: %d frame(s) could not be decoded and were written as NaN "
                               "(timestamps preserved); see failed_frames in the metadata",
                               label, recovered)
            result.outcomes.append(FileOutcome(
                input=path, status=outcome_status, output=primary,
                frames=conversion.frames_written, recovered_frames=recovered,
                duration_s=duration, warnings=conversion.warnings,
                error=(f"{recovered} unreadable frame(s) kept as NaN" if recovered else None)))
            _record(state, options, path, current, key, "ok", outputs=conversion.outputs,
                    frames=conversion.frames_written)
            save_state(options.output_dir, state)

            if options.progress:
                options.progress(index / len(files), label)
    except KeyboardInterrupt:
        logger.warning("interrupted: %d file(s) finished, state saved; re-run to continue",
                       result.converted)

    result.finished_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    result.duration_s = time.perf_counter() - started
    save_state(options.output_dir, state)
    _write_summary(result, options)
    logger.info("finished: %d converted, %d skipped, %d failed, %d frames in %.1f s",
                result.converted, result.skipped, result.failed, result.frames,
                result.duration_s)
    if result.log_file:
        logger.info("log: %s", result.log_file)
    for outcome in result.outcomes:
        if outcome.status == "failed":
            logger.error("failed: %s — %s", outcome.input, outcome.error)
    return result


def _convert_one(path: Path, outputs: dict[str, Path], options: BatchOptions,
                 label: str) -> ConversionResult:
    target_dir = next(iter(outputs.values())).parent
    conversion = convert(ConversionOptions(
        input_path=path,
        output_dir=target_dir,
        formats=options.formats,
        save_raw=options.save_raw,
        save_mask=options.save_mask,
        save_metadata=options.save_metadata,
        mode=options.mode,
        stride=options.stride,
        overwrite=True,  # the batch decides, and records why in the state file
        compression_level=options.compression_level,
        mat_version=options.mat_version,
        on_decode_error=options.on_decode_error,
        dump_bad_frames=options.dump_bad_frames,
        allow_fallback_decoder=options.allow_fallback_decoder,
        emissivity=options.emissivity,
        object_distance_m=options.object_distance_m,
        reflected_temperature_c=options.reflected_temperature_c,
        atmospheric_temperature_c=options.atmospheric_temperature_c,
        relative_humidity=options.relative_humidity,
        window_transmission=options.window_transmission,
        window_temperature_c=options.window_temperature_c,
        progress=_Progress(LOG, label, options.progress),
    ))
    _rename_outputs(conversion, outputs)
    return conversion


def _rename_outputs(conversion: ConversionResult, planned: dict[str, Path]) -> None:
    """Apply ``--naming stem`` after the fact (directories are already unique)."""
    import shutil

    for fmt, target in planned.items():
        actual = conversion.outputs.get(fmt)
        if actual is None or Path(actual).resolve() == target.resolve():
            continue
        source = Path(actual)
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(source), str(target))
        conversion.outputs[fmt] = target


def _can_skip(entry: dict[str, Any], current: dict[str, Any], key: str,
              outputs: dict[str, Path], options: BatchOptions) -> bool:
    if not options.skip_existing or options.overwrite:
        return False
    if entry.get("status") != "ok" or entry.get("conversion_key") != key:
        return False
    if entry.get("signature") != current:
        return False
    primary = outputs.get("h5") or next(iter(outputs.values()), None)
    if primary is None or not primary.exists():
        return False
    recorded_size = entry.get("output_size")
    if recorded_size is not None and primary.is_file():
        if primary.stat().st_size != recorded_size:
            return False
    return True


def _record(state: dict[str, Any], options: BatchOptions, path: Path,
            current: dict[str, Any], key: str, status: str, *,
            outputs: dict[str, Path] | None = None,
            frames: int = 0, error: str | None = None) -> None:
    entry = {
        "input": str(path.resolve()),
        "signature": current,
        "conversion_key": key,
        "status": status,
        "frames": frames,
        "error": error,
        "finished_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "converter_version": CONVERTER_VERSION,
    }
    outputs = outputs or {}
    entry["outputs"] = {fmt: str(p) for fmt, p in outputs.items() if fmt != "json"}
    primary = outputs.get("h5") or (next(iter(outputs.values())) if outputs else None)
    if primary is not None and Path(primary).is_file():
        entry["output_size"] = Path(primary).stat().st_size
        entry["output"] = str(primary)
    state["files"][_record_key(path, options)] = entry


def _write_summary(result: BatchResult, options: BatchOptions) -> None:
    path = options.summary_path or (options.output_dir / SUMMARY_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    result.summary_path = path


def iter_failed(result: BatchResult) -> Iterator[FileOutcome]:
    return (o for o in result.outcomes if o.status == "failed")
