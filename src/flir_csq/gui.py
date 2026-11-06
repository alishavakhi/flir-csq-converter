"""PySide6 desktop interface.

Layout
------
left   : file drop zone + selected file, recording information
right  : output configuration (directory, formats, raw/metadata, mode)
bottom : progress bar, per-frame status, log, convert/cancel, open-folder

Conversion runs in a :class:`QThread`; the widget thread only ever receives
signals, so the window keeps repainting (and the Cancel button keeps working)
for the whole recording. Cancel is a :class:`threading.Event` checked before
every frame, and the partial HDF5 file is deleted when it fires.
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .converter import ConversionOptions, convert, inspect_recording
from .errors import ConversionCancelled, CsqError
from .exporter import CONVERTER_VERSION


class Worker(QObject):
    """Runs inspection or conversion off the GUI thread."""

    progress = Signal(float, str, int, int)
    inspected = Signal(dict)
    converted = Signal(object)
    failed = Signal(str, str)

    def __init__(self, kind: str, path: Path, options: ConversionOptions | None = None,
                 cancel: threading.Event | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.path = path
        self.options = options
        self.cancel = cancel or threading.Event()

    def run(self) -> None:
        try:
            if self.kind == "inspect":
                self.inspected.emit(inspect_recording(self.path))
                return
            assert self.options is not None
            self.options.progress = lambda f, m, d, t: self.progress.emit(f, m, d, t)
            self.options.should_cancel = self.cancel.is_set
            self.converted.emit(convert(self.options))
        except ConversionCancelled as exc:
            self.failed.emit("Conversion cancelled", str(exc))
        except CsqError as exc:
            hint = f"\n\n{exc.hint}" if exc.hint else ""
            self.failed.emit(exc.message, hint)
        except Exception as exc:  # noqa: BLE001 - never let a worker die silently
            self.failed.emit(f"Unexpected error: {type(exc).__name__}",
                             f"{exc}\n\n{traceback.format_exc(limit=3)}")


class DropZone(QFrame):
    """Dashed drop target that also acts as a button."""

    fileDropped = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(104)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop a .csq recording here\nor click to choose a file")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("dropLabel")
        layout.addWidget(self.label)

    # -- drag & drop ---------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return

    def mousePressEvent(self, event) -> None:
        self.window().choose_file()  # type: ignore[attr-defined]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"FLIR CSQ → Temperature  ({CONVERTER_VERSION})")
        self.resize(980, 720)
        self.recording_path: Path | None = None
        self.info: dict[str, Any] = {}
        self.worker_thread: QThread | None = None
        self.worker: Worker | None = None
        self.cancel_event = threading.Event()
        self.last_output_dir: Path | None = None
        self.silent = False  # --self-test: no modal dialogs
        self._build()
        self._busy(False)
        self._apply_style()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("FLIR .csq → temperature matrices")
        title.setObjectName("title")
        subtitle = QLabel("Radiometric recordings of a FLIR T-series camera are decoded frame by "
                          "frame; every pixel becomes a temperature in °C, written to HDF5.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        root.addLayout(columns, 1)

        # -- left column
        left = QVBoxLayout()
        left.setSpacing(10)
        self.drop = DropZone()
        self.drop.fileDropped.connect(self.select_file)
        left.addWidget(self.drop)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        left.addWidget(self.file_label)

        info_box = QGroupBox("Recording information")
        info_layout = QFormLayout(info_box)
        info_layout.setLabelAlignment(Qt.AlignRight)
        info_layout.setVerticalSpacing(6)
        self.info_fields = {
            "camera": QLabel("—"), "resolution": QLabel("—"), "frames": QLabel("—"),
            "duration": QLabel("—"), "payload": QLabel("—"), "timestamps": QLabel("—"),
            "radiometry": QLabel("—"), "planck": QLabel("—"),
        }
        rows = [("Camera", "camera"), ("Thermal resolution", "resolution"),
                ("Frames", "frames"), ("Duration", "duration"),
                ("Frame payload", "payload"), ("Timestamps", "timestamps"),
                ("Radiometry", "radiometry"), ("Planck constants", "planck")]
        for label, key in rows:
            value = self.info_fields[key]
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True)
            info_layout.addRow(f"{label}:", value)
        left.addWidget(info_box)
        left.addStretch(1)
        columns.addLayout(left, 3)

        # -- right column
        right = QVBoxLayout()
        right.setSpacing(10)
        out_box = QGroupBox("Output")
        out_layout = QGridLayout(out_box)
        out_layout.setVerticalSpacing(8)

        out_layout.addWidget(QLabel("Directory:"), 0, 0)
        self.out_dir = QLineEdit(str(Path.home() / "csq_results"))
        out_layout.addWidget(self.out_dir, 0, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_output_dir)
        out_layout.addWidget(browse, 0, 2)

        out_layout.addWidget(QLabel("Formats:"), 1, 0, Qt.AlignTop)
        formats = QHBoxLayout()
        self.fmt_h5 = QCheckBox("HDF5 (default)")
        self.fmt_h5.setChecked(True)
        self.fmt_npy = QCheckBox("NumPy")
        self.fmt_csv = QCheckBox("CSV per frame")
        self.fmt_mat = QCheckBox("MATLAB")
        for box in (self.fmt_h5, self.fmt_npy, self.fmt_csv, self.fmt_mat):
            formats.addWidget(box)
        formats.addStretch(1)
        out_layout.addLayout(formats, 1, 1, 1, 2)

        self.save_raw = QCheckBox("Store raw radiometric counts (uint16)")
        self.save_raw.setChecked(True)
        self.save_mask = QCheckBox("Store per-pixel valid mask")
        self.save_metadata = QCheckBox("Store metadata and conversion report (JSON)")
        self.save_metadata.setChecked(True)
        self.overwrite = QCheckBox("Overwrite existing outputs")
        out_layout.addWidget(self.save_raw, 2, 1, 1, 2)
        out_layout.addWidget(self.save_mask, 3, 1, 1, 2)
        out_layout.addWidget(self.save_metadata, 4, 1, 1, 2)
        out_layout.addWidget(self.overwrite, 5, 1, 1, 2)

        out_layout.addWidget(QLabel("Conversion:"), 6, 0)
        self.mode = QComboBox()
        self.mode.addItem("Auto — correct for emissivity/atmosphere when available", "auto")
        self.mode.addItem("Planck apparent temperature only", "apparent")
        self.mode.addItem("Force object-temperature correction", "object")
        out_layout.addWidget(self.mode, 6, 1, 1, 2)

        out_layout.addWidget(QLabel("Emissivity override:"), 7, 0)
        self.emissivity = QDoubleSpinBox()
        self.emissivity.setRange(0.0, 1.0)
        self.emissivity.setDecimals(3)
        self.emissivity.setSingleStep(0.01)
        self.emissivity.setSpecialValueText("from file")
        self.emissivity.setValue(0.0)
        out_layout.addWidget(self.emissivity, 7, 1)
        right.addWidget(out_box)

        # -- progress
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        progress_layout.addWidget(self.progress)
        self.status = QLabel("Ready.")
        self.status.setObjectName("status")
        progress_layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFont(QFont("monospace", 9))
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        progress_layout.addWidget(self.log)
        right.addWidget(progress_box, 1)

        buttons = QHBoxLayout()
        self.choose_button = QPushButton("Select .csq file…")
        self.choose_button.clicked.connect(self.choose_file)
        self.convert_button = QPushButton("Start conversion")
        self.convert_button.setObjectName("primary")
        self.convert_button.clicked.connect(self.start_conversion)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_conversion)
        self.open_button = QPushButton("Open output folder")
        self.open_button.clicked.connect(self.open_output_folder)
        buttons.addWidget(self.choose_button)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.convert_button)
        right.addLayout(buttons)
        columns.addLayout(right, 4)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f6f7f9; }
            QLabel#title { font-size: 22px; font-weight: 600; color: #12263a; }
            QLabel#subtitle { color: #46586b; }
            QLabel#fileLabel { color: #12263a; font-weight: 500; }
            QLabel#status { color: #33475b; }
            QGroupBox { border: 1px solid #d5dbe3; border-radius: 8px; margin-top: 10px;
                        background: #ffffff; padding: 10px; font-weight: 500; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                               color: #33475b; }
            QFrame#dropZone { border: 2px dashed #9aaec2; border-radius: 10px;
                              background: #eef3f8; }
            QFrame#dropZone:hover { background: #e3edf7; border-color: #4a7fb5; }
            QLabel#dropLabel { color: #3b5b7a; font-size: 13px; }
            QPushButton { padding: 7px 14px; border-radius: 6px; border: 1px solid #c3cdd8;
                          background: #ffffff; }
            QPushButton:hover { background: #f0f4f8; }
            QPushButton:disabled { color: #9aa4ae; background: #f3f5f7; }
            QPushButton#primary { background: #1f6feb; color: white; border: none;
                                  font-weight: 600; }
            QPushButton#primary:hover { background: #1a5fd0; }
            QPushButton#primary:disabled { background: #b9c8dd; color: #eef2f7; }
            QLineEdit, QComboBox, QDoubleSpinBox { padding: 5px; border-radius: 5px;
                                                   border: 1px solid #c3cdd8; background: white; }
            QPlainTextEdit { border: 1px solid #d5dbe3; border-radius: 6px; background: #fbfcfd; }
            QProgressBar { border: 1px solid #d5dbe3; border-radius: 6px; height: 18px;
                           text-align: center; background: #ffffff; }
            QProgressBar::chunk { background: #1f6feb; border-radius: 5px; }
        """)

    # ------------------------------------------------------------------ state
    def _busy(self, busy: bool) -> None:
        self.convert_button.setEnabled(bool(self.recording_path) and not busy)
        self.choose_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.open_button.setEnabled(self.last_output_dir is not None and not busy)
        for box in (self.fmt_h5, self.fmt_npy, self.fmt_csv, self.fmt_mat, self.save_raw,
                    self.save_mask, self.save_metadata, self.overwrite):
            box.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.emissivity.setEnabled(not busy)

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    # ------------------------------------------------------------------ actions
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a FLIR .csq recording", str(Path.home()),
            "FLIR radiometric sequence (*.csq *.seq);;All files (*)")
        if path:
            self.select_file(Path(path))

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    def select_file(self, path: Path) -> None:
        if not path.is_file():
            return
        self.recording_path = path
        size_mb = path.stat().st_size / (1024 * 1024)
        self.file_label.setText(f"{path.name}  ({size_mb:.1f} MB)\n{path.parent}")
        self.drop.label.setText(f"{path.name}\nclick to choose another file")
        self.log.clear()
        self._log(f"selected {path}")
        self.progress.setValue(0)
        self.status.setText("Reading recording…")
        for label in self.info_fields.values():
            label.setText("…")
        self._start_worker("inspect", path, None)

    def _collect_formats(self) -> tuple[str, ...]:
        formats: list[str] = []
        if self.fmt_h5.isChecked():
            formats.append("h5")
        if self.fmt_npy.isChecked():
            formats.append("npy")
        if self.fmt_csv.isChecked():
            formats.append("csv")
        if self.fmt_mat.isChecked():
            formats.append("mat")
        return tuple(formats or ["h5"])

    def start_conversion(self) -> None:
        if self.recording_path is None:
            return
        formats = self._collect_formats()
        out_dir = Path(self.out_dir.text()).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot create output directory", str(exc))
            return
        emissivity = self.emissivity.value() or None
        options = ConversionOptions(
            input_path=self.recording_path,
            output_dir=out_dir,
            formats=formats,
            save_raw=self.save_raw.isChecked(),
            save_mask=self.save_mask.isChecked(),
            save_metadata=self.save_metadata.isChecked(),
            overwrite=self.overwrite.isChecked(),
            mode=self.mode.currentData(),
            emissivity=emissivity,
        )
        self.progress.setValue(0)
        self.status.setText("Converting…")
        self._log(f"converting to {out_dir} — formats: {', '.join(formats)}")
        self._start_worker("convert", self.recording_path, options)

    def cancel_conversion(self) -> None:
        self.cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling after the current frame…")
        self._log("cancel requested")

    def open_output_folder(self) -> None:
        if self.last_output_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_dir)))

    # ------------------------------------------------------------------ worker
    def _start_worker(self, kind: str, path: Path, options: ConversionOptions | None) -> None:
        if self.worker_thread is not None:
            return
        self.cancel_event = threading.Event()
        thread = QThread(self)
        worker = Worker(kind, path, options, self.cancel_event)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.inspected.connect(self._on_inspected)
        worker.converted.connect(self._on_converted)
        worker.failed.connect(self._on_failed)
        worker.progress.connect(self._on_progress)
        for signal in (worker.inspected, worker.converted):
            signal.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        self.worker_thread = thread
        self.worker = worker
        self._busy(True)
        thread.start()

    def _on_thread_finished(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None
        self._busy(False)

    def _on_progress(self, fraction: float, message: str, done: int, total: int) -> None:
        self.progress.setValue(round(fraction * 100))
        if total:
            self.status.setText(f"{message} — {done}/{total} frames")
        else:
            self.status.setText(message)

    def _on_inspected(self, info: dict) -> None:
        self.info = info
        cam = info.get("camera", {})
        dims = info.get("dimensions", {})
        timing = info.get("timing", {})
        frames = info.get("frames", {})
        model = cam.get("camera_model") or "unknown camera"
        if cam.get("serial_number"):
            model += f"  (s/n {cam['serial_number']})"
        self.info_fields["camera"].setText(model)
        self.info_fields["resolution"].setText(
            f"{dims.get('width')} × {dims.get('height')} px"
            f"  (T540 native is 464 × 348)")
        self.info_fields["frames"].setText(
            f"{frames.get('complete', 0)} complete"
            + (f", {frames['truncated']} incomplete" if frames.get("truncated") else ""))
        duration = timing.get("duration_s")
        rate = timing.get("measured_frame_rate_hz")
        self.info_fields["duration"].setText(
            f"{duration:.2f} s at {rate:.3f} Hz measured" if duration and rate else "unknown")
        self.info_fields["payload"].setText(str(info.get("decode", {}).get("codec")))
        self.info_fields["timestamps"].setText(
            "camera clock (ms resolution)" if not timing.get("timestamps_estimated")
            else "estimated from frame rate (not in file)")
        if info.get("radiometric"):
            self.info_fields["radiometry"].setText(
                f"calibrated · ε {cam.get('emissivity')} · {cam.get('object_distance_m')} m · "
                f"RH {cam.get('relative_humidity')}")
            self.info_fields["planck"].setText(
                f"R1 {cam.get('planck_r1'):.3f} · B {cam.get('planck_b'):.3f} · "
                f"F {cam.get('planck_f')} · O {cam.get('planck_o')} · "
                f"R2 {cam.get('planck_r2'):.6g}")
        else:
            missing = ", ".join(info.get("missing_for_temperature", []))
            self.info_fields["radiometry"].setText(f"NO CALIBRATION (missing {missing})")
            self.info_fields["planck"].setText("—")
        self.status.setText(f"Ready — {frames.get('complete', 0)} frames can be converted.")
        for warning in info.get("warnings", [])[:6]:
            self._log(f"warning: {warning}")
        self._log(f"decoder: {info.get('decode', {}).get('codec')} · "
                  f"dimensions {dims.get('width')}×{dims.get('height')}")

    def _on_converted(self, result) -> None:
        self.progress.setValue(100)
        primary = result.primary_output
        if primary is not None:
            self.last_output_dir = Path(self.out_dir.text()).expanduser()
            if primary.is_dir():
                self.last_output_dir = primary.parent
        agg = result.metadata.get("statistics", {}).get("aggregate", {})
        span = (f"{agg['global_min_C']:.2f} … {agg['global_max_C']:.2f} °C"
                if agg.get("global_min_C") is not None else "raw counts only")
        ok = result.validation.get("passed") if result.validation else None
        self.status.setText(
            f"Done — {result.frames_written} frames, {span}"
            + ("  · output verified" if ok else "  · verification FAILED" if ok is False else ""))
        self._log(f"converted {result.frames_written} frames in {result.duration_s:.1f} s")
        for fmt, path in result.outputs.items():
            self._log(f"  {fmt}: {path}")
        for warning in result.warnings[:10]:
            self._log(f"warning: {warning}")
        self.open_button.setEnabled(True)
        details = "\n".join(str(p) for p in result.outputs.values())
        if self.silent:
            return
        QMessageBox.information(
            self, "Conversion finished",
            f"{result.frames_written} frames converted ({span}).\n\n{details}"
            + ("\n\nOutput verified: temperature reproduced from the stored raw counts."
               if ok else ""))

    def _on_failed(self, message: str, detail: str) -> None:
        self.progress.setValue(0)
        self.status.setText(f"Failed: {message}")
        self._log(f"error: {message}")
        if detail:
            self._log(detail.strip())
        if not self.silent:
            QMessageBox.warning(self, "Conversion failed", f"{message}\n{detail}".strip())


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    self_test = "--self-test" in args
    if self_test:
        import os

        args = [a for a in args if a != "--self-test"]
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(["flir-csq-gui", *args])
    window = MainWindow()
    window.silent = self_test
    window.show()
    if self_test:
        app.processEvents()
        # exercise the worker path end to end on a synthetic recording
        from tempfile import TemporaryDirectory

        from .synthetic import make_synthetic_csq

        with TemporaryDirectory() as tmp:
            demo = Path(tmp) / "selftest.csq"
            make_synthetic_csq(demo, n_frames=3, height=16, width=12)
            window.out_dir.setText(str(Path(tmp) / "out"))
            window.select_file(demo)
            for _ in range(2000):
                app.processEvents()
                if window.worker_thread is None and window.info:
                    break
                QThread.msleep(5)
            assert window.info, "inspection did not finish"
            window.fmt_h5.setChecked(True)
            window.start_conversion()
            for _ in range(6000):
                app.processEvents()
                if window.worker_thread is None and window.last_output_dir is not None:
                    break
                QThread.msleep(5)
            app.processEvents()
            ok = window.last_output_dir is not None
            print(f"self-test: window constructed, inspected {window.info.get('frames')}, "
                  f"conversion {'ok' if ok else 'FAILED'}")
            return 0 if ok else 1
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
