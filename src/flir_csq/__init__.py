"""flir-csq: FLIR T-series ``.csq`` radiometric recordings -> temperature matrices in degC.

Public API::

    from flir_csq import inspect_recording, convert, ConversionOptions

    info = inspect_recording("FLIR0001.csq")
    result = convert(ConversionOptions(input_path="FLIR0001.csq",
                                       output_dir="results",
                                       formats=("h5", "csv")))
    print(result.outputs["h5"], result.validation["passed"])
"""

from __future__ import annotations

from .container import CsqReader, scan_csq
from .converter import ConversionOptions, ConversionResult, convert, inspect_recording
from .errors import (
    ConversionCancelled,
    CsqError,
    MalformedFrame,
    MissingCalibration,
    NotACsqFile,
    NotRadiometric,
    OutputError,
    UnsupportedPayload,
)
from .exporter import CONVERTER_VERSION

__version__ = CONVERTER_VERSION
__all__ = [
    "CONVERTER_VERSION",
    "ConversionCancelled",
    "ConversionOptions",
    "ConversionResult",
    "CsqError",
    "CsqReader",
    "MalformedFrame",
    "MissingCalibration",
    "NotACsqFile",
    "NotRadiometric",
    "OutputError",
    "UnsupportedPayload",
    "__version__",
    "convert",
    "inspect_recording",
    "scan_csq",
]
