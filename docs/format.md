# The `.csq` format: what a recording is, and what this implementation assumes

Status: container, record layout, payload codec, calibration fields and timestamps **verified on a
real T-series recording** (FLIR T1020, 22 frames) against ExifTool's independent extraction. See
`validation.md` for the numbers. A T540 recording has not been seen yet; `validation.md` lists what
to check.

## 1. It is not a video file

FLIR's T5xx manual describes radiometric recording as follows: `*.csq` "supports full radiometry
but is only supported by FLIR Systems software. The file does not include any visual image
information." Nothing in a `.csq` is a colour frame: there is no palette, no YUV stream, no
FFmpeg-demuxable video. It is a **sequence of complete FLIR FFF blocks**, one per recorded frame,
concatenated without a container header or index.

Consequently:

* decoding it as video (FFmpeg, OpenCV) yields nothing usable;
* the only correct unit of work is the FFF block, and the only correct framing is the block's own
  record directory;
* any "temperature" derived from an 8-bit colour image is fabricated. This tool refuses such
  input (`NotRadiometric`) instead of inventing a palette-to-°C mapping.

## 2. Sources this implementation was written from

| ref | what | where |
|---|---|---|
| **[ET]** | ExifTool `Image::ExifTool::FLIR` (`%FFF`, `%RawData`, `%CameraInfo`) — tag names, offsets, byte-order rule, Planck tags, timestamp layout | <https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/FLIR.pm>, ExifTool 12.76 |
| **[FP]** | flirpy `io/fff.py`, `io/seq.py` — independent reader of the same structures, framing "record offset + length" | <https://github.com/LJMUAstroecology/flirpy> (MIT) |
| **[FR]** | flirpy `util/raw.py` — `raw2temp`, the ThermImage port of the Minkina & Dudzik correction | same repository |
| **[AP]** | AlexanderProd/csq — a Python CSQ reader (no licence file in the repository) | <https://github.com/AlexanderProd/csq> |
| **[LS]** | JPEG-LS, ITU-T T.87 (SOF55 = `FF F7`) | <https://jpeg.org/jpegls/> |

What was actually read from each is described below. Nothing was copied from a project without a
licence, and no vendor SDK was used.

### On the existing open-source CSQ decoder

`AlexanderProd/csq` was examined as the task requested, and was **not reused**:

* the repository has **no licence file** (GitHub reports no licence), so adapting its code would
  not be compliant. This project was implemented from the public format descriptions instead;
* it splits frames by searching for the four-byte pattern `FFF\0RT` (creator `RTP`), which fails
  for recordings written by other creators (for example `CAMCTRL`), and it can cut a frame where
  the pattern occurs inside pixel data. This implementation computes frame extents from the record
  directory;
* it shells out to `exiftool` **once per frame** through a temporary file, which dominates runtime
  on long recordings. Metadata here is parsed in-process, a few kilobytes per frame;
* it reads the recording's metadata from the first frame only, so per-frame parameters and
  timestamps are lost;
* it does return °C (`raw2temp`) rather than counts, but with the first frame's parameters and no
  per-frame validation.

It remains useful as a reference and its sample recording (`hog.csq`) was used to validate this
implementation — see `validation.md`. Note that the same JPEG-LS payload decoded with
`pylibjpeg-libjpeg` (the package it uses) is bit-identical to `imagecodecs`/CharLS here, so its
pixel path does work; the container and metadata handling are where it is limited.

### FLIR Atlas SDK / Research Studio / ResearchIR

These are proprietary, Windows-oriented and licence-gated. FLIR's Science File SDK (Atlas) can read
SEQ/CSQ and expose temperature data, but it cannot be redistributed, cannot be installed on Linux
without Wine, and its licence is not compatible with an MIT repository. This project therefore
implements the public format instead. If an organisation already holds a licence, an Atlas backend
could be added *behind an optional interface* without changing the outputs — the exporter and the
HDF5 schema are independent of the decoder. No proprietary library is bundled, downloaded or
called.

## 3. FFF block layout

All multi-byte fields are little-endian unless the format version says otherwise; the version at
`0x14` is a small integer (100/101), which is how byte order is detected (same rule as [ET] and
[FP]).

```
offset  size  field
0x00    4     magic "FFF\0"  (AFF\0 accepted)
0x04    16    creator, NUL padded ("CAMCTRL", "RTP", ...)
0x14    4     format version (100, 101 seen)
0x18    4     offset of the record directory
0x1c    4     number of directory entries
0x20    4     next free index id
0x24    2     swap pattern
0x26    14    spares
0x34    8     reserved
0x3c    4     checksum
```

Record-directory entry (32 bytes): `uint16 type, uint16 subtype, uint32 version, uint32 index_id,
uint32 offset, uint32 length, uint32 parent, uint32 object, uint32 checksum`. Offsets are relative
to the start of the block.

A block's extent is `max(directory end, max(record.offset + record.length))` — exactly the
definition used by [FP]. Scanning for the next magic instead is unsafe, because compressed pixel
data can contain the bytes `46 46 46 00`.

### Record types

| type | name | used here |
|---|---|---|
| `0x01` | RawData | **yes** — the measurement counts |
| `0x20` | CameraInfo | **yes** — calibration, parameters, identity, timestamp |
| `0x21` | MeasurementInfo | listed only (spot/area tools) |
| `0x22` | PaletteInfo | listed only (display palette) |
| `0x24` | EmbeddedAudioFile | listed only |
| `0x2a` | PiP | listed only |
| `0x2b` | GPSInfo | listed only |
| `0x2e` | ParameterInfo | listed only |

Unknown types are counted and shown by `--inspect` instead of being ignored silently, so firmware
that changes the container is visible rather than mis-decoded.

## 4. RawData record (`0x01`)

```
0x00  uint16  byte-order sentinel: 0x0002 when the record is little-endian
0x02  uint16  width
0x04  uint16  height
0x06  uint16  width - 1        (informational)
0x08  uint16  height - 1       (informational)
0x20  ...     payload
```

Payload variants actually handled:

| payload starts with | meaning | decoder |
|---|---|---|
| any | uncompressed 16-bit counts (subtype 1 big-endian / 2 little-endian) | numpy |
| `89 50 4E 47` | 16-bit PNG (byte order swapped on many cameras — detected from an implausible full-scale span, as [ET] documents) | Pillow |
| `49 49 2A 00` / `4D 4D 00 2A` | TIFF wrapper | Pillow |
| `FF D8 FF F7` | **JPEG-LS** (SOF55) — what every T-series CSQ examined stores | imagecodecs/CharLS, or pylibjpeg-libjpeg |
| `FF D8 FF C3` | lossless JPEG (SOF3) | imagecodecs |

The JPEG-LS payload is a 16-bit **lossless** compression of the sensor counts, so the decoded
array is bit-identical to what the sensor measured — verified here by decoding the same payload
with two independent codecs and by comparing to ExifTool's own extraction (sha256 of the payload
matches; both decoders return identical arrays).

If no decoder is installed the frame is refused with the install command. There is **no** fallback
to a colour preview: that would silently replace a measurement with a picture.

## 5. CameraInfo record (`0x20`)

Endianness again follows a `0x0002` sentinel at offset 0. Offsets implemented (from [ET]):

```
0x02 w  uint16   image width            0x58  float  Planck R1
0x04 h  uint16   image height           0x5C  float  Planck B
0x20    float    Emissivity             0x60  float  Planck F
0x24    float    Object distance [m]    0x70  float  AtmosphericTransAlpha1
0x28    float K  Reflected apparent T   0x74  float  AtmosphericTransAlpha2
0x2C    float K  Atmospheric T          0x78  float  AtmosphericTransBeta1
0x30    float K  IR window T            0x7C  float  AtmosphericTransBeta2
0x34    float    IR window transmission 0x80  float  AtmosphericTransX
0x3C    float    Relative humidity      0x90  float K Camera range max
0xD4    char[32] Camera model           0x94  float K Camera range min
0xF4    char[16] Camera part number     0x308 int32  Planck O
0x104   char[16] Camera serial number   0x30C float  Planck R2
0x114   char[16] Camera software        0x310 uint16 RawValueRangeMin
0x170   char[32] Lens model             0x312 uint16 RawValueRangeMax
0x1B4   float    Field of view [deg]    0x338 uint16 RawValueMedian
0x384   10 bytes DateTimeOriginal       0x33C uint16 RawValueRange
0x464   uint16   FrameRate (declared)
```

`0x384` is read as `uint32` Unix seconds, `uint32` with milliseconds in the low half, `int16`
timezone minutes — the layout [ET] documents and the values that make a measured frame rate
possible. Values that are physically implausible (emissivity > 1, field of view > 180°, Kelvin
outside −50…1200 °C, humidity outside 0…1 after normalization) are treated as **absent**, not
clamped, and a zero-filled Planck block means "no calibration" — the conversion is then refused.

The record is read **per frame**: cameras rewrite it every frame, and the metadata report records
how many times the calibration signature changed during a recording.

## 6. What remains camera-specific (the T540 question)

The format is the T-series one, but a T540 recording has not been inspected here. The decoder
therefore never assumes the model, the dimensions, the codec or the presence of timestamps: it
reads them from the frame and reports them. If a T540 writes a different payload codec (for
example a newer JPEG-LS profile), a differently sized CameraInfo record, or no per-frame clock,
the tool says so explicitly instead of producing numbers. `validation.md` has the exact checklist
to confirm on the first real T540 file.

## 7. Provenance and licence

* This implementation is original code written against the public descriptions above.
* `imagecodecs` (BSD-3), `pylibjpeg`/`pylibjpeg-libjpeg` (MIT), Pillow (HPND), h5py (BSD-3),
  numpy (BSD-3), scipy (BSD-3), PySide6 (LGPL) are third-party dependencies, installed by pip,
  not vendored.
* No FLIR SDK file, DLL or header is included.
