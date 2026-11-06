# Troubleshooting: JPEG-LS decode failures and memory

This page started from a real server report: a 25 661-frame recording ran for 6 min
50 s, reached roughly frame 3000, exited with code 4, and SLURM recorded
`MaxRSS = 4.8 GiB`. The other files in the same batch failed with "JPEG-LS decoding
errors" as well. This is what was measured, what was actually wrong, and what to run
on your data.

## 1. Measurements: the decoder and the writers do not leak

All measured on this repository, T540 geometry where noted, with the synthetic
long-recording generator (`scripts/profile_memory.py` is the script used):

| stage | frames | peak RSS | growth over run |
|---|---|---|---|
| `imagecodecs.jpegls_decode`, real 1024×768 payload | 1 000 | 41.9 MB | **+0.2 MB** |
| decode only (464×348 frames from a recording) | 3 000 | 39.6 MB | **+2.1 MB** |
| decode + radiometry (Planck, corrections) | 3 000 | 48.9 MB | **+5.9 MB** |
| full conversion → HDF5 (temperatures + raw counts) | 3 000 | 92.8 MB | +50 MB (allocator warm-up, not linear) |
| full conversion → MATLAB v7.3, no raw counts | 2 500 | 77.1 MB | +34.7 MB (same plateau) |
| Python heap retained (tracemalloc) | 2 000 | 14.5 MB | 7.1 MB retained |

The growth figures are a warm-up: 1 000 frames grew +45 MB while 2 500 frames grew
+34.7 MB, so it is not a function of frame count. The retained Python heap is the
per-frame statistics list (~0.4 KB/frame) plus the camera metadata, i.e. about
10 MB for 25 000 frames.

**Conclusion:** the decode path, the radiometry, the HDF5 writer and the MATLAB
v7.3 writer are bounded. A 4.8 GiB MaxRSS is not the steady-state cost of
converting a long recording.

## 2. What was actually wrong

### 2.1 A decoder failure was reported as "no decoder installed"

The fallback chain caught the codec's exception, tried the next decoder, and when
that also failed raised

```
cannot decode the jpegls payload (the frame is JPEG-LS compressed and no JPEG-LS
decoder is installed)
```

even though `imagecodecs` was installed and working on every other frame. The
original `JpeglsError` text — the only thing that says *why* — was thrown away.
That is why the server log was not actionable.

Fixed: a failing codec now raises `PayloadDecodeError`, which carries the codec's
own exception type and message, every decoder that was tried, the payload size and
the **frame index**:

```
frame 3041: the jpegls decoder could not read this payload
  (imagecodecs/CharLS (JPEG-LS): JpeglsError: charls_jpegls_decoder_decode_to_buffer
   returned Invalid JPEG-LS stream, the encoded bit stream contains a general
   structural problem)
```

"no decoder installed" is now only reported when no decoder is installed.

### 2.2 A corrupt frame header could ask for 6.7 GiB

This is the only mechanism found that can inflate RSS by gigabytes from a single
frame. A JPEG-LS stream declares its dimensions in the SOF55 segment. If that header
is corrupt and claims 60000×60000, `imagecodecs` allocates the output before it
notices the data is short:

```
decoding a stream whose header claims 60000x60000, payload 198848 bytes
raised MemoryError: Unable to allocate 6.71 GiB for an array with shape
(60000, 60000) and data type uint16
```

On a 4.8 GiB job this looks exactly like "the decoder has a memory leak": RSS
jumps, then the frame fails. It also explains why failures appear at different
points in different runs — it depends which corrupt frame is reached first.

Fixed twice over:

* the payload's SOF55 dimensions are read **before** decoding and compared with the
  recording's own frame dimensions; a mismatch is a clean error, not a decode
  attempt;
* the decode is given an `out=` buffer of exactly the recording's frame size, so the
  codec cannot allocate more than one frame whatever the header says. The same
  poisoned payload now peaks at **39 MB** instead of requesting 6.7 GiB.

### 2.3 Truncated payloads were decoded silently by the fallback codec

`imagecodecs`/CharLS rejects a truncated JPEG-LS stream. `pylibjpeg-libjpeg` — the
fallback — accepts it and returns **wrong pixel values**. Measured on a real frame
cut by 5 KB: 25 286 pixels differ from the correct decode, by up to 247 counts
(≈1.6 °C at 20 °C). That is silent scientific corruption.

Fixed:

* every JPEG-LS payload must carry an end-of-image marker (`FF D9`) near its end —
  all 22 frames of the reference T1020 recording do (within 9 bytes) — so a
  truncated frame is reported, not decoded;
* the fallback decoder is only used when the primary codec is **not installed**.
  `--allow-fallback-decoder` restores the old behaviour for exceptional files, with
  the caveat that it may accept damaged streams.

## 3. What to run on the server

### 3.1 Which frames are unreadable? (no conversion)

```bash
python convert_csq.py /path/to/test5.csq --check-frames --json > test5_frames.json
```

```
checked    25661 frames (703.4 frames/s, 36.5 s)
decoded    25658
failed     3  first at frame 3041
  frame 3041: jpegls payload is truncated: no end-of-image marker in the last 64 bytes
  frame 3042: ...
```

Run it twice. If the same indices fail, the payloads are corrupt in the file (a bad
copy, a camera hiccup, or the recording was still being written when it was copied)
— not a decoder race and not something more RAM will fix. `--dump-bad-frames DIR`
saves each unreadable payload so a single bad frame can be inspected or reported.

### 3.2 Keep the rest of the recording

```bash
python convert_csq.py /path/to/test5.csq --output-dir out \
  --on-decode-error recover --dump-bad-frames out/bad_frames --memory-log-every 2000
```

* unreadable frames become **NaN** in `temperature_C` and 0 in `raw_counts`;
* their **timestamps are preserved**, so the time axis stays intact;
* `/frame_status` is 1 for those frames (0 = decoded), and `metadata_json` lists
  them under `failed_frames` with the error, payload size and SHA-256;
* the file's status becomes `ok-with-gaps`; the batch summary counts
  `files_with_gaps` and `recovered_frames`;
* verification still passes: NaN frames are excluded from the recomputation check.

`--memory-log-every N` writes RSS and peak RSS to the log every N frames, so the
next SLURM run can be compared against the table above.

### 3.3 Re-run the failed files only

The batch state file records failures, so re-running the same command retries
exactly those files and skips the 3 that already converted:

```bash
FLIR_FORMATS=mat FLIR_MAT_VERSION=7.3 \
  ./run_server.sh /nfsd/iaslab4/Users/shavakhial/termography \
                  /nfsd/iaslab4/Users/shavakhial/termography/output
```

## 4. About the 4.8 GiB MaxRSS

The measurements above rule out the decode, radiometry, HDF5 and MATLAB v7.3 paths
as the steady-state cause. Before blaming the converter, check which settings the
job used:

| possibility | how to tell | fix |
|---|---|---|
| an old version (before this fix) met a frame with a corrupt SOF55 header | the error text said "no JPEG-LS decoder is installed" while `imagecodecs` was installed | update; the header is now validated first |
| `--format mat` with the **classic v7** container | `conversion.matlab_container` is `7` in the JSON report; peak RAM ≈ 2× cube size | use `--mat-version 7.3` (streams, no 2 GB limit) |
| `--format mat` with `auto` on a medium cube | same field says `7` | `auto` now switches to v7.3 above 256 MB |
| several files, one process | MaxRSS is the peak over the whole batch | run with `--memory-log-every` to see which file grows |
| the node really was short of memory | `sacct -j <job> -o MaxRSS,State,ExitCode`; dmesg/OOM messages | see §5 |

`--memory-log-every 2000` plus `sacct` is enough to attribute the peak to a file and
a frame range.

## 5. Sizing a job

Per frame of a T540 native recording (464×348 = 161 472 pixels):

* JPEG-LS payload: ~0.1–0.2 MB in the file;
* decoded counts: 0.32 MB (uint16);
* temperature (float32): 0.65 MB;
* HDF5/MAT v7.3 output with raw counts: ~0.65 MB/frame (gzip), ~0.45 MB without.

The converter holds one frame plus HDF5 buffers: **peak RSS stays under ~100 MB**
in the measurements above, so 2 GB of RAM is comfortable and 1 GB is enough. The
cube itself never has to fit in memory — that is what "streaming" means here. If a
job needs more than a few hundred MB, it is not the conversion loop; look at the
table in §4.

## 6. What is not claimed

* These measurements are from synthetic long recordings and a 22-frame real T1020
  file. The failing T540 recordings were not available, so the specific corrupt
  payloads could not be inspected.
* Recovering frames does not repair them: a NaN frame is missing data, and any
  statistic over it must use `nanmean`/`nansum` (the JSON report's per-frame
  statistics already do).
* If `--check-frames` reports *different* indices on repeated runs of the same
  unchanged file, that is not data corruption but something outside this tool
  (unstable storage, a network copy still in progress); re-copy and re-check before
  converting.
