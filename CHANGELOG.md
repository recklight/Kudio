# Changelog

All notable changes to **Kudio** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-11-28

kudio reads, writes, measures and enhances speech audio, on Python 3.9 and
later.

- The core install is numpy 1.24, scipy 1.10, soundfile 0.12, soxr 0.3.2,
  PyYAML 6.0 and tqdm 4.60 or later.
- kudio computes its spectral and pitch analysis itself, on librosa's
  conventions: the short-time transform and its inverse, mel filters, mel
  spectrograms and MFCC, cutting at silence, pYIN and resampling through soxr.
  The results are librosa's to the bit except where
  [the validation page](https://recklight.github.io/Kudio/validation/#kudio-and-librosa)
  lists a difference on purpose: the length of a clip resampled from the
  11025 Hz family, pYIN's search against librosa 1.0's default, and the time
  stretch and pitch shift, which are a phase vocoder of its own.
- The wavelet denoiser, BSS Eval's SDR, SIR and SAR, STOI and ESTOI, and the
  BS.1770 loudness meter are kudio's own as well.
- `tests/test_against_*.py` hold them to librosa 0.11 and 1.0, PyWavelets,
  mir_eval 0.8.2 and pystoi 0.4.1, and `test_loudness` to pyloudnorm.
  `[check]` installs these, and CI runs the comparison so that it cannot skip;
  [How the metrics are checked](https://recklight.github.io/Kudio/validation/)
  has the numbers.
- Extras:
  - `[audio]`: sounddevice, for recording, playback and the device list;
  - `[pesq]`: PESQ, which compiles;
  - `[dnsmos]`: onnxruntime;
  - `[viz]`: matplotlib;
  - `[data]`: pandas and openpyxl;
  - `[compressed]`: audioread, for m4a, aac and webm with ffmpeg;
  - `[eval]`: an empty extra, so that `pip install "kudio[eval]"`
    installs;
  - `[all]`: every one of these but `[pesq]`.
- `kudio` on the command line reads, converts, scores, enhances and
  synthesises datasets; `kudio devices` lists the sound devices by the one
  index every recording function takes.
- `kudio.cite()` gives the Zenodo concept DOI, 10.5281/zenodo.24219274.

[5.0.0]: https://github.com/recklight/Kudio/releases/tag/v5.0.0
