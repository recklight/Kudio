# Kudio

Lightweight, composable audio toolkit for acoustic inspection.

See the [README](https://github.com/recklight/kudio) for the quick start and
the [API Reference](api.md) for the full function list.

## Install

```bash
pip install kudio            # core: I/O, features, effects, synthesis, metrics
pip install kudio[audio]     # + live capture / playback (PyAudio, sounddevice)
pip install kudio[eval]      # + STOI / SDR back-ends (pystoi, mir_eval)
pip install kudio[pesq]      # + PESQ (pesq) -- builds from source, needs a C compiler
pip install kudio[viz]       # + plotting helpers (matplotlib)
pip install kudio[data]      # + Excel / dataframe export (pandas, openpyxl)
pip install kudio[all]       # everything but [pesq]
```
