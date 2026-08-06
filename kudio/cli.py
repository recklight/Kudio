# -*- coding: utf-8 -*-
"""``kudio`` command line: quick audio utilities.

    kudio devices                       # list input/output audio devices
    kudio info  file.wav                # duration / sr / channels / peak
    kudio convert in.wav out.wav --rate 16000 --subtype PCM_16
    kudio synth  --clean C --noise N --out O --snr -5 0 5
    kudio trim  in.wav out.wav --top-db 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("kudio.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kudio", description="kudio audio utilities")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio devices (needs kudio[audio])")

    s_info = sub.add_parser("info", help="show audio file properties")
    s_info.add_argument("file")

    s_conv = sub.add_parser("convert", help="resample / re-encode a wave file")
    s_conv.add_argument("src")
    s_conv.add_argument("dst")
    s_conv.add_argument("--rate", type=int, default=None)
    s_conv.add_argument("--subtype", default="PCM_16")

    s_syn = sub.add_parser("synth", help="mix clean + noise at given SNRs")
    s_syn.add_argument("--clean", required=True)
    s_syn.add_argument("--noise", required=True)
    s_syn.add_argument("--out", required=True)
    s_syn.add_argument("--snr", type=int, nargs="+", default=[-5, 0, 5])
    s_syn.add_argument("--mode", default="inc", choices=["inc", "reg"])
    s_syn.add_argument("--seed", type=int, default=None)

    s_trim = sub.add_parser("trim", help="trim leading/trailing silence")
    s_trim.add_argument("src")
    s_trim.add_argument("dst")
    s_trim.add_argument("--top-db", type=float, default=30.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import kudio

    if args.command == "devices":
        # the two backends number devices independently, so print both spaces
        # rather than let an index from one be pasted into the other
        print("sounddevice indices — use with kudio.record(device=...)")
        try:
            for d in kudio.list_devices():
                print(f"  [{d['index']:>2}] in={d['max_input_channels']} "
                      f"out={d['max_output_channels']} "
                      f"fs={int(d['default_samplerate'])}  {d['name']}")
        except Exception as e:
            print(f"  unavailable: {e}")

        print("\nPyAudio indices — use with kudio.Recorder / stream readers")
        try:
            with kudio.CheckDevice() as cd:
                for d in cd.system_devices():
                    print(f"  [{d['index']:>2}] in={d['in']} out={d['out']} "
                          f"fs={int(d['fs'])}  {d['Device']}")
        except Exception as e:
            print(f"  unavailable: {e}")
    elif args.command == "info":
        y, sr = kudio.file_load(args.file, sr=None, mono=False)
        ch = 1 if y.ndim == 1 else y.shape[1]
        n = y.shape[0]
        import numpy as np
        print(f"file      : {args.file}")
        print(f"samplerate: {sr} Hz")
        print(f"channels  : {ch}")
        print(f"duration  : {n / sr:.3f} s ({n} samples)")
        print(f"peak      : {float(np.max(np.abs(y))):.4f}")
    elif args.command == "convert":
        y, sr = kudio.file_load(args.src, sr=args.rate, mono=True)
        kudio.save_wave(args.dst, y, sr, subtype=args.subtype)
        print(f"wrote {args.dst} ({sr} Hz, {args.subtype})")
    elif args.command == "synth":
        syx = kudio.Synthesizer(args.clean, args.noise, out_path=args.out,
                                snr_ratio=args.snr)
        result = syx.syn(mode=args.mode, seed=args.seed)
        print(f"synthesized {len(result or [])} files -> {args.out}")
    elif args.command == "trim":
        y, sr = kudio.file_load(args.src, sr=None, mono=True)
        trimmed, (a, b) = kudio.trim_silence(y, top_db=args.top_db)
        kudio.save_wave(args.dst, trimmed, sr)
        print(f"trimmed [{a}:{b}] -> {args.dst} ({len(trimmed)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
