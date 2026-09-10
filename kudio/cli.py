# -*- coding: utf-8 -*-
"""``kudio`` command line: quick audio utilities.

    kudio devices                       # list input/output audio devices
    kudio info  file.wav                # duration / sr / channels / peak
    kudio report file.wav               # reference-free health check
    kudio loudness file.wav             # BS.1770 integrated loudness
    kudio normalize in.wav out.wav --lufs -23
    kudio vad   file.wav                # where the speech is
    kudio pitch file.wav                # f0 contour summary (pYIN)
    kudio enhance in.wav out.wav --method logmmse --noise mcra
    kudio compare in.wav --reference clean.wav   # rank every method
    kudio convert in.wav out.wav --rate 16000 --subtype PCM_16
    kudio convert in_dir out_dir --rate 16000 --recursive
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

    s_report = sub.add_parser(
        "report",
        help="reference-free health check (clipping, level, bandwidth); "
             "exits 1 when something is wrong, so it works in a shell test")
    s_report.add_argument("file")

    s_loud = sub.add_parser("loudness", help="integrated loudness in LUFS (BS.1770)")
    s_loud.add_argument("file")

    s_norm = sub.add_parser("normalize", help="normalise to a loudness or peak target")
    s_norm.add_argument("src")
    s_norm.add_argument("dst")
    target = s_norm.add_mutually_exclusive_group()
    target.add_argument("--lufs", type=float, default=None,
                        help="loudness target, e.g. -23 (EBU R 128)")
    target.add_argument("--peak", type=float, default=None,
                        help="peak target in [0, 1], e.g. 0.99")
    s_norm.add_argument("--subtype", default="PCM_16")

    s_enh = sub.add_parser("enhance", help="denoise a file (kudio.spectral_enhance)")
    s_enh.add_argument("src")
    s_enh.add_argument("dst")
    s_enh.add_argument("--method", default="logmmse")
    s_enh.add_argument("--noise", default="mcra")
    s_enh.add_argument("--n-fft", type=int, default=512)
    s_enh.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                       help="method parameter, repeatable (e.g. --set q=0.4)")
    s_enh.add_argument("--recursive", action="store_true",
                       help="treat src/dst as folders (kudio.enhance_folder)")
    s_enh.add_argument("--rate", type=int, default=None,
                       help="resample on the way in (folder mode)")

    s_cmp = sub.add_parser(
        "compare", help="run every enhancement method over one file and rank them")
    s_cmp.add_argument("src")
    s_cmp.add_argument("--reference", default=None,
                       help="the clean signal, if you have one")
    s_cmp.add_argument("--noise", default=None,
                       help="hold the noise estimator fixed (default: mcra)")
    s_cmp.add_argument("--noises", nargs="+", default=None,
                       help="sweep these estimators too — usually the wider "
                            "axis; 'all' expands to every one")
    s_cmp.add_argument("--methods", nargs="+", default=None)
    s_cmp.add_argument("--write-to", default=None,
                       help="write each method's output into this folder")
    s_cmp.add_argument("--no-metrics", action="store_true",
                       help="skip PESQ / STOI even when installed")

    s_vad = sub.add_parser("vad", help="print the speech spans found in a file "
                                       "(exits 1 when there is no speech)")
    s_vad.add_argument("file")
    s_vad.add_argument("--threshold-db", type=float, default=8.0)
    s_vad.add_argument("--split-to", default=None,
                       help="write each span as its own file in this folder")

    s_pitch = sub.add_parser(
        "pitch", help="summarise the f0 contour of a file (kudio.f0); "
                      "exits 1 when nothing is voiced")
    s_pitch.add_argument("file")
    s_pitch.add_argument("--fmin", type=float, default=None,
                         help="bottom of the search range in Hz (default 65)")
    s_pitch.add_argument("--fmax", type=float, default=None,
                         help="top of the search range in Hz (default 400)")
    s_pitch.add_argument("--hop", type=int, default=None,
                         help="samples between frames (default: 10 ms)")
    s_pitch.add_argument("--frame-length", type=int, default=2048)
    s_pitch.add_argument("--segments", action="store_true",
                         help="list the voiced spans as well as the summary")
    s_pitch.add_argument("--labels", default=None,
                         help="write the voiced spans as a label track")

    s_conv = sub.add_parser("convert", help="resample / re-encode a wave file or folder")
    s_conv.add_argument("src")
    s_conv.add_argument("dst")
    s_conv.add_argument("--rate", type=int, default=None)
    s_conv.add_argument("--subtype", default="PCM_16")
    s_conv.add_argument("--recursive", action="store_true",
                        help="treat src/dst as folders (kudio.convert_folder)")
    s_conv.add_argument("--lufs", type=float, default=None,
                        help="also loudness-normalise (folder mode)")
    s_conv.add_argument("--trim-db", type=float, default=None,
                        help="also trim silence at this top_db (folder mode)")

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


def _cmd_devices(kudio) -> None:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import kudio

    if args.command == "devices":
        _cmd_devices(kudio)

    elif args.command == "info":
        info = kudio.audio_info(args.file)
        print(f"file      : {args.file}")
        print(f"samplerate: {info.sr} Hz")
        print(f"channels  : {info.channels}")
        print(f"duration  : {info.duration:.3f} s ({info.frames} samples)")
        print(f"encoding  : {info.format} / {info.subtype}")
        print(f"peak      : {info.peak:.4f}")

    elif args.command == "report":
        y, sr = kudio.file_load(args.file, sr=None, mono=True)
        report = kudio.audio_report(y, sr)
        print(report)
        problems = report.problems()
        if not problems:
            print("\nNothing wrong that a measurement can see.")
        else:
            print()
            for problem in problems:
                print(f"· {problem}")
        # a technically clean recording is not the same as one that sounds
        # good; say so rather than let the empty list read as a verdict
        print("\n(This is a technical check. A perceptual score with no "
              "reference needs a trained model — see kudio.dnsmos.)")
        return 1 if problems else 0

    elif args.command == "loudness":
        y, sr = kudio.file_load(args.file, sr=None, mono=True)
        print(f"{kudio.loudness(y, sr):.2f} LUFS")

    elif args.command == "normalize":
        y, sr = kudio.file_load(args.src, sr=None, mono=True)
        if args.peak is not None:
            out = kudio.normalize(y, peak=args.peak)
            what = f"peak {args.peak}"
        else:
            lufs = -23.0 if args.lufs is None else args.lufs
            out = kudio.normalize_lufs(y, sr, lufs=lufs)
            what = f"{lufs} LUFS"
        kudio.save_wave(args.dst, out, sr, subtype=args.subtype)
        print(f"wrote {args.dst} normalised to {what}")

    elif args.command == "enhance":
        # the folder branch is chosen before anything is read: in --recursive
        # mode `src` is a directory, which file_load cannot open
        params = {}
        for item in args.set:
            if "=" not in item:
                print(f"--set expects NAME=VALUE, got {item!r}")
                return 2
            name, value = item.split("=", 1)
            params[name.strip()] = float(value)

        if args.recursive:
            result = kudio.enhance_folder(
                args.src, args.dst, args.method, noise=args.noise,
                sr=args.rate, n_fft=args.n_fft, report=True, **params)
            print(f"{result.written}/{result.total} file(s) -> {args.dst}")
            reduction = result.mean_reduction_db()
            if reduction is not None:
                print(f"  noise floor down {reduction:.1f} dB on average")
            for path, reason in result.failed[:10]:
                print(f"  failed: {path.name}: {reason}")
            if len(result.failed) > 10:
                print(f"  … and {len(result.failed) - 10} more")
            return 1 if result.failed else 0

        y, sr = kudio.file_load(args.src, sr=None, mono=True)
        out = kudio.spectral_enhance(y, sr, args.method, noise=args.noise,
                                     n_fft=args.n_fft, **params)
        kudio.save_wave(args.dst, out, sr)
        method = kudio.METHODS[args.method]
        print(f"{method.label} + {args.noise} noise estimate -> {args.dst}")
        print(f"  noise floor {kudio.audio_report(y, sr).noise_floor_dbfs:+.1f} "
              f"-> {kudio.audio_report(out, sr).noise_floor_dbfs:+.1f} dBFS")

    elif args.command == "compare":
        y, sr = kudio.file_load(args.src, sr=None, mono=True)
        reference = None
        if args.reference:
            reference, ref_sr = kudio.file_load(args.reference, sr=sr, mono=True)
        if args.noise and args.noises:
            print("pass --noise to fix the estimator or --noises to sweep it, "
                  "not both")
            return 2
        sweep = args.noises
        if sweep == ["all"]:
            sweep = list(kudio.NOISE_ESTIMATORS)
        options = {"noises": sweep} if sweep else {"noise": args.noise or "mcra"}
        results = kudio.compare_enhancers(
            y, sr, args.methods, reference=reference,
            metrics=not args.no_metrics, **options)
        # best first when there is something to rank by; a ranking table that
        # opens unranked makes the reader do the sorting
        if reference is not None:
            results = sorted(
                results, key=lambda r: (r.error is not None, -(r.si_sdr_db or 0)))

        header = f"{'method + noise':<28}{'ms':>7}{'floor dBFS':>12}"
        if reference is not None:
            header += f"{'SI-SDR':>10}{'dSNR':>8}{'PESQ':>7}{'STOI':>7}"
        print(header)
        print("-" * len(header))
        for r in results:
            if r.error:
                print(f"{r.label:<28}  failed: {r.error}")
                continue
            row = f"{r.label:<28}{r.seconds * 1000:7.0f}{r.noise_floor_db:12.1f}"
            if reference is not None:
                row += f"{r.si_sdr_db:10.2f}{r.delta_snr_db:8.2f}"
                row += f"{r.pesq:7.2f}" if r.pesq is not None else f"{'—':>7}"
                row += f"{r.stoi:7.2f}" if r.stoi is not None else f"{'—':>7}"
            print(row)

        if reference is not None and results and not results[0].error:
            scored = [r for r in results if r.error is None]
            if not any(r.delta_snr_db > 0 for r in scored):
                print("\nNothing beat leaving the file alone. Either it is "
                      "already clean enough that the distortion costs more "
                      "than the noise removed, or the reference is not clean.")

        if reference is None:
            print("\nNo reference given, so these are residual-noise readings, "
                  "not quality scores. Listen before believing the smallest "
                  "floor is the best result — total suppression sounds worse "
                  "than a quiet noise bed.")
        if args.write_to:
            out_dir = Path(args.write_to)
            stem = Path(args.src).stem
            for r in results:
                if r.error:
                    continue
                kudio.save_wave(out_dir / f"{stem}_{r.method}.wav", r.audio, sr)
            print(f"\nwrote {len([r for r in results if not r.error])} file(s) "
                  f"to {out_dir}")

    elif args.command == "vad":
        y, sr = kudio.file_load(args.file, sr=None, mono=True)
        spans = kudio.vad(y, sr, threshold_db=args.threshold_db)
        if not spans:
            print("no speech detected")
            return 1
        total = sum(b - a for a, b in spans)
        for n, (start, end) in enumerate(spans, 1):
            print(f"  {n:>3}  {start:8.3f} → {end:8.3f}  ({end - start:.3f}s)")
        print(f"{len(spans)} span(s), {total:.2f}s of "
              f"{len(y) / sr:.2f}s ({total / (len(y) / sr) * 100:.0f}%)")
        if args.split_to:
            out_dir = Path(args.split_to)
            for n, segment in enumerate(kudio.vad_split(
                    y, sr, threshold_db=args.threshold_db), 1):
                kudio.save_wave(out_dir / f"{Path(args.file).stem}_{n:03d}.wav",
                                segment, sr)
            print(f"wrote {len(spans)} file(s) to {out_dir}")

    elif args.command == "pitch":
        y, sr = kudio.file_load(args.file, sr=None, mono=True)
        kwargs = {'frame_length': args.frame_length}
        if args.fmin is not None:
            kwargs['fmin'] = args.fmin
        if args.fmax is not None:
            kwargs['fmax'] = args.fmax
        if args.hop:
            kwargs['hop_length'] = args.hop
        track = kudio.f0(y, sr, **kwargs)
        print(track)
        if not track.voiced_ratio:
            return 1
        segments = track.voiced_segments()
        if args.segments:
            for n, (start, end) in enumerate(segments, 1):
                print(f"  {n:>3}  {start:8.3f} → {end:8.3f}  ({end - start:.3f}s)")
        if args.labels:
            kudio.save_labels(args.labels, track.to_labels())
            print(f"wrote {len(segments)} label(s) to {args.labels}")

    elif args.command == "convert":
        if args.recursive:
            result = kudio.convert_folder(
                args.src, args.dst, sr=args.rate, subtype=args.subtype,
                trim_db=args.trim_db, lufs=args.lufs)
            print(f"{result.written}/{result.total} file(s) -> {args.dst}")
            for path, reason in result.failed[:10]:
                print(f"  failed: {path.name}: {reason}")
            if len(result.failed) > 10:
                print(f"  … and {len(result.failed) - 10} more")
            return 1 if result.failed else 0
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
