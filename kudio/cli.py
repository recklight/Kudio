# -*- coding: utf-8 -*-
"""``kudio`` command line: quick audio utilities.

    kudio devices                       # list input/output audio devices
    kudio info  file.wav                # duration / sr / channels / peak
    kudio report file.wav               # reference-free health check
    kudio loudness file.wav             # integrated, range, true peak
    kudio loudness file.wav --target -23   # exit 1 if it misses EBU R 128
    kudio normalize in.wav out.wav --lufs -23
    kudio vad   file.wav                # where the speech is
    kudio pitch file.wav                # f0 contour summary (pYIN)
    kudio align ref.wav deg.wav         # how far apart are these two?
    kudio room ir.wav                   # T30 / EDT / C50 / DRR of a room
    kudio room --make room.wav --rt60 0.6   # ...or synthesise one
    kudio channel in.wav out.wav --telephone   # what the line did to it
    kudio channel in.wav out.wav --loss 0.05 --burst 6   # ...a bad link
    kudio enhance in.wav out.wav --method logmmse --noise mcra
    kudio compare in.wav --reference clean.wav   # rank every method
    kudio convert in.wav out.wav --rate 16000 --subtype PCM_16
    kudio convert in_dir out_dir --rate 16000 --recursive
    kudio synth  --clean C --noise N --out O --snr -5 0 5
    kudio synth  ... --rt60 0.3 0.6    # ...in rooms, too
    kudio synth  ... --channel telephone --targets
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

    s_loud = sub.add_parser(
        "loudness", help="integrated loudness in LUFS (BS.1770), plus range, "
                         "true peak and where the level moved")
    s_loud.add_argument("file")
    s_loud.add_argument("--over-time", action="store_true",
                        help="print the short-term curve, one line per second")
    s_loud.add_argument("--momentary", action="store_true",
                        help="use the 400 ms window rather than the 3 s one")
    s_loud.add_argument("--target", type=float, default=None,
                        help="a delivery target in LUFS, e.g. -23 (EBU R 128); "
                             "exits 1 when the file misses it by over 1 LU")

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

    s_align = sub.add_parser(
        "align", help="measure the delay between two recordings of the same "
                      "thing (kudio.find_delay); exits 1 when they do not "
                      "look like the same recording")
    s_align.add_argument("ref")
    s_align.add_argument("deg")
    s_align.add_argument("--max-seconds", type=float, default=None,
                         help="bound the search to this much delay either way")
    s_align.add_argument("--write-to", default=None,
                         help="write the aligned pair into this folder")
    s_align.add_argument("--metrics", action="store_true",
                         help="score deg against ref, before and after")

    s_room = sub.add_parser(
        "room", help="measure a room from its impulse response (ISO 3382-1), "
                     "or synthesise one; exits 1 when the decay does not look "
                     "like an impulse response")
    s_room.add_argument("file", nargs="?", default=None,
                        help="an impulse response to measure")
    s_room.add_argument("--make", default=None, metavar="OUT.WAV",
                        help="synthesise an impulse response instead")
    s_room.add_argument("--rt60", type=float, default=0.5,
                        help="reverberation time to build, in seconds")
    s_room.add_argument("--drr", type=float, default=0.0,
                        help="direct-to-reverberant ratio to build, in dB")
    s_room.add_argument("--rate", type=int, default=16000,
                        help="sample rate to build at")
    s_room.add_argument("--seed", type=int, default=None)
    s_room.add_argument("--apply", default=None, metavar="OUT.WAV",
                        help="treat `file` as audio, put it through a room "
                             "built from --rt60/--drr, and write it here")

    s_chan = sub.add_parser(
        "channel", help="degrade a file the way a transmission does: "
                        "companding, quantisation, packet loss")
    s_chan.add_argument("src")
    s_chan.add_argument("dst")
    s_chan.add_argument("--telephone", action="store_true",
                        help="the lot: 300-3400 Hz at 8 kHz through G.711")
    s_chan.add_argument("--codec", default=None,
                        choices=["mu_law", "a_law"],
                        help="compand without the band-limiting")
    s_chan.add_argument("--bits", type=int, default=None,
                        help="requantise linearly to this many bits")
    s_chan.add_argument("--loss", type=float, default=None,
                        help="fraction of packets to drop, e.g. 0.05")
    s_chan.add_argument("--packet-ms", type=float, default=20.0)
    s_chan.add_argument("--burst", type=float, default=None,
                        help="lose those packets in runs of this many on "
                             "average, the way a real link does. Independent "
                             "loss (the default) already runs to 1/(1-loss)")
    s_chan.add_argument("--conceal", default="silence",
                        choices=["silence", "hold"])
    s_chan.add_argument("--seed", type=int, default=None)

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
    s_syn.add_argument("--rt60", type=float, nargs="+", default=None,
                       help="reverberate the speech in rooms of these "
                            "reverberation times, in seconds. A fourth axis: "
                            "in inc mode the output count multiplies by it")
    s_syn.add_argument("--drr", type=float, default=0.0,
                       help="direct-to-reverberant ratio for those rooms, in "
                            "dB (default 0, about a metre or two away)")
    s_syn.add_argument("--channel", default=None,
                       help="send every mixture down a link: telephone, voip, "
                            "mu_law, a_law. Constant for the run, not an axis "
                            "-- a corpus is recorded over a phone line or it "
                            "is not")
    s_syn.add_argument("--channel-loss", type=float, default=None,
                       help="override that link's packet loss, e.g. 0.05")
    s_syn.add_argument("--channel-burst", type=float, default=None,
                       help="...lost in runs of this many packets")
    s_syn.add_argument("--targets", action="store_true",
                       help="also write the reverberant-but-clean signal to "
                            "OUT/TARGETS, for training that should remove the "
                            "noise and leave the room (needs --rt60)")
    s_syn.add_argument("--manifest", default=None, metavar="OUT.JSON",
                       help="write what came from what, rooms and link "
                            "included")

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
    import numpy as np

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
        integrated = kudio.loudness(y, sr)
        print(f"integrated  {integrated:8.2f} LUFS")
        window = kudio.MOMENTARY if args.momentary else kudio.SHORT_TERM
        try:
            curve = kudio.loudness_over_time(y, sr, window=window)
        except Exception:                    # the clip is shorter than a window
            print(f"range       — the clip is shorter than the "
                  f"{window:g} s window")
            curve = None
        else:
            print(f"range       {kudio.loudness_range(y, sr):8.2f} LU")
            loud_at, loud = curve.loudest()
            quiet_at, quiet = curve.quietest()
            label = "momentary" if args.momentary else "short-term"
            print(f"{label:<11} {loud:8.2f} LUFS at {loud_at:7.2f}s  (loudest)")
            print(f"{'':<11} {quiet:8.2f} LUFS at {quiet_at:7.2f}s  (quietest)")
        print(f"true peak   {kudio.true_peak(y, sr):8.2f} dBTP")

        if curve is not None and args.over_time:
            print()
            for t, value in zip(curve.times, curve.lufs):
                # one column per 2 LU above a -60 LUFS floor, so the shape of
                # the take is visible without opening anything
                blocks = 0 if not np.isfinite(value) else \
                    max(0, int(round(value + 60)) // 2)
                print(f"  {t:7.2f}s  {value:8.2f}  {'#' * blocks}")

        if args.target is not None:
            gap = integrated - args.target
            if not np.isfinite(gap):
                print(f"\nsilent — nothing to compare against "
                      f"{args.target:g} LUFS")
                return 1
            print(f"\n{abs(gap):.2f} LU {'over' if gap > 0 else 'under'} "
                  f"the {args.target:g} LUFS target")
            if abs(gap) > 1.0:
                return 1

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

    elif args.command == "align":
        ref, sr = kudio.file_load(args.ref, sr=None, mono=True)
        deg, deg_sr = kudio.file_load(args.deg, sr=None, mono=True)
        if deg_sr != sr:
            # a sample offset is not a quantity yet if the two are not on the
            # same clock; resample first and say so
            print(f"resampling {Path(args.deg).name} from {deg_sr} to {sr} Hz")
            deg = kudio.resample(deg, deg_sr, sr)

        found = kudio.find_delay(ref, deg, sr, max_seconds=args.max_seconds)
        print(found)
        print(f"overlap     {found.overlap / sr:.3f}s of "
              f"{len(ref) / sr:.3f}s / {len(deg) / sr:.3f}s")
        if not found.confident():
            print("\nthese do not look like the same recording — the delay "
                  "above does not mean anything")
            return 1

        a, b = found.apply(ref, deg)
        if args.metrics:
            before = kudio.si_sdr(ref, deg[:len(ref)]) if len(deg) else float("nan")
            print(f"\nsi_sdr      {before:8.2f} dB  as the files sit")
            print(f"            {kudio.si_sdr(a, b):8.2f} dB  aligned")
        if args.write_to:
            out_dir = Path(args.write_to)
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, wave in ((Path(args.ref).stem, a), (Path(args.deg).stem, b)):
                kudio.save_wave(out_dir / f"{name}_aligned.wav", wave, sr)
            print(f"\nwrote 2 file(s) to {out_dir}")

    elif args.command == "room":
        if args.make:
            ir = kudio.rir(args.rate, rt60=args.rt60, drr_db=args.drr,
                           seed=args.seed)
            kudio.save_wave(args.make, ir, args.rate)
            measured = kudio.rt60(ir, args.rate)
            print(f"wrote {args.make} ({len(ir) / args.rate:.2f}s at "
                  f"{args.rate} Hz)")
            print(measured)
            return 0

        if not args.file:
            print("give an impulse response to measure, audio with --apply, "
                  "or --make a room")
            return 1

        if args.apply:
            y, sr = kudio.file_load(args.file, sr=None, mono=True)
            room = kudio.rir(sr, rt60=args.rt60, drr_db=args.drr,
                             seed=args.seed)
            kudio.save_wave(args.apply, kudio.apply_rir(y, room), sr)
            print(f"wrote {args.apply} — {args.rt60:g}s room, "
                  f"DRR {args.drr:+g} dB, at {sr} Hz")
            print(kudio.rt60(room, sr))
            return 0

        ir, sr = kudio.file_load(args.file, sr=None, mono=True)
        measured = kudio.rt60(ir, sr)
        print(measured)
        print(f"T20         {measured.t20:8.3f} s")
        print(f"T30         {measured.t30:8.3f} s")
        print(f"EDT         {measured.edt:8.3f} s")
        print(f"C50         {measured.c50:+8.2f} dB")
        print(f"C80         {measured.c80:+8.2f} dB")
        print(f"DRR         {measured.drr:+8.2f} dB")

        if args.apply:
            print("\n--apply needs audio to put through the room, and this "
                  "file is the room itself; pass the audio as `file` and the "
                  "room with --make first")
            return 1
        return 0 if measured.reliable() else 1

    elif args.command == "channel":
        source, sr = kudio.file_load(args.src, sr=None, mono=True)
        y = source
        before = kudio.audio_report(y, sr)
        steps = []

        if args.telephone:
            y = kudio.telephone(y, sr, codec=args.codec or "mu_law")
            steps.append(f"telephone ({args.codec or 'mu_law'})")
        elif args.codec:
            y = (kudio.mu_law if args.codec == "mu_law" else kudio.a_law)(y)
            steps.append(args.codec)
        if args.bits is not None:
            y = kudio.bit_depth(y, args.bits)
            steps.append(f"{args.bits}-bit")
        if args.loss is not None:
            y, gaps = kudio.dropouts(y, sr, loss=args.loss,
                                     packet_ms=args.packet_ms,
                                     conceal=args.conceal, burst=args.burst,
                                     seed=args.seed, return_spans=True)
            lost = sum(b - a for a, b in gaps)
            if gaps:
                runs = 1 + sum(1 for a, b in zip(gaps, gaps[1:])
                               if b[0] - a[1] > 1e-9)
                steps.append(f"{len(gaps)} packet(s) lost in {runs} run(s) of "
                             f"{len(gaps) / runs:.1f}, {lost:.2f}s, "
                             f"{args.conceal}")
            else:
                steps.append("nothing lost — the file is shorter than a packet")

        if not steps:
            print("nothing asked for — try --telephone, --codec, --bits "
                  "or --loss")
            return 1

        kudio.save_wave(args.dst, y, sr)
        after = kudio.audio_report(y, sr)
        print(f"wrote {args.dst}")
        for step in steps:
            print(f"  · {step}")
        print(f"bandwidth  {before.bandwidth_hz:7.0f} Hz -> "
              f"{after.bandwidth_hz:7.0f} Hz")
        # the channel does not delay anything, so the two already line up;
        # `align` would find a lag of zero and cost a correlation to say so
        print(f"si_sdr     {kudio.si_sdr(source[:len(y)], y):+7.2f} dB "
              f"against the original")

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
        link = args.channel
        if link is not None and (args.channel_loss is not None
                                 or args.channel_burst is not None):
            link = {"preset": link}
            if args.channel_loss is not None:
                link["loss"] = args.channel_loss
            if args.channel_burst is not None:
                link["burst"] = args.channel_burst
        syx = kudio.Synthesizer(args.clean, args.noise, out_path=args.out,
                                snr_ratio=args.snr, rt60=args.rt60,
                                drr_db=args.drr, channel=link,
                                write_targets=args.targets)
        result = syx.syn(mode=args.mode, seed=args.seed)
        print(f"synthesized {len(result or [])} files -> {args.out}")
        if args.rt60:
            rooms = ", ".join(f"{v:g}s" for v in args.rt60)
            print(f"rooms: {rooms} at DRR {args.drr:+g} dB — the SNR is "
                  f"against the reverberant speech, and the clean file is "
                  f"still the target")
        if syx.channel_tag:
            print(f"link: {syx.channel_tag} — applied to the mixture, noise "
                  f"and all, and the same for every file. The clean file is "
                  f"still the target, so a model is asked to undo it")
        if syx.targets:
            print(f"targets: {len(syx.targets)} reverberant-clean file(s) -> "
                  f"{args.out}/TARGETS")
        if args.manifest:
            kudio.save_manifest(args.manifest, syx.manifest())
            print(f"manifest -> {args.manifest}")

    elif args.command == "trim":
        y, sr = kudio.file_load(args.src, sr=None, mono=True)
        trimmed, (a, b) = kudio.trim_silence(y, top_db=args.top_db)
        kudio.save_wave(args.dst, trimmed, sr)
        print(f"trimmed [{a}:{b}] -> {args.dst} ({len(trimmed)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
