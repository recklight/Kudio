# -*- coding: utf-8 -*-
"""What the transmission did to it: companding, quantisation, packet loss.

Speech arrives degraded in three ways and kudio could describe two of them.
Noise is **additive** -- :class:`kudio.Synthesizer` mixes it at a chosen SNR.
A room is **convolutive** -- :func:`kudio.apply_rir` smears the signal across
time. The third is the channel, and it is neither: a codec quantises, a link
drops packets, a converter throws bits away. None of that can be written as
``y + n`` or ``y * h``, which is why none of it was here.

>>> phone = kudio.telephone(speech, sr)          # 300-3400 Hz, 8 kHz, G.711
>>> lossy = kudio.dropouts(speech, sr, loss=0.05, seed=0)
>>> rough = kudio.bit_depth(speech, bits=8)

This is the augmentation that matters for anything that will meet a telephone
line or a conferencing codec, and it is well established for exactly that
(Audio Codec Simulation based Data Augmentation for Telephony Speech
Recognition, APSIPA 2019; packet-loss augmentation for contact-centre ASR,
Applied Sciences 2022).

**Companding is the interesting one.** :func:`mu_law` spends its 8 bits
logarithmically, so the quantisation noise follows the signal down instead of
sitting at a fixed floor: on this repo's fixture it holds ~37 dB of SNR over a
24 dB range of levels, where linear 8-bit loses the textbook 6 dB per halving.
That is the whole reason telephony uses it, and it is what the tests check.

**These are models, not bit-exact codecs.** :func:`mu_law` and :func:`a_law`
implement the G.711 companding *curves* quantised to 8 bits, not the
standard's piecewise-linear segment layout. Measured against Python's own
``audioop`` -- the reference, and removed in 3.13, which is why it cannot be a
dependency -- the round-trip SNR agrees to half a decibel and the waveforms to
0.013 full scale. For augmentation that is the same thing; for a bit-exact
decoder it is not, and this does not claim to be one.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from kudio.exceptions import FeatureError

__all__ = ['mu_law', 'a_law', 'bit_depth', 'dropouts', 'telephone',
           'independent_burst', 'channel_spec', 'channel_tag', 'apply_channel',
           'MU_LAW_MU', 'A_LAW_A', 'PACKET_MS', 'TELEPHONE_BAND',
           'CHANNEL_DEFAULTS', 'CHANNEL_PRESETS']

log = logging.getLogger(__name__)

#: G.711 companding constants: mu for the North American / Japanese law, A for
#: the European one.
MU_LAW_MU = 255.0
A_LAW_A = 87.6

#: Default packet length, in milliseconds. 20 ms is what G.711 over RTP uses,
#: so a lost packet is 20 ms of nothing.
PACKET_MS = 20.0

#: The passband a telephone line gives you, in Hz, and the rate it runs at.
TELEPHONE_BAND = (300.0, 3400.0)
TELEPHONE_SR = 8000

#: How a lost packet is filled in.
CONCEALMENT = ('silence', 'hold')


def independent_burst(loss: float) -> float:
    """The mean run length that *independent* loss at rate *loss* produces.

    Worth a function because the obvious guess is 1 and it is wrong.
    Independent draws do land side by side sometimes, so the runs already
    average ``1 / (1 - loss)`` -- 1.05 packets at 5% loss, 1.25 at 20%.
    Asking :func:`dropouts` for ``burst=1`` is therefore asking for something
    *less* clustered than the default, not for the default.

    >>> kudio.independent_burst(0.05)
    1.0526315789473684
    """
    if not 0.0 <= loss < 1.0:
        raise FeatureError(f"loss must be a fraction in [0, 1), got {loss}")
    return 1.0 / (1.0 - loss)


def _burst_runs(packets: int, n_lost: int, burst: float,
                rng: np.random.Generator) -> np.ndarray:
    """Which packets go, lost in runs averaging *burst* packets.

    Gilbert's two-state chain is the standard model and it fixes the *rate*,
    not the count -- ask it for 5% and a given file carries 4.1% or 6.3%.
    :func:`dropouts` promises the count, because a corpus labelled 5% should
    be 5%, so what is taken from Gilbert is the part that matters: the run
    lengths are geometric with mean *burst*, exactly as its bad state gives,
    and the total is then conditioned to be the number asked for.

    Runs are kept apart by at least one surviving packet. Without that two
    runs could abut and the realised mean would drift above the one
    requested, which would make *burst* unmeasurable and therefore not worth
    having as a parameter.
    """
    runs: List[int] = []
    total = 0
    probability = 1.0 / burst
    while total < n_lost:
        length = int(rng.geometric(probability)) if probability < 1.0 else 1
        length = min(length, n_lost - total)
        runs.append(length)
        total += length

    count = len(runs)
    survivors = packets - n_lost
    free = survivors - (count - 1)
    if free < 0:
        raise FeatureError(
            f"loss={n_lost / packets:.3f} in runs of {burst:g} packet(s) "
            f"needs about {count - 1} surviving packets to keep the runs "
            f"apart and only {survivors} survive. Either lose less or lose it "
            f"in longer runs.")

    # A uniform composition of the spare survivors over the gaps before each
    # run: stars and bars, so every arrangement is equally likely.
    if count and free:
        bars = np.sort(rng.choice(free + count, size=count, replace=False))
        edges = bars - np.arange(count)
        spare = np.diff(np.concatenate(([0], edges)))
    else:
        spare = np.zeros(count, dtype=int)

    lost: List[int] = []
    position = 0
    for index, length in enumerate(runs):
        position += int(spare[index])
        lost.extend(range(position, position + length))
        position += length + 1          # the mandatory survivor between runs
    return np.array(lost, dtype=int)


def _mono(name: str, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1:
        raise FeatureError(
            f"{name} takes mono audio, got shape {y.shape}. A channel degrades "
            f"each channel differently and mixing down first is a decision, "
            f"not a detail.")
    return y.astype(np.float64, copy=False).reshape(-1)


def _quantise(values: np.ndarray, bits: int) -> np.ndarray:
    """Round *values* in ``[-1, 1]`` onto a sign-and-magnitude grid.

    Sign kept separately, magnitude quantised to ``2**(bits-1) - 1`` steps.
    That is G.711's own layout, and it is the version with a code exactly at
    zero: spreading ``2**bits`` levels evenly across ``[-1, 1]`` leaves none
    there, so digital silence comes back as a small DC offset and the curve
    stops being odd. Both were true of the first version of this.
    """
    levels = 2 ** (bits - 1) - 1
    return np.clip(np.round(values * levels) / levels, -1.0, 1.0)


def mu_law(y: np.ndarray, *, bits: int = 8) -> np.ndarray:
    """Push *y* through the G.711 mu-law companding curve and back.

    :param bits: code word size. 8 is G.711; fewer is a cruder link and a way
        to ask how far a model's robustness reaches.

    The point of companding is that the quantisation noise tracks the signal:
    loud passages get coarse steps and quiet ones get fine steps, so the
    signal-to-noise ratio stays roughly constant instead of collapsing on
    anything quiet. :func:`bit_depth` is the same number of bits spent
    linearly, and is the comparison that shows what it buys.

    >>> kudio.mu_law(speech)                                 # doctest: +SKIP
    """
    y = _mono("mu_law()", y)
    if bits < 2:
        raise FeatureError(f"bits must be >= 2, got {bits}")
    y = np.clip(y, -1.0, 1.0)

    compressed = np.sign(y) * np.log1p(MU_LAW_MU * np.abs(y)) \
        / np.log1p(MU_LAW_MU)
    coded = _quantise(compressed, bits)
    expanded = np.sign(coded) * ((1.0 + MU_LAW_MU) ** np.abs(coded) - 1.0) \
        / MU_LAW_MU
    return expanded.astype(np.float32)


def a_law(y: np.ndarray, *, bits: int = 8) -> np.ndarray:
    """The same, through the A-law curve -- G.711 as Europe specifies it.

    A-law is slightly finer near silence and slightly coarser at the top than
    mu-law. The difference is audible only on very quiet material; it is here
    because a corpus recorded off a European line went through this one.
    """
    y = _mono("a_law()", y)
    if bits < 2:
        raise FeatureError(f"bits must be >= 2, got {bits}")
    y = np.clip(y, -1.0, 1.0)

    magnitude = np.abs(y)
    denominator = 1.0 + np.log(A_LAW_A)
    low = magnitude < 1.0 / A_LAW_A
    compressed = np.where(
        low,
        A_LAW_A * magnitude / denominator,
        (1.0 + np.log(np.maximum(A_LAW_A * magnitude, 1e-300))) / denominator)
    coded = _quantise(np.sign(y) * compressed, bits)

    size = np.abs(coded)
    low = size < 1.0 / denominator
    expanded = np.where(
        low,
        size * denominator / A_LAW_A,
        np.exp(size * denominator - 1.0) / A_LAW_A)
    return (np.sign(coded) * expanded).astype(np.float32)


def bit_depth(y: np.ndarray, bits: int = 8, *,
              dither: bool = False, seed: Optional[int] = None) -> np.ndarray:
    """Requantise *y* to *bits*, linearly.

    :param dither: add a triangular dither of one step before rounding. It
        raises the noise floor and removes the correlation between the error
        and the signal, which is what stops quiet passages sounding granular
        rather than merely noisy.

    A full-scale sine comes back at about ``6.02 * bits + 1.76`` dB of SNR,
    which is the textbook figure and what the tests assert.
    """
    y = _mono("bit_depth()", y)
    if bits < 2:
        raise FeatureError(f"bits must be >= 2, got {bits}")
    y = np.clip(y, -1.0, 1.0)

    if dither:
        step = 2.0 / (2 ** bits)
        rng = np.random.default_rng(seed)
        # triangular: the sum of two uniforms, one step peak to peak
        y = y + (rng.random(len(y)) - rng.random(len(y))) * step / 2.0
        y = np.clip(y, -1.0, 1.0)
    return _quantise(y, bits).astype(np.float32)


def dropouts(y: np.ndarray, sr: int, *, loss: float = 0.05,
             packet_ms: float = PACKET_MS, conceal: str = 'silence',
             burst: Optional[float] = None, seed: Optional[int] = None,
             return_spans: bool = False):
    """Drop packets, the way a bad link does.

    :param loss: fraction of packets to lose. Achieved exactly -- ``round(loss
        * packets)`` of them go, chosen without replacement -- so the number
        asked for is the number a corpus actually carries.
    :param packet_ms: packet length. 20 ms is G.711 over RTP.
    :param conceal: ``'silence'`` leaves a hole; ``'hold'`` repeats the last
        packet that arrived, which is the crudest concealment a decoder does
        and sounds very different from a hole.
    :param burst: mean run length, in packets. ``None`` (the default) loses
        them independently. Real links lose them in runs -- a fade, a queue
        overflowing, a handover -- and a run is far harder to conceal than
        the same number of packets scattered about, because there is no
        recent speech left to repeat.
    :param seed: which packets go, reproducibly.
    :param return_spans: also return the lost packets as ``(start, end)``
        seconds, one per packet, so an experiment can say where the damage
        was. Packets in a burst come back as neighbouring spans rather than
        one merged span; merging is a line of downstream code and unmerging
        is not.

    ``burst=1`` is not the default in disguise. Independent loss already
    averages :func:`independent_burst` -- ``1 / (1 - loss)`` -- because
    independent draws sometimes land side by side, so ``burst=1`` asks for
    runs strictly of one and is *less* clustered than leaving it alone.

    >>> lossy = kudio.dropouts(speech, sr, loss=0.05, seed=0)
    >>> bursty = kudio.dropouts(speech, sr, loss=0.05, burst=6, seed=0)
    >>> lossy, gaps = kudio.dropouts(speech, sr, loss=0.05, seed=0,
    ...                              return_spans=True)         # doctest: +SKIP
    """
    y = _mono("dropouts()", y)
    if sr <= 0:
        raise FeatureError(f"dropouts() needs a positive rate, got {sr}")
    if not 0.0 <= loss <= 1.0:
        raise FeatureError(f"loss must be a fraction in [0, 1], got {loss}")
    if packet_ms <= 0:
        raise FeatureError(f"packet_ms must be positive, got {packet_ms}")
    if conceal not in CONCEALMENT:
        raise FeatureError(
            f"unknown concealment {conceal!r}; choose from "
            f"{', '.join(CONCEALMENT)}")
    if burst is not None and burst < 1.0:
        raise FeatureError(
            f"burst is a run length in packets and cannot be under 1, got "
            f"{burst}. Independent loss is burst=None, and already runs to "
            f"{independent_burst(min(loss, 0.999)):.2f} packets at this rate.")

    size = max(1, int(round(packet_ms / 1000.0 * sr)))
    packets = int(np.ceil(len(y) / size))
    out = y.astype(np.float32, copy=True)
    spans: List[Tuple[float, float]] = []
    if packets == 0 or loss == 0.0:
        return (out, spans) if return_spans else out

    rng = np.random.default_rng(seed)
    n_lost = min(int(round(loss * packets)), packets)
    if n_lost == 0:
        return (out, spans) if return_spans else out
    if burst is None:
        lost = np.sort(rng.choice(packets, size=n_lost, replace=False))
    else:
        lost = _burst_runs(packets, n_lost, float(burst), rng)

    for index in lost:
        start, stop = index * size, min((index + 1) * size, len(y))
        if conceal == 'hold' and start > 0:
            # repeat the previous packet, tiled if this one runs short
            previous = out[max(0, start - size):start]
            if previous.size:
                out[start:stop] = np.resize(previous, stop - start)
            else:
                out[start:stop] = 0.0
        else:
            out[start:stop] = 0.0
        spans.append((start / sr, stop / sr))

    return (out, spans) if return_spans else out


def telephone(y: np.ndarray, sr: int, *, codec: str = 'mu_law',
              band: Tuple[float, float] = TELEPHONE_BAND) -> np.ndarray:
    """Everything a phone line does, in the order it does it.

    Band-limit to 300-3400 Hz, run at 8 kHz, compand to 8 bits, come back.
    The result is returned at the rate it went in and the same length, so it
    drops straight into a dataset beside its own source.

    :param codec: ``'mu_law'``, ``'a_law'`` or ``'none'`` for the band and the
        rate without the companding.

    The bandwidth is the part that survives being measured:
    ``kudio.audio_report`` reports about 3.4 kHz afterwards and flags the file
    as band-limited, which is exactly what it is.

    >>> kudio.telephone(speech, 16000)                       # doctest: +SKIP
    """
    y = _mono("telephone()", y)
    if sr <= 0:
        raise FeatureError(f"telephone() needs a positive rate, got {sr}")
    if codec not in ('mu_law', 'a_law', 'none'):
        raise FeatureError(
            f"unknown codec {codec!r}; choose mu_law, a_law or none")
    if y.size == 0:
        return y.astype(np.float32)

    from kudio.core.io import resample
    from kudio.effects.filters import bandpass

    low, high = band
    # Band-limit at the original rate first: the anti-alias filter of the
    # resampler is not the telephone's passband, and doing it afterwards would
    # measure the resampler rather than the line.
    narrow = bandpass(y.astype(np.float32), sr, low=low,
                      high=min(high, sr / 2 * 0.99))
    if sr != TELEPHONE_SR:
        narrow = resample(narrow, sr, TELEPHONE_SR)

    if codec == 'mu_law':
        narrow = mu_law(narrow)
    elif codec == 'a_law':
        narrow = a_law(narrow)

    if sr != TELEPHONE_SR:
        narrow = resample(narrow, TELEPHONE_SR, sr)
    # resampling twice rarely lands on the same sample count
    out = np.zeros(len(y), dtype=np.float32)
    keep = min(len(out), len(narrow))
    out[:keep] = narrow[:keep]
    return out


# ------------------------------------------------------- the chain, as a unit

#: What a channel can be asked for, and what it defaults to. One dictionary,
#: so a corpus builder, the command line and a GUI all describe a link the
#: same way instead of each inventing an argument order.
CHANNEL_DEFAULTS = {
    'band': False,          # band-limit to TELEPHONE_BAND and run at 8 kHz
    'codec': 'none',        # 'mu_law', 'a_law' or 'none'
    'bits': None,           # linear requantisation, after the codec
    'loss': 0.0,            # fraction of packets lost
    'burst': None,          # ...in runs of this many, or independently
    'packet_ms': PACKET_MS,
    'conceal': 'silence',
}

#: The named links, so the common cases are one word.
CHANNEL_PRESETS = {
    'none': {},
    'telephone': {'band': True, 'codec': 'mu_law'},
    'telephone_a': {'band': True, 'codec': 'a_law'},
    'mu_law': {'codec': 'mu_law'},
    'a_law': {'codec': 'a_law'},
    'voip': {'codec': 'mu_law', 'loss': 0.05, 'burst': 4.0,
             'conceal': 'hold'},
}


def channel_spec(spec) -> dict:
    """Normalise *spec* -- a preset name, a dict, or ``None`` -- to a dict.

    :param spec: ``None`` or ``'none'`` for no channel; one of
        :data:`CHANNEL_PRESETS` (``'telephone'``, ``'voip'``, ``'mu_law'``,
        ...); or a dict of any of :data:`CHANNEL_DEFAULTS`. A preset with
        overrides is a dict with a ``'preset'`` key.

    Returns ``{}`` for "no channel", which is falsy, so callers can write
    ``if spec:`` and mean it.

    >>> kudio.channel_spec('telephone')['codec']
    'mu_law'
    >>> kudio.channel_spec({'preset': 'voip', 'loss': 0.1})['loss']
    0.1
    """
    if spec is None:
        return {}
    if isinstance(spec, str):
        if spec not in CHANNEL_PRESETS:
            raise FeatureError(
                f"unknown channel {spec!r}; choose from "
                f"{', '.join(sorted(CHANNEL_PRESETS))}, or pass a dict")
        spec = dict(CHANNEL_PRESETS[spec])
    elif isinstance(spec, dict):
        spec = dict(spec)
        preset = spec.pop('preset', None)
        if preset is not None:
            base = channel_spec(preset)
            base.update(spec)
            spec = base
    else:
        raise FeatureError(
            f"channel must be a preset name, a dict or None, got "
            f"{type(spec).__name__}")

    unknown = set(spec) - set(CHANNEL_DEFAULTS)
    if unknown:
        raise FeatureError(
            f"unknown channel setting(s) {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(CHANNEL_DEFAULTS))}")

    out = dict(CHANNEL_DEFAULTS)
    out.update(spec)
    if out['codec'] not in ('mu_law', 'a_law', 'none'):
        raise FeatureError(
            f"unknown codec {out['codec']!r}; choose mu_law, a_law or none")
    if out['conceal'] not in CONCEALMENT:
        raise FeatureError(
            f"unknown concealment {out['conceal']!r}; choose from "
            f"{', '.join(CONCEALMENT)}")
    if not 0.0 <= float(out['loss']) <= 1.0:
        raise FeatureError(
            f"loss must be a fraction in [0, 1], got {out['loss']}")
    if out['bits'] is not None and int(out['bits']) < 2:
        raise FeatureError(f"bits must be >= 2, got {out['bits']}")

    # Nothing asked for is not a channel. Saying so here means one falsy
    # check downstream instead of five truthy ones.
    if (not out['band'] and out['codec'] == 'none'
            and out['bits'] is None and not float(out['loss'])):
        return {}
    return out


def channel_tag(spec) -> str:
    """A short, stable description of *spec*, for a manifest or a filename.

    Readable rather than exhaustive: it is what went on the corpus label, and
    a reader should be able to tell two runs apart at a glance.

    >>> kudio.channel_tag('telephone')
    'telephone'
    >>> kudio.channel_tag({'codec': 'mu_law', 'loss': 0.05, 'burst': 4})
    'mu_law+loss5%x4'
    """
    spec = channel_spec(spec)
    if not spec:
        return ''
    parts = []
    if spec['band']:
        parts.append('telephone' if spec['codec'] == 'mu_law'
                     else f"band{spec['codec']}" if spec['codec'] != 'none'
                     else 'band')
    elif spec['codec'] != 'none':
        parts.append(spec['codec'])
    if spec['bits'] is not None:
        parts.append(f"{int(spec['bits'])}bit")
    loss = float(spec['loss'])
    if loss:
        tag = f"loss{loss * 100:g}%"
        if spec['burst']:
            tag += f"x{float(spec['burst']):g}"
        if spec['conceal'] != 'silence':
            tag += f"-{spec['conceal']}"
        parts.append(tag)
    return '+'.join(parts)


def apply_channel(y: np.ndarray, sr: int, spec,
                  seed: Optional[int] = None) -> np.ndarray:
    """Run *y* through a whole link in the order a link does it.

    Band and rate first, then the codec, then the converter, then the wire --
    packets are lost from the encoded signal, not from the microphone, so the
    loss goes last. Same length and same rate as it went in.

    >>> kudio.apply_channel(mix, sr, 'telephone')          # doctest: +SKIP
    >>> kudio.apply_channel(mix, sr, {'preset': 'voip', 'loss': 0.1}, seed=7)
    ...                                                    # doctest: +SKIP
    """
    spec = channel_spec(spec)
    if not spec:
        return np.asarray(y, dtype=np.float32)

    out = _mono("apply_channel()", y).astype(np.float32)
    if spec['band']:
        out = telephone(out, sr, codec=spec['codec'])
    elif spec['codec'] == 'mu_law':
        out = mu_law(out)
    elif spec['codec'] == 'a_law':
        out = a_law(out)

    if spec['bits'] is not None:
        out = bit_depth(out, int(spec['bits']))
    if float(spec['loss']):
        out = dropouts(out, sr, loss=float(spec['loss']),
                       packet_ms=float(spec['packet_ms']),
                       conceal=spec['conceal'], burst=spec['burst'],
                       seed=seed)
    return out
