# -*- coding: utf-8 -*-
"""What the transmission did to it: companding, quantisation, packet loss.

Speech arrives degraded in three ways. Noise is **additive** --
:class:`kudio.Synthesizer` mixes it at a chosen SNR. A room is
**convolutive** -- :func:`kudio.apply_rir` smears the signal across time. The
third is the channel: a codec quantises, a link drops packets, a converter
throws bits away. The telephone band is a filter, but the codec and the lost
packets cannot be written as ``y + n`` or ``y * h``, and they are what this
module adds.

>>> phone = kudio.telephone(speech, sr)          # 300-3400 Hz, 8 kHz, G.711
>>> lossy = kudio.dropouts(speech, sr, loss=0.05, seed=0)
>>> rough = kudio.bit_depth(speech, bits=8)

This is the augmentation that matters for anything that will meet a telephone
line or a conferencing codec, and it is well established for exactly that
(T.-L. Vu, Z. Zeng, H. Xu and E.-S. Chng, "Audio codec simulation based data
augmentation for telephony speech recognition", Proc. APSIPA ASC 2019,
pp. 198-203, doi:10.1109/APSIPAASC47483.2019.9023257; M. P.
Fernández-Gallego and D. T. Toledano, "A study of data augmentation for ASR
robustness in low bit rate contact center recordings including packet
losses", Applied Sciences 12(3), 1580, 2022, doi:10.3390/app12031580).

**Companding is the interesting one.** :func:`mu_law` spends its 8 bits
logarithmically, so the quantisation noise follows the signal down instead of
sitting at a fixed floor: on this repo's fixture it holds ~37 dB of SNR over a
24 dB range of levels, where linear 8-bit loses the textbook 6 dB per halving.
That is the whole reason telephony uses it, and it is what the tests check.

**These are models, not bit-exact codecs.** :func:`mu_law` and :func:`a_law`
implement the continuous mu-law and A-law *curves* quantised to 8 bits, not
the piecewise-linear segment tables G.711 defines them by. Measured against
Python's own ``audioop``, which implements those tables (and was removed in
Python 3.13, so it cannot be a dependency), the round-trip SNR comes out
0.26 to 0.97 dB higher than audioop's over a 24 dB range of levels, and
single samples differ by a few hundredths of full scale at most. Those are
the clip and levels of the companding table in the degradation guide, with
the samples rounded to int16 before audioop, as :func:`kudio.save_wave`
rounds them. For augmentation that is the same thing; for a bit-exact
decoder it is not, and this does not claim to be one.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from kudio._numbers import flag, generator, real, whole
from kudio._samples import finite, one_channel
from kudio.core._policy import sample_rate
from kudio.exceptions import FeatureError

__all__ = ['mu_law', 'a_law', 'bit_depth', 'dropouts', 'telephone',
           'independent_burst', 'channel_spec', 'channel_tag', 'apply_channel',
           'MU_LAW_MU', 'A_LAW_A', 'PACKET_MS', 'TELEPHONE_BAND',
           'CHANNEL_DEFAULTS', 'CHANNEL_PRESETS']

log = logging.getLogger(__name__)

#: mu of the continuous mu-law curve, the law of North America and Japan.
#: G.711 defines the law by an 8-segment table that approximates it.
MU_LAW_MU = 255.0
#: A of the continuous A-law curve, the law of Europe; G.711's segment table
#: approximates it.
A_LAW_A = 87.6

#: Default packet length, in milliseconds. 20 ms is what G.711 over RTP uses,
#: so a lost packet is 20 ms of nothing.
PACKET_MS = 20.0

#: The passband a telephone line gives you, in Hz, and the rate it runs at.
TELEPHONE_BAND = (300.0, 3400.0)
#: The rate a telephone line runs at, in Hz.
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
    rate = real(loss)
    if rate is None or not 0.0 <= rate < 1.0:
        raise FeatureError(f"loss must be a fraction in [0, 1), got {loss!r}")
    return 1.0 / (1.0 - rate)


def _packets(n: int, sr: int, packet_ms: float, loss: float) -> Tuple[int, int, int]:
    """``(size, packets, lost)`` of a clip of *n* samples: the samples in a
    packet, the packets, and how many of them go."""
    size = max(1, int(round(packet_ms / 1000.0 * sr)))
    packets = int(np.ceil(n / size))
    return size, packets, min(int(round(loss * packets)), packets)


def _run_lengths(packets: int, n_lost: int, burst: float,
                 rng: np.random.Generator) -> List[int]:
    """How many packets go in each run, *n_lost* altogether, or a FeatureError
    when the packets that survive are too few to keep the runs apart."""
    runs: List[int] = []
    total = 0
    probability = 1.0 / burst
    while total < n_lost:
        length = int(rng.geometric(probability)) if probability < 1.0 else 1
        if length < 1:                       # more than an integer holds, to numpy 1.24
            length = n_lost
        length = min(length, n_lost - total)
        runs.append(length)
        total += length

    count = len(runs)
    survivors = packets - n_lost
    if survivors < count - 1:
        raise FeatureError(
            f"loss={n_lost / packets:.3f} in runs of {burst:g} packet(s) "
            f"needs about {count - 1} surviving packets to keep the runs "
            f"apart and only {survivors} survive. Either lose less or lose it "
            f"in longer runs.")
    return runs


def _check_loss(n: int, sr: int, spec: dict, seed: Optional[int]) -> None:
    """The FeatureError, if any, that :func:`dropouts` ends in on a clip of *n*
    samples. It depends on nothing but how many packets there are and on
    *seed*, so no audio is needed to find it."""
    loss = _loss(spec['loss'], 'dropouts')
    burst = _burst(spec['burst'], loss, 'dropouts')
    _, packets, n_lost = _packets(n, sr, float(spec['packet_ms']), loss)
    # there are never more runs than packets lost
    if burst is not None and packets - n_lost < n_lost - 1:
        _run_lengths(packets, n_lost, burst, np.random.default_rng(seed))


def _burst_runs(packets: int, n_lost: int, burst: float,
                rng: np.random.Generator) -> np.ndarray:
    """Which packets go, lost in runs averaging *burst* packets.

    Gilbert's two-state chain (E. N. Gilbert, "Capacity of a burst-noise
    channel", Bell Syst. Tech. J. 39(5), pp. 1253-1265, 1960,
    doi:10.1002/j.1538-7305.1960.tb03959.x), with every packet in its bad
    state lost, is the standard model and it fixes the *rate*, not the count
    -- ask it for 5% and a given file carries 4.1% or 6.3%.
    :func:`dropouts` promises the count, because a corpus labelled 5% should
    be 5%, so what is taken from Gilbert is the part that matters: the run
    lengths are geometric with mean *burst*, exactly as its bad state gives,
    and the total is then conditioned to be the number asked for.

    Runs are kept apart by at least one surviving packet. Without that two
    runs could abut and the realised mean would drift above the one
    requested, which would make *burst* unmeasurable and therefore not worth
    having as a parameter.
    """
    runs = _run_lengths(packets, n_lost, burst, rng)
    count = len(runs)
    free = packets - n_lost - (count - 1)

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
    y = one_channel(y, name, FeatureError)
    if y.ndim > 1:
        raise FeatureError(
            f"{name}() takes mono audio, got shape {y.shape}. A channel degrades "
            f"each channel differently and mixing down first is a decision, "
            f"not a detail.")
    return y.astype(np.float64, copy=False).reshape(-1)


#: The most bits a code word is taken to have. float64 has a step of 2**-52 at
#: full scale, so nothing past it is quantised any more, and 2**bits has to
#: stay a number.
_MAX_BITS = 64


def _bits(bits, name: str) -> int:
    """*bits* as an int, or a FeatureError. 8.5 bits gives 361 levels, which is
    no code word's count."""
    value = whole(bits)
    if value is None or not 2 <= value <= _MAX_BITS:
        raise FeatureError(f"{name}(): bits must be >= 2, a whole number of at most "
                           f"{_MAX_BITS}, got {bits!r}")
    return value


def _loss(loss, name: str) -> float:
    value = real(loss)
    if value is None or not 0.0 <= value <= 1.0:
        raise FeatureError(f"{name}(): loss must be a fraction in [0, 1], got {loss!r}")
    return value


def _packet_ms(packet_ms, name: str) -> float:
    value = real(packet_ms)
    if value is None or value <= 0:
        raise FeatureError(f"{name}(): packet_ms must be a finite number > 0, got "
                           f"{packet_ms!r}")
    return value


def _burst(burst, loss: float, name: str) -> Optional[float]:
    value = None if burst is None else real(burst)
    if burst is not None and (value is None or value < 1.0):
        raise FeatureError(
            f"{name}(): burst is a mean run length in packets, a finite number that "
            f"cannot be under 1, got {burst!r}. Independent loss is burst=None, and "
            f"already runs to {independent_burst(min(loss, 0.999)):.2f} packets at this "
            f"rate.")
    return value


def _quantise(values: np.ndarray, bits: int) -> np.ndarray:
    """Round *values* in ``[-1, 1]`` onto a sign-and-magnitude grid.

    Sign kept separately, magnitude quantised to ``2**(bits-1) - 1`` steps
    above zero. At 8 bits that is how G.711's mu-law is laid out, magnitudes
    0 to 127 with 0 decoding to exactly zero; G.711's A-law has no zero code,
    its smallest output being half its first step. The code at zero matters
    here: spreading ``2**bits`` levels evenly across ``[-1, 1]`` leaves none
    there, so digital silence would come back as a small DC offset and the
    curve would stop being odd.
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
    y = finite(_mono("mu_law", y), "mu_law", FeatureError)
    bits = _bits(bits, "mu_law")
    y = np.clip(y, -1.0, 1.0)

    compressed = np.sign(y) * np.log1p(MU_LAW_MU * np.abs(y)) \
        / np.log1p(MU_LAW_MU)
    coded = _quantise(compressed, bits)
    expanded = np.sign(coded) * ((1.0 + MU_LAW_MU) ** np.abs(coded) - 1.0) \
        / MU_LAW_MU
    return expanded.astype(np.float32)


def a_law(y: np.ndarray, *, bits: int = 8) -> np.ndarray:
    """The same, through the A-law curve -- G.711 as Europe specifies it.

    A-law is coarser than mu-law near silence, its first step twice mu-law's
    in G.711's tables and 2.8 times on these curves (4.9e-4 of full scale
    against 1.75e-4), and very slightly finer at the top (4.22% against
    4.29%). The difference is audible only on very quiet material; it is here
    because a corpus recorded off a European line went through this one.
    """
    y = finite(_mono("a_law", y), "a_law", FeatureError)
    bits = _bits(bits, "a_law")
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

    :param dither: add triangular (TPDF) dither two steps wide, peak to
        peak, before rounding. It raises the noise floor and makes the error's
        mean and power independent of the signal, which is what stops quiet
        passages sounding granular rather than merely noisy. Pass *seed* with
        it, or every call dithers differently.

    A full-scale sine comes back at about ``6.02 * bits + 1.76`` dB of SNR,
    which is the textbook figure and what the tests assert.
    """
    dither = flag(dither, 'dither', 'bit_depth', FeatureError)
    y = finite(_mono("bit_depth", y), "bit_depth", FeatureError)
    bits = _bits(bits, "bit_depth")
    y = np.clip(y, -1.0, 1.0)

    if dither:
        # the grid's own step: _quantise has 2**(bits-1) - 1 levels each side
        # of zero, and 2 / 2**bits is half of that at 2 bits
        step = 1.0 / (2 ** (bits - 1) - 1)
        rng = generator(seed, 'bit_depth', FeatureError)
        # triangular: the difference of two uniforms of one step each, two
        # steps peak to peak -- the width that takes the signal out of the
        # error's mean and power; one step only loosened the tie
        y = y + (rng.random(len(y)) - rng.random(len(y))) * step
        y = np.clip(y, -1.0, 1.0)
    return _quantise(y, bits).astype(np.float32)


def dropouts(y: np.ndarray, sr: int, *, loss: float = 0.05,
             packet_ms: float = PACKET_MS, conceal: str = 'silence',
             burst: Optional[float] = None, seed: Optional[int] = None,
             return_spans: bool = False
             ) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[float, float]]]]:
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
    return_spans = flag(return_spans, 'return_spans', 'dropouts', FeatureError)
    y = _mono("dropouts", y)
    sr = sample_rate(sr, "dropouts", FeatureError)
    loss = _loss(loss, "dropouts")
    packet_ms = _packet_ms(packet_ms, "dropouts")
    if conceal not in CONCEALMENT:
        raise FeatureError(
            f"unknown concealment {conceal!r}; choose from "
            f"{', '.join(CONCEALMENT)}")
    burst = _burst(burst, loss, "dropouts")

    size, packets, n_lost = _packets(len(y), sr, packet_ms, loss)
    out = y.astype(np.float32, copy=True)
    spans: List[Tuple[float, float]] = []
    if n_lost == 0:
        return (out, spans) if return_spans else out

    rng = generator(seed, 'dropouts', FeatureError)
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
    ``kudio.audio_report`` reports content up to about 3.8 kHz afterwards --
    the 3.4 kHz band edge plus the filter's roll-off -- and flags the file as
    band-limited, which is exactly what it is.

    >>> kudio.telephone(speech, 16000)                       # doctest: +SKIP
    """
    y = _mono("telephone", y)
    sr = sample_rate(sr, "telephone", FeatureError)
    if codec not in ('mu_law', 'a_law', 'none'):
        raise FeatureError(
            f"unknown codec {codec!r}; choose mu_law, a_law or none")
    if y.size == 0:
        return y.astype(np.float32)

    from kudio.core.io import resample
    from kudio.effects.filters import bandpass

    try:
        low, high = band
    except (TypeError, ValueError):
        low = high = None
    if real(low) is None or real(high) is None:
        raise FeatureError(f"telephone(): band is a (low, high) pair in Hz, got {band!r}")
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


def channel_spec(spec: Optional[Union[str, dict]]) -> dict:
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
    if not (isinstance(out['band'], (bool, np.bool_)) or out['band'] in (None, 0, 1)):
        # any text is true, and 'no' would be the telephone band
        raise FeatureError(f"channel_spec(): band is True for the telephone band or False, got "
                           f"{out['band']!r}")
    if out['conceal'] not in CONCEALMENT:
        raise FeatureError(
            f"unknown concealment {out['conceal']!r}; choose from "
            f"{', '.join(CONCEALMENT)}")
    # all of it here rather than on the first file, so Synthesizer(channel=...)
    # refuses a bad link when it is built
    loss = _loss(out['loss'], 'channel_spec')
    if out['bits'] is not None:
        _bits(out['bits'], 'channel_spec')
    _burst(out['burst'], loss, 'channel_spec')
    _packet_ms(out['packet_ms'], 'channel_spec')

    # Nothing asked for is not a channel. Saying so here means one falsy
    # check downstream instead of five truthy ones.
    if (not out['band'] and out['codec'] == 'none'
            and out['bits'] is None and not float(out['loss'])):
        return {}
    return out


def channel_tag(spec: Optional[Union[str, dict]]) -> str:
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
        # named as the presets are: 'banda_law' read as one made-up word
        parts.append({'mu_law': 'telephone', 'a_law': 'telephone_a'}.get(
            spec['codec'], 'band'))
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


def apply_channel(y: np.ndarray, sr: int, spec: Optional[Union[str, dict]],
                  seed: Optional[int] = None) -> np.ndarray:
    """Run *y* through a whole link in the order a link does it.

    Band and rate first, then the codec, then the converter, then the wire --
    packets are lost from the encoded signal, not from the microphone, so the
    loss goes last. Same length and same rate as it went in.

    >>> kudio.apply_channel(mix, sr, 'telephone')          # doctest: +SKIP
    >>> kudio.apply_channel(mix, sr, {'preset': 'voip', 'loss': 0.1}, seed=7)
    ...                                                    # doctest: +SKIP
    """
    sr = sample_rate(sr, 'apply_channel', FeatureError)
    spec = channel_spec(spec)
    out = _mono("apply_channel", y).astype(np.float32)
    if not spec:
        return out
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
