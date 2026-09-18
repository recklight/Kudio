# -*- coding: utf-8 -*-
"""The channel: companding, quantisation and packet loss.

Additive noise and a room can both be written down as arithmetic on the
signal. None of this can, which is why it needed its own module -- and why the
tests here are about textbook properties rather than about output shapes.
"""
from __future__ import annotations

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

from conftest import SR, make_speech


def snr(ref: np.ndarray, deg: np.ndarray) -> float:
    n = min(len(ref), len(deg))
    ref, deg = np.asarray(ref[:n], float), np.asarray(deg[:n], float)
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - deg) ** 2) + 1e-30))


def sine(freq: float = 997.0, sr: int = 48000, seconds: float = 1.0,
         amp: float = 0.999) -> np.ndarray:
    """Deliberately not a bin centre, so the error is not periodic with it."""
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ============================================================== companding

def test_eight_bit_companding_lands_where_telephony_says_it_does():
    assert snr(make_speech(), kudio.mu_law(make_speech())) == \
        pytest.approx(37.7, abs=2.0)
    assert snr(make_speech(), kudio.a_law(make_speech())) == \
        pytest.approx(38.1, abs=2.0)


def test_companding_keeps_its_signal_to_noise_ratio_as_the_level_falls():
    """This is the whole reason telephony compands, and the one property a
    wrong implementation would not have.

    Logarithmic steps follow the signal down; linear steps do not, and a
    quiet passage through linear 8-bit loses the textbook 6 dB per halving.
    """
    speech = make_speech()
    speech = speech / np.max(np.abs(speech))

    companded, linear = [], []
    for amp in (1.0, 0.5, 0.25, 0.125, 0.0625):
        quiet = (speech * amp).astype(np.float32)
        companded.append(snr(quiet, kudio.mu_law(quiet)))
        linear.append(snr(quiet, kudio.bit_depth(quiet, 8)))

    assert max(companded) - min(companded) < 3.0, companded
    # four halvings, six decibels each
    assert linear[0] - linear[-1] == pytest.approx(24.0, abs=3.0), linear
    assert companded[-1] > linear[-1] + 15.0


def test_fewer_bits_is_a_worse_link():
    speech = make_speech()
    scores = [snr(speech, kudio.mu_law(speech, bits=b)) for b in (4, 6, 8, 10)]
    assert scores == sorted(scores)
    assert scores[-1] - scores[0] > 20.0


@pytest.mark.parametrize("law", [kudio.mu_law, kudio.a_law])
def test_the_curve_is_odd_and_leaves_silence_alone(law):
    speech = make_speech()
    assert np.all(law(np.zeros(1000, dtype=np.float32)) == 0.0)
    assert np.allclose(law(-speech), -law(speech), atol=1e-6)


@pytest.mark.parametrize("law", [kudio.mu_law, kudio.a_law])
def test_the_curve_is_monotone(law):
    ramp = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    assert np.all(np.diff(law(ramp)) >= -1e-7)


def test_the_two_laws_are_not_the_same_law():
    speech = make_speech()
    assert not np.allclose(kudio.mu_law(speech), kudio.a_law(speech),
                           atol=1e-4)


def test_anything_past_full_scale_comes_back_at_full_scale():
    loud = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    for law in (kudio.mu_law, kudio.a_law):
        out = law(loud)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)
        assert out[0] == pytest.approx(out[1], abs=1e-6)
        assert out[-1] == pytest.approx(out[-2], abs=1e-6)


def test_agrees_with_the_reference_codec_where_it_is_available():
    """`audioop` is G.711 itself and was removed in Python 3.13, so it cannot
    be a dependency -- but it can be checked against, the way the loudness
    filters are checked against pyloudnorm.

    The curve here is the analytic one, not the standard's segment layout, so
    the two are close rather than identical. Half a decibel of round-trip SNR
    is the claim the docstring makes; this is where it comes from.
    """
    audioop = pytest.importorskip("audioop",
                                  reason="removed in Python 3.13")
    speech = make_speech()
    pcm = (np.clip(speech, -1, 1) * 32767).astype("<i2").tobytes()
    reference = np.frombuffer(
        audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2),
        dtype="<i2").astype(np.float64) / 32767.0

    mine = kudio.mu_law(speech)
    assert snr(speech, mine) == pytest.approx(snr(speech, reference), abs=0.5)
    assert np.max(np.abs(reference - mine)) < 0.02


# ============================================================ quantisation

@pytest.mark.parametrize("bits", [4, 6, 8, 10, 12, 16])
def test_quantisation_noise_is_six_decibels_a_bit(bits):
    """A full-scale sine gives 6.02 x bits + 1.76 dB. There is no fudge in
    that figure, so it is the right thing to assert.

    The tolerance is a decibel because the formula is asymptotic: at four bits
    there are seven magnitude steps, and "the error is uniform over one step"
    stops being quite true.
    """
    tone = sine()
    assert snr(tone, kudio.bit_depth(tone, bits)) == \
        pytest.approx(6.02 * bits + 1.76, abs=1.0)


def test_requantising_to_the_depth_it_already_has_barely_moves_it():
    tone = sine()
    assert snr(tone, kudio.bit_depth(tone, 16)) > 90.0


def test_dither_costs_a_little_noise_and_is_reproducible():
    tone = sine()
    plain = snr(tone, kudio.bit_depth(tone, 8))
    dithered = snr(tone, kudio.bit_depth(tone, 8, dither=True, seed=0))

    assert dithered < plain, "dither raises the noise floor -- that is the deal"
    assert plain - dithered < 4.0, "and not by much"
    assert np.array_equal(kudio.bit_depth(tone, 8, dither=True, seed=0),
                          kudio.bit_depth(tone, 8, dither=True, seed=0))


def test_quantisation_stays_inside_full_scale():
    for bits in (2, 8, 16):
        out = kudio.bit_depth(np.array([-1.0, 1.0], dtype=np.float32), bits)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)


# ============================================================= packet loss

def test_the_loss_rate_asked_for_is_the_loss_rate_delivered():
    """A corpus labelled 5% has to carry 5%, or the label is decoration."""
    speech = make_speech()
    packets = int(np.ceil(len(speech) / (PACKET := int(0.02 * SR))))
    for loss in (0.0, 0.01, 0.05, 0.2, 0.5):
        _, spans = kudio.dropouts(speech, SR, loss=loss, seed=0,
                                  return_spans=True)
        assert len(spans) == round(loss * packets), loss


def test_the_spans_say_where_the_holes_are():
    speech = make_speech()
    lossy, spans = kudio.dropouts(speech, SR, loss=0.1, seed=1,
                                  return_spans=True)
    assert spans
    for start, end in spans:
        piece = lossy[int(start * SR):int(end * SR)]
        assert np.all(piece == 0.0), (start, end)
        assert end - start == pytest.approx(0.02, abs=0.001)


def test_holding_the_last_packet_fills_the_hole():
    speech = make_speech()
    silent, spans = kudio.dropouts(speech, SR, loss=0.1, seed=2,
                                   conceal="silence", return_spans=True)
    held = kudio.dropouts(speech, SR, loss=0.1, seed=2, conceal="hold")

    start, end = spans[1]                       # not the first, which may be at 0
    hole = silent[int(start * SR):int(end * SR)]
    filled = held[int(start * SR):int(end * SR)]
    before = held[int(start * SR) - int(0.02 * SR):int(start * SR)]

    assert np.all(hole == 0.0)
    assert np.any(filled != 0.0)
    assert np.allclose(filled, before[:len(filled)], atol=1e-6)


def _harmonic(f0: float, sr: int = SR, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (sum(np.sin(2 * np.pi * f0 * k * t) / k
                for k in range(1, 10)) / 5).astype(np.float32)


@pytest.mark.parametrize("f0, cycles_per_packet", [
    (50.0, 1.0), (100.0, 2.0), (150.0, 3.0), (200.0, 4.0),
])
def test_holding_is_perfect_when_the_period_divides_the_packet(f0,
                                                               cycles_per_packet):
    """A 20 ms packet holds exactly three cycles of a 150 Hz voice, so the
    repeat lands in phase and the gap is filled with the right waveform."""
    assert 0.02 * f0 == pytest.approx(cycles_per_packet)
    tone = _harmonic(f0)
    held = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="hold")
    assert snr(tone, held) > 100.0


@pytest.mark.parametrize("f0", [120.0, 133.0, 175.0])
def test_and_worse_than_the_hole_when_it_does_not(f0):
    """Out of phase, the repeat is the right *kind* of sound at the wrong
    place, and as an error signal that is larger than the silence it
    replaced."""
    assert (0.02 * f0) % 1.0 > 0.2, "this pitch has to not divide the packet"
    tone = _harmonic(f0)
    silent = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="silence")
    held = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="hold")
    assert snr(tone, held) < snr(tone, silent)


def test_so_si_sdr_cannot_judge_a_concealment_strategy():
    """Stated once, on its own, because it was nearly asserted the other way.

    Whether holding scores better or worse than a hole is decided by whether
    the repeated packet lands in phase -- an accident of the talker's pitch
    against the packet length, and nothing to do with how the gap sounds.
    Anything choosing a concealment strategy on a waveform metric is measuring
    that accident.
    """
    in_phase = _harmonic(150.0)          # 3.00 cycles per 20 ms packet
    out_of_phase = _harmonic(133.0)      # 2.66

    def wins(tone):
        silent = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="silence")
        held = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="hold")
        return "hold" if snr(tone, held) > snr(tone, silent) else "hole"

    assert wins(in_phase) == "hold"
    assert wins(out_of_phase) == "hole"

    # what is true of both, and is what `hold` actually promises
    for tone in (in_phase, out_of_phase):
        silent = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="silence")
        held = kudio.dropouts(tone, SR, loss=0.1, seed=2, conceal="hold")
        assert np.count_nonzero(held == 0.0) < np.count_nonzero(silent == 0.0)


def test_losing_nothing_changes_nothing():
    speech = make_speech()
    assert np.array_equal(kudio.dropouts(speech, SR, loss=0.0), speech)


def test_the_same_seed_loses_the_same_packets():
    speech = make_speech()
    a = kudio.dropouts(speech, SR, loss=0.1, seed=5)
    b = kudio.dropouts(speech, SR, loss=0.1, seed=5)
    c = kudio.dropouts(speech, SR, loss=0.1, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_packet_length_changes_the_damage_not_its_amount():
    """A loss *rate* is a fraction of packets, so the total lost is the same
    whatever the packet size -- what changes is whether it arrives as many
    short holes or a few long ones, and those are very different to listen to
    and to conceal."""
    speech = make_speech()
    short, short_spans = kudio.dropouts(speech, SR, loss=0.1, packet_ms=10.0,
                                        seed=0, return_spans=True)
    long, long_spans = kudio.dropouts(speech, SR, loss=0.1, packet_ms=60.0,
                                      seed=0, return_spans=True)

    assert np.count_nonzero(short == 0) == pytest.approx(
        np.count_nonzero(long == 0), rel=0.02)
    assert len(short_spans) > len(long_spans) * 4
    assert (long_spans[0][1] - long_spans[0][0]) == pytest.approx(0.06, abs=0.002)


# =============================================================== telephone

def test_a_phone_line_band_limits_and_says_so():
    speech = make_speech()
    phone = kudio.telephone(speech, SR)

    before = kudio.audio_report(speech, SR)
    after = kudio.audio_report(phone, SR)
    assert before.bandwidth_hz > 6000
    assert 3000 < after.bandwidth_hz < 4200
    assert after.band_limited


def test_the_length_and_the_rate_survive():
    speech = make_speech()
    assert len(kudio.telephone(speech, SR)) == len(speech)
    assert len(kudio.telephone(speech[:1000], SR)) == 1000


def test_the_rumble_below_the_passband_is_gone():
    sr = 16000
    t = np.arange(sr * 2) / sr
    rumble = (0.5 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    assert np.max(np.abs(kudio.telephone(rumble, sr))) < 0.05


def test_the_codec_can_be_left_out_and_the_line_still_narrows():
    speech = make_speech()
    plain = kudio.telephone(speech, SR, codec="none")
    companded = kudio.telephone(speech, SR, codec="mu_law")

    assert kudio.audio_report(plain, SR).band_limited
    assert not np.allclose(plain, companded, atol=1e-4)
    assert snr(plain, companded) > 20.0, "the codec is a small change on top"


def test_a_law_is_offered_too():
    speech = make_speech()
    assert not np.allclose(kudio.telephone(speech, SR, codec="a_law"),
                           kudio.telephone(speech, SR, codec="mu_law"),
                           atol=1e-4)


def test_a_clip_already_at_the_phone_rate_is_not_resampled_twice():
    sr = 8000
    speech = kudio.resample(make_speech(), SR, sr)
    phone = kudio.telephone(speech, sr)
    assert len(phone) == len(speech)


# ============================================================ refused input

@pytest.mark.parametrize("call", [
    lambda: kudio.mu_law(np.zeros((100, 2), dtype=np.float32)),
    lambda: kudio.a_law(np.zeros((100, 2), dtype=np.float32)),
    lambda: kudio.bit_depth(np.zeros((100, 2), dtype=np.float32)),
    lambda: kudio.dropouts(np.zeros((100, 2), dtype=np.float32), SR),
    lambda: kudio.telephone(np.zeros((100, 2), dtype=np.float32), SR),
])
def test_stereo_is_refused_rather_than_flattened(call):
    with pytest.raises(FeatureError, match="mono"):
        call()


@pytest.mark.parametrize("call,match", [
    (lambda: kudio.mu_law(make_speech(), bits=1), "bits"),
    (lambda: kudio.bit_depth(make_speech(), 1), "bits"),
    (lambda: kudio.dropouts(make_speech(), SR, loss=1.5), "loss"),
    (lambda: kudio.dropouts(make_speech(), SR, loss=-0.1), "loss"),
    (lambda: kudio.dropouts(make_speech(), SR, packet_ms=0), "packet_ms"),
    (lambda: kudio.dropouts(make_speech(), SR, conceal="magic"), "concealment"),
    (lambda: kudio.dropouts(make_speech(), 0), "positive rate"),
    (lambda: kudio.telephone(make_speech(), SR, codec="opus"), "codec"),
    (lambda: kudio.telephone(make_speech(), 0), "positive rate"),
])
def test_impossible_settings_are_refused(call, match):
    with pytest.raises(FeatureError, match=match):
        call()


# ============================================================== burst loss

def runs_of(spans, packet_s: float):
    """Lost packets grouped into the runs a listener would hear as one gap."""
    if not spans:
        return []
    lengths, current = [], 1
    for before, after in zip(spans, spans[1:]):
        if after[0] - before[1] < packet_s * 1e-6:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def long_speech(minutes: float = 4.0):
    """Long enough that a mean run length is worth measuring at all.

    Six seconds is 300 packets and about 15 losses at 5%, which is two or
    three runs -- a sample far too small to say anything about their mean.
    """
    return np.tile(make_speech(), int(np.ceil(minutes * 10)))


def test_independent_loss_is_not_runs_of_one():
    """The obvious guess about the default, and it is wrong.

    Independent draws land side by side sometimes, so the default already
    produces runs -- ``1 / (1 - loss)`` of them on average. ``burst=1`` is
    therefore a request for something *less* clustered, not for the default,
    and that is the whole reason ``burst=None`` is spelled ``None``.
    """
    assert kudio.independent_burst(0.05) == pytest.approx(1.0526, abs=1e-3)
    assert kudio.independent_burst(0.2) == pytest.approx(1.25, abs=1e-3)

    y = long_speech()
    _, spans = kudio.dropouts(y, SR, loss=0.2, seed=5, return_spans=True)
    measured = np.mean(runs_of(spans, 0.02))
    assert measured == pytest.approx(kudio.independent_burst(0.2), abs=0.1)

    _, spans = kudio.dropouts(y, SR, loss=0.2, burst=1, seed=5,
                              return_spans=True)
    assert max(runs_of(spans, 0.02)) == 1
    assert np.mean(runs_of(spans, 0.02)) < measured


@pytest.mark.parametrize("burst", [2.0, 4.0, 8.0])
def test_bursts_come_out_the_length_they_were_asked_for(burst):
    y = long_speech()
    _, spans = kudio.dropouts(y, SR, loss=0.05, burst=burst, seed=1,
                              return_spans=True)
    lengths = runs_of(spans, 0.02)
    assert np.mean(lengths) == pytest.approx(burst, rel=0.25)
    # geometric, so the tail should be long rather than everything at the mean
    assert max(lengths) > burst


@pytest.mark.parametrize("burst", [None, 1.0, 3.0, 10.0])
def test_the_loss_rate_is_the_loss_rate_however_it_is_clustered(burst):
    """Gilbert's chain fixes the rate and delivers a different one per file.
    This does not: 5% asked for is 5% carried, bursts or no bursts, because
    the number on a corpus label should be true of the corpus.
    """
    y = long_speech()
    packets = int(np.ceil(len(y) / (0.02 * SR)))
    _, spans = kudio.dropouts(y, SR, loss=0.05, burst=burst, seed=2,
                              return_spans=True)
    assert len(spans) == round(0.05 * packets)


def test_a_burst_is_one_long_gap_rather_than_many_short_ones():
    """What makes burst loss hard, stated as something measurable: the same
    number of packets, arranged so that the longest hole is far longer."""
    y = long_speech()
    _, scattered = kudio.dropouts(y, SR, loss=0.05, seed=3, return_spans=True)
    _, bursty = kudio.dropouts(y, SR, loss=0.05, burst=8, seed=3,
                               return_spans=True)
    assert len(scattered) == len(bursty)
    assert max(runs_of(bursty, 0.02)) > 4 * max(runs_of(scattered, 0.02))


def test_holding_cannot_help_across_a_long_burst():
    """Repeating the last good packet is a 20 ms idea. Across a run it tiles
    the same 20 ms over and over, so the longer the run the less of the
    result is anything the talker did.
    """
    y = long_speech()
    short = kudio.dropouts(y, SR, loss=0.05, burst=1, conceal="hold", seed=4)
    long = kudio.dropouts(y, SR, loss=0.05, burst=16, conceal="hold", seed=4)
    assert snr(y, long) < snr(y, short)


def test_seeding_a_burst_reproduces_it():
    y = long_speech(1.0)
    first = kudio.dropouts(y, SR, loss=0.05, burst=5, seed=11)
    again = kudio.dropouts(y, SR, loss=0.05, burst=5, seed=11)
    other = kudio.dropouts(y, SR, loss=0.05, burst=5, seed=12)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_losing_most_of_it_in_short_runs_is_refused_rather_than_fudged():
    """80% of packets lost in runs of one needs four survivors for every
    loss, and there are not four. Silently merging the runs would mean the
    burst length asked for was not the burst length delivered.
    """
    with pytest.raises(FeatureError, match="surviving packets"):
        kudio.dropouts(long_speech(1.0), SR, loss=0.8, burst=1, seed=0)


def test_a_burst_shorter_than_a_packet_is_refused():
    with pytest.raises(FeatureError, match="cannot be under 1"):
        kudio.dropouts(make_speech(), SR, loss=0.05, burst=0.5)


# ========================================================= the link as a unit

def test_a_preset_is_the_dict_it_stands_for():
    assert kudio.channel_spec("telephone")["band"] is True
    assert kudio.channel_spec("telephone")["codec"] == "mu_law"
    assert kudio.channel_spec("voip")["burst"] == 4.0
    # every preset names only settings that exist
    for name in kudio.CHANNEL_PRESETS:
        assert set(kudio.channel_spec(name)) <= set(kudio.CHANNEL_DEFAULTS)


def test_nothing_asked_for_is_falsy_rather_than_a_dict_of_defaults():
    """So that one ``if spec:`` downstream means "is there a channel at all",
    instead of five checks that have to agree with each other."""
    assert kudio.channel_spec(None) == {}
    assert kudio.channel_spec("none") == {}
    assert kudio.channel_spec({"loss": 0.0, "bits": None}) == {}
    assert kudio.channel_spec({"loss": 0.01})


def test_a_preset_can_be_overridden_without_restating_it():
    spec = kudio.channel_spec({"preset": "voip", "loss": 0.1})
    assert spec["loss"] == 0.1
    assert spec["conceal"] == "hold"      # still voip's
    assert spec["burst"] == 4.0


@pytest.mark.parametrize("spec,match", [
    ("opus", "unknown channel"),
    ({"looss": 0.1}, "unknown channel setting"),
    ({"codec": "opus"}, "codec"),
    ({"loss": 2.0}, "loss"),
    ({"bits": 1}, "bits"),
    ({"conceal": "guess"}, "concealment"),
    (7, "preset name"),
])
def test_an_impossible_link_is_refused_when_it_is_described(spec, match):
    with pytest.raises(FeatureError, match=match):
        kudio.channel_spec(spec)


def test_the_tag_says_what_happened_and_stays_short():
    assert kudio.channel_tag("telephone") == "telephone"
    assert kudio.channel_tag(None) == ""
    assert kudio.channel_tag({"codec": "mu_law", "loss": 0.05,
                              "burst": 4}) == "mu_law+loss5%x4"
    assert kudio.channel_tag({"bits": 8}) == "8bit"
    # and it is stable: the same request always writes the same label
    assert kudio.channel_tag("voip") == kudio.channel_tag(
        kudio.channel_spec("voip"))


def test_the_link_runs_in_the_order_a_link_does():
    """Packets are lost from the encoded signal, so the loss is last. If it
    ran first the resampling inside ``telephone`` would smear every hole into
    its neighbours and the gaps would no longer be 20 ms long.
    """
    y = make_speech()
    out = kudio.apply_channel(y, SR, {"preset": "telephone", "loss": 0.1},
                              seed=0)
    assert len(out) == len(y)
    silent = np.flatnonzero(out == 0.0)
    assert silent.size > 0
    # the holes are whole packets, not smeared fragments
    breaks = np.flatnonzero(np.diff(silent) > 1)
    runs = np.diff(np.concatenate(([-1], breaks, [silent.size - 1])))
    assert np.all(runs % int(0.02 * SR) == 0)


def test_no_link_is_a_pass_through_that_still_returns_float32():
    y = make_speech()
    out = kudio.apply_channel(y, SR, None)
    assert np.array_equal(out, y) and out.dtype == np.float32


def test_the_same_link_and_seed_is_the_same_file():
    y = make_speech()
    spec = {"preset": "voip"}
    assert np.array_equal(kudio.apply_channel(y, SR, spec, seed=3),
                          kudio.apply_channel(y, SR, spec, seed=3))
    assert not np.array_equal(kudio.apply_channel(y, SR, spec, seed=3),
                              kudio.apply_channel(y, SR, spec, seed=4))
