# -*- coding: utf-8 -*-
"""Voice activity detection."""
import inspect
import re

import numpy as np
import pytest

import kudio
from kudio.exceptions import FeatureError

SR = 16000


def test_finds_the_two_bursts(speech):
    spans = kudio.vad(speech, SR)
    assert len(spans) == 2
    (a0, a1), (b0, b1) = spans
    # the bursts sit at 1.0-2.0 and 3.5-4.8; padding widens each by 0.05
    assert a0 == pytest.approx(0.95, abs=0.12)
    assert a1 == pytest.approx(2.05, abs=0.12)
    assert b0 == pytest.approx(3.45, abs=0.12)
    assert b1 == pytest.approx(4.85, abs=0.12)


def test_spans_are_ordered_and_disjoint(speech):
    spans = kudio.vad(speech, SR)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start


def test_noise_alone_is_not_speech():
    rng = np.random.default_rng(1)
    noise = (0.01 * rng.standard_normal(SR * 4)).astype(np.float32)
    assert kudio.vad(noise, SR) == []


def test_digital_silence_is_not_speech():
    assert kudio.vad(np.zeros(SR, dtype=np.float32), SR) == []


def test_empty_input_returns_nothing():
    assert kudio.vad(np.zeros(0, dtype=np.float32), SR) == []


def test_adapts_to_the_noise_floor(speech):
    """The same speech under ten times the noise is still found."""
    rng = np.random.default_rng(2)
    noisier = speech + (0.05 * rng.standard_normal(len(speech))).astype(np.float32)
    assert len(kudio.vad(noisier, SR)) >= 1


def test_min_speech_drops_short_blips():
    rng = np.random.default_rng(3)
    y = (0.005 * rng.standard_normal(SR * 3)).astype(np.float32)
    y[SR:SR + 320] += 0.5          # a 20 ms click
    assert kudio.vad(y, SR, min_speech=0.5) == []


def test_min_silence_bridges_a_short_gap(speech):
    """A pause shorter than min_silence must not split a segment."""
    y = speech.copy()
    y[int(1.4 * SR):int(1.5 * SR)] *= 0.0        # 100 ms hole inside burst one
    joined = kudio.vad(y, SR, min_silence=0.3)
    split = kudio.vad(y, SR, min_silence=0.02, pad=0.0)
    assert len(joined) < len(split)


def test_flatness_test_rejects_a_loud_noise_burst():
    """Loud broadband noise passes the level gate; it must fail on flatness."""
    rng = np.random.default_rng(4)
    y = (0.005 * rng.standard_normal(SR * 4)).astype(np.float32)
    y[SR:2 * SR] += (0.3 * rng.standard_normal(SR)).astype(np.float32)
    assert kudio.vad(y, SR) == []
    # ...and relaxing the flatness test finds it, so it *was* loud enough
    assert kudio.vad(y, SR, flatness_max=1.0) != []


def test_vad_split_returns_the_audio(speech):
    spans = kudio.vad(speech, SR)
    segments = kudio.vad_split(speech, SR)
    assert len(segments) == len(spans)
    for segment, (start, end) in zip(segments, spans):
        assert len(segment) == pytest.approx((end - start) * SR, abs=2)


def test_vad_trim_keeps_the_middle(speech):
    trimmed, (start, end) = kudio.vad_trim(speech, SR)
    assert len(trimmed) == end - start
    assert start > 0
    assert end < len(speech)


def test_vad_trim_returns_the_clip_when_nothing_is_found():
    """Deleting the audio is never the safer reading of 'no speech'."""
    noise = (0.01 * np.random.default_rng(5).standard_normal(SR)).astype(np.float32)
    trimmed, (start, end) = kudio.vad_trim(noise, SR)
    assert (start, end) == (0, len(noise))
    assert len(trimmed) == len(noise)


def test_speech_ratio_is_a_fraction(speech):
    ratio = kudio.speech_ratio(speech, SR)
    assert 0.0 < ratio < 1.0
    assert kudio.speech_ratio(np.zeros(SR, dtype=np.float32), SR) == 0.0
    assert kudio.speech_ratio(np.zeros(0, dtype=np.float32), SR) == 0.0


def test_rejects_stereo_and_bad_rates(speech):
    with pytest.raises(FeatureError):
        kudio.vad(np.stack([speech, speech], axis=1), SR)
    with pytest.raises(FeatureError):
        kudio.vad(speech, 0)


def test_clip_shorter_than_one_frame_does_not_crash():
    assert isinstance(kudio.vad(np.zeros(100, dtype=np.float32), SR), list)


# -- one decision per spectrogram frame -------------------------------------------

@pytest.mark.parametrize("hop", [128, 256, 160])
def test_vad_frames_line_up_with_the_spectrogram(speech, hop):
    mask = kudio.vad_frames(speech, SR, hop_length=hop)
    spec = kudio.waveform_to_spectrogram(speech, hop_length=hop)
    assert mask.dtype == bool and mask.shape == (spec.shape[0],)


def test_vad_frames_uncentred_matches_librosa(speech):
    import librosa
    mask = kudio.vad_frames(speech, SR, hop_length=256, n_fft=512, center=False)
    frames = librosa.stft(speech, n_fft=512, hop_length=256, center=False)
    assert len(mask) == frames.shape[1]


def test_vad_frames_mark_the_bursts(speech):
    mask = kudio.vad_frames(speech, SR, hop_length=160)
    t = np.arange(len(mask)) * 160 / SR
    assert mask[(t > 1.15) & (t < 1.85)].all()
    assert mask[(t > 3.65) & (t < 4.65)].all()
    assert not mask[t < 0.8].any() and not mask[(t > 2.3) & (t < 3.2)].any()


def test_vad_frames_do_not_pad_unless_asked(speech):
    """Padding protects a cut; on a frame label it calls silence speech."""
    plain = kudio.vad_frames(speech, SR)
    padded = kudio.vad_frames(speech, SR, pad=0.05)
    assert padded.sum() > plain.sum()
    assert not (plain & ~padded).any()


# -- the docstring the docs build reads -------------------------------------------

#: A line that is nothing but a bracketed phrase, which is markdown's
#: shortcut-reference syntax: mkdocstrings hands it to autorefs as an
#: identifier to resolve, and the docs build stops when it cannot.
_SHORTCUT_REFERENCE = re.compile(r'^\[.+\]$')


def _markdown_lines(doc):
    """The docstring lines the docs build parses as prose rather than code."""
    inside_a_fence = False
    for line in doc.splitlines():
        if line.lstrip().startswith('```'):
            inside_a_fence = not inside_a_fence
            continue
        if not inside_a_fence:
            yield line


def test_the_example_output_is_fenced_not_a_link():
    """vad()'s example prints a list of tuples, and the docs choke on it.

    ``[(0.31, 1.84), (2.42, 3.97)]`` alone on a line is a link to markdown,
    so building the docs tries to resolve it as a python identifier. Fencing
    the example keeps it literal. Escaping the bracket would please the docs
    build and leave a stray backslash in ``help(kudio.vad)``, which is why
    the backslash is checked here too.
    """
    doc = inspect.getdoc(kudio.vad)
    assert '[(0.31, 1.84), (2.42, 3.97)]' in doc, "the example went missing"
    assert '\\' not in doc, "an escape leaked into the docstring"
    offenders = [line for line in _markdown_lines(doc)
                 if _SHORTCUT_REFERENCE.match(line.strip())]
    assert not offenders, offenders


# -- frame counts for any n_fft, and corrupt clips -----------------------------
@pytest.mark.parametrize("n_fft", [512, 511, 400])
def test_vad_frames_has_one_label_per_spectrogram_row(n_fft):
    y = (0.1 * np.random.default_rng(0).standard_normal(32000)).astype(np.float32)
    spec = kudio.waveform_to_spectrogram(y, n_fft=n_fft, hop_length=256)
    assert len(kudio.vad_frames(y, 16000, hop_length=256, n_fft=n_fft)) == len(spec)


def test_a_corrupt_clip_is_refused():
    y = 0.1 * np.random.default_rng(0).standard_normal(16000)
    y[100] = np.nan
    with pytest.raises(kudio.FeatureError, match="not finite"):
        kudio.vad(y, 16000)


# -- judging the frames on a band ----------------------------------------------
def _under_an_ultrasonic_floor(seed=1):
    """make_speech at a tenth of its level in a 192 kHz file, over a floor
    above 20 kHz that holds most of each frame's energy."""
    from conftest import make_speech
    voice = kudio.resample(make_speech(), SR, 192000).astype(np.float64)
    floor = kudio.highpass(np.random.default_rng(seed).standard_normal(len(voice)),
                           192000, cutoff=20000)
    return 0.1 * voice + 0.02 * floor, 0.02 * floor


def test_a_band_finds_speech_that_an_ultrasonic_floor_hides(speech):
    y, _ = _under_an_ultrasonic_floor()
    assert kudio.vad(y, 192000) == [], "the floor outweighs the voice"
    found = kudio.vad(y, 192000, band=(100, 4000))
    expected = kudio.vad(speech, SR)
    assert len(found) == len(expected) == 2
    assert np.ravel(found) == pytest.approx(np.ravel(expected), abs=0.02)


def test_a_band_does_not_find_speech_in_the_floor_alone():
    _, floor = _under_an_ultrasonic_floor()
    white = 0.01 * np.random.default_rng(2).standard_normal(len(floor))
    assert kudio.vad(floor + white, 192000, band=(100, 4000)) == []


@pytest.mark.parametrize("band", [(0, 4000), (100, 8000), (100, 4000)])
def test_a_band_on_a_16_khz_clip_finds_the_same_bursts(speech, band):
    found = kudio.vad(speech, SR, band=band)
    assert np.ravel(found) == pytest.approx(np.ravel(kudio.vad(speech, SR)), abs=0.02)


@pytest.mark.parametrize("band", [(4000, 100), (-1, 100), (9000, 10000), (1005, 1020)])
def test_a_band_that_is_not_one_is_refused(speech, band):
    """(1005, 1020) holds no bin of a 30 ms frame, 33.3 Hz apart, and came back
    as no speech with a numpy warning about an empty mean."""
    with pytest.raises(FeatureError, match="band"):
        kudio.vad(speech, SR, band=band)


# -- short clips ------------------------------------------------------------------
def _hiss(seconds, sr=SR, seed=7):
    return 0.001 * np.random.default_rng(seed).standard_normal(int(round(seconds * sr)))


def _word(seconds, sr=SR):
    t = np.arange(int(round(seconds * sr))) / sr
    return 0.4 * np.sin(2 * np.pi * 220 * t + 0.3)


@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("band", [None, (100, 3500)])
def test_a_clip_of_any_length_is_judged(sr, band):
    """The reach of a fricative, 41 frames of 10 ms, is wider than a clip
    under 0.43 s, and np.convolve answers with the longer of the two."""
    word = _word(0.5, sr)
    for seconds in (0.19, 0.0, 2.0 / sr, 0.001, 0.01, 0.0299, 0.03, 0.0399, 0.04, 0.05,
                    0.1, 0.3, 0.42, 0.43, 0.44, 0.5):
        y = word[:int(round(seconds * sr))]
        assert kudio.vad(y, sr, band=band) == [], "nothing in it is quieter than the rest"
        assert kudio.vad_split(y, sr, band=band) == []
        trimmed, span = kudio.vad_trim(y, sr, band=band)
        assert span == (0, len(y)) and len(trimmed) == len(y)
        assert kudio.speech_ratio(y, sr, band=band) == 0.0
        mask = kudio.vad_frames(y, sr, band=band)
        assert mask.dtype == bool and len(mask) == 1 + len(y) // 256 and not mask.any()


@pytest.mark.parametrize("lead, word", [(0.1, 0.1), (0.1, 0.2), (0.1, 0.3), (0.05, 0.15)])
def test_a_word_in_a_short_clip_is_found(lead, word):
    """What it takes is room tone for the floor: 15% of the frames."""
    y = np.concatenate([_hiss(lead), _word(word) + _hiss(word, seed=8)])
    (start, end), = kudio.vad(y, SR, pad=0.0)
    # a frame that holds the first 10 ms of the word is loud already
    assert start == pytest.approx(lead - 0.02, abs=0.0101)
    assert end == pytest.approx(lead + word, abs=0.0101)
    assert kudio.speech_ratio(y, SR, pad=0.0) == pytest.approx((end - start) / (lead + word))
    piece, = kudio.vad_split(y, SR, pad=0.0)
    assert len(piece) == pytest.approx((end - start) * SR, abs=1)


def test_a_fricative_joins_a_vowel_in_a_short_clip():
    """Every frame of a clip this short is in reach of every other."""
    vowel = _word(0.1)
    hiss = 0.2 * np.random.default_rng(9).standard_normal(int(0.1 * SR))
    clip = np.concatenate([_hiss(0.08), vowel, hiss])
    assert len(clip) / SR < 0.43
    (start, end), = kudio.vad(clip, SR, pad=0.0)
    assert end == pytest.approx(len(clip) / SR, abs=0.0101), "the hiss is part of the word"
    alone = np.concatenate([_hiss(0.08), np.zeros(len(vowel)), hiss]) + _hiss(0.28, seed=10)
    assert kudio.vad(alone, SR, pad=0.0) == [], "and on its own it is not"


def test_a_frame_of_its_own_is_its_own_floor():
    """Under 40 ms there is one frame, and it is never above itself, unless
    the threshold says it need not be."""
    for seconds in (0.001, 0.02, 0.0399):
        y = _word(seconds)
        assert kudio.vad(y, SR, min_speech=0.0, pad=0.0) == []
        found = kudio.vad(y, SR, threshold_db=-1.0, min_speech=0.0, pad=0.0)
        assert found == [(0.0, len(y) / SR)]
        assert kudio.vad(y, SR, threshold_db=-1.0, min_speech=0.0, pad=0.0,
                         band=(100, 4000)) == found


def test_one_sample_has_no_spectrum(recwarn):
    for y in (np.array([0.3]), np.array([[0.3]]), [0.3]):
        assert kudio.vad(y, SR) == []
        assert kudio.vad(y, SR, band=(100, 4000)) == []
        assert kudio.vad_frames(y, SR).tolist() == [False]
        assert kudio.vad_frames(y, SR, center=False).tolist() == []
    assert not recwarn.list


@pytest.mark.parametrize("m", [1, 2, 3, 6, 40, 41, 62])
def test_the_smoothing_has_one_answer_per_frame(m):
    """Against the sum written out, for a kernel shorter than the values and
    for one longer."""
    from kudio.core._framing import convolve_same
    rng = np.random.default_rng(m)
    kernel = rng.integers(0, 5, m).astype(float)
    reach = (m - 1) // 2
    for n in (1, 2, 3, 5, 16, 41, 60):
        values = rng.integers(0, 5, n).astype(float)
        by_hand = [sum(values[j] * kernel[i + reach - j]
                       for j in range(n) if 0 <= i + reach - j < m) for i in range(n)]
        out = convolve_same(values, kernel)
        assert out.tolist() == by_hand
        if m <= n:
            assert out.tolist() == np.convolve(values, kernel, mode='same').tolist()


# -- what comes back is the caller's to write into --------------------------------
def test_the_cuts_are_copies(speech):
    """As trim_silence's and split_on_silence's are: silencing a cut must not
    silence the recording it came from."""
    before = speech.copy()
    trimmed, _ = kudio.vad_trim(speech, SR)
    untouched, _ = kudio.vad_trim(np.zeros(SR, dtype=np.float32), SR)
    for piece in [trimmed, untouched] + kudio.vad_split(speech, SR):
        assert not np.shares_memory(piece, speech)
        piece[:] = 0.0
    assert np.array_equal(speech, before)


@pytest.mark.parametrize("kind", [np.int16, np.float32, np.float64])
def test_a_cut_keeps_its_type_and_its_channel_axis(speech, kind):
    y = (speech * 30000).astype(kind) if kind is np.int16 else speech.astype(kind)
    assert kudio.vad_trim(y, SR)[0].dtype == kind
    assert all(piece.dtype == kind for piece in kudio.vad_split(y, SR))
    assert kudio.vad_trim(y[:, None], SR)[0].shape[1:] == (1,)
