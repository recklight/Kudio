# -*- coding: utf-8 -*-
"""A setting that is none and a clip that is no sound are refused by the function they were
given to, and named. Each row of REFUSED was answered before, or ended in an error of
numpy's, of scipy's or of Python's; what the answer was stands beside it. TAKEN is what
looks like one of them and is not."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import kudio
from kudio import Synthesizer
from kudio.core.stream import Recorder, wave_decode, wave_encode

NAN, INF = float('nan'), float('inf')
SR = 16000
RNG = np.random.default_rng(0)
T = np.arange(2 * SR) / SR
#: two seconds with a voice in the middle, over a floor
VOICE = (0.3 * sum(np.sin(2 * np.pi * 220 * k * T) / k for k in range(1, 8))
         * ((T > 0.5) & (T < 1.5)) + 0.002 * RNG.standard_normal(len(T)))
NOISE = 0.1 * RNG.standard_normal(SR)
IR = kudio.rir(SR, rt60=0.3, seed=0).astype(np.float64)
POWER = np.abs(kudio.STFT(sr=SR).analyse(VOICE)) ** 2
LOGSPEC = kudio.waveform_to_spectrogram(VOICE)
DENOISED = kudio.denoise_wav2spec(VOICE)


def one_bad(y, value=NAN):
    y = np.array(y, dtype=np.float64)
    y[len(y) // 3] = value
    return y


def curve(y=None):
    return kudio.loudness_over_time(VOICE if y is None else y, SR, window=0.4)


TRACK = kudio.f0(VOICE, SR)
MASKED = np.ma.masked_array(VOICE, mask=np.abs(VOICE) > 0.3)
COMPLEX = VOICE + 0.1j

REFUSED = {
    # -- how far above a floor, for how long ---------------------------------------------
    'vad threshold_db': (lambda: kudio.vad(VOICE, SR, threshold_db=NAN), r"vad\(\): threshold_db has to be a number of dB, got nan"),            # []
    'vad flatness_max': (lambda: kudio.vad(VOICE, SR, flatness_max=NAN), r"vad\(\): flatness_max has to be a number, got nan"),                  # []
    'vad min_speech': (lambda: kudio.vad(VOICE, SR, min_speech=NAN), r"vad\(\): min_speech has to be a finite number of seconds of 0 or more, got nan"),   # []
    'vad min_silence': (lambda: kudio.vad(VOICE, SR, min_silence='1'), r"vad\(\): min_silence .*, got '1'"),             # TypeError, or an answer where there was one span
    'vad pad': (lambda: kudio.vad(VOICE, SR, pad=-0.1), r"vad\(\): pad .* of 0 or more, got -0.1"),                      # as 0
    'vad frame': (lambda: kudio.vad(VOICE, SR, frame=-1), r"vad\(\): frame .* above 0, got -1"),                         # a frame of 16 samples
    'vad hop': (lambda: kudio.vad(VOICE, SR, hop=-1), r"vad\(\): hop .* above 0, got -1"),                               # a hop of one sample
    'vad hop 0': (lambda: kudio.vad(VOICE, SR, hop=0), r"vad\(\): hop .* above 0, got 0"),                               # ZeroDivisionError
    'vad band': (lambda: kudio.vad(VOICE, SR, band=(NAN, 4000)), r"vad\(\) needs a band"),                               # refused by 4.3.0 as well
    'speech_ratio': (lambda: kudio.speech_ratio(VOICE, SR, threshold_db=NAN), r"vad\(\): threshold_db"),                 # 0.0
    'vad_frames': (lambda: kudio.vad_frames(VOICE, SR, hop_length=2.5), r"vad_frames\(\): hop_length must be >= 1"),     # frames that line up with nothing
    'tonal_lines prominence': (lambda: kudio.tonal_lines(VOICE, SR, min_prominence_db=NAN), r"tonal_lines\(\): min_prominence_db .*, got nan"),    # no lines
    'tonal_lines resolution': (lambda: kudio.tonal_lines(VOICE, SR, resolution_hz=1000), r"tonal_lines\(\): resolution_hz=1000 is too coarse at 16000 Hz"),   # no lines
    'tonal_lines lines': (lambda: kudio.tonal_lines(VOICE, SR, max_lines=-1), r"tonal_lines\(\): max_lines .*, got -1"),  # all but one
    'voiced_segments': (lambda: TRACK.voiced_segments(NAN), r"PitchTrack.voiced_segments\(\): min_seconds .*, got nan"),  # []
    'spans_below': (lambda: curve().spans_below(NAN), r"LoudnessCurve.spans_below\(\): lufs .*, got nan"),                # []
    'spans_above': (lambda: curve().spans_above(-30, NAN), r"LoudnessCurve.spans_above\(\): min_seconds .*, got nan"),    # []
    'audio_report': (lambda: kudio.audio_report(VOICE, SR, clip_run=NAN), r"audio_report\(\): clip_run must be >= 1"),    # ValueError of Python's
    # -- rooms and alignment ----------------------------------------------------------------
    'rir rt60': (lambda: kudio.rir(SR, rt60=INF), r"rir\(\): rt60 .* above 0, got inf"),                                  # OverflowError
    'rir drr_db': (lambda: kudio.rir(SR, drr_db=NAN), r"rir\(\): drr_db .*, got nan"),                                    # all NaN
    'rir seconds': (lambda: kudio.rir(SR, seconds=0), r"rir\(\): seconds .*None.* above 0, got 0"),                       # the default
    'rir seed': (lambda: kudio.rir(SR, seed=-1), r"rir\(\): seed has to be a whole number of 0 or more, or None, got -1"),  # ValueError of numpy's
    'apply_rir clip': (lambda: kudio.apply_rir(one_bad(VOICE), IR), r"apply_rir\(\) got 1 sample\(s\) that are not finite"),      # all NaN
    'apply_rir room': (lambda: kudio.apply_rir(VOICE, one_bad(IR, INF)), r"apply_rir\(\) got 1 sample\(s\) that are not finite"),  # all NaN
    'rt60': (lambda: kudio.rt60(one_bad(IR), SR), r"rt60\(\) got 1 sample\(s\) that are not finite"),                     # a DRR of inf
    'rt60 complex': (lambda: kudio.rt60(IR + 0.1j, SR), r"rt60\(\): the samples are complex128, not real numbers"),       # that of the real part
    'schroeder_curve': (lambda: kudio.schroeder_curve(one_bad(IR)), r"schroeder_curve\(\) got 1 sample\(s\)"),            # NaN up to it
    'find_delay': (lambda: kudio.find_delay(one_bad(NOISE), NOISE, SR), r"find_delay\(\) got 1 sample\(s\) that are not finite"),  # "leads by 15999 samples"
    'find_delay of nothing': (lambda: kudio.find_delay(None, NOISE, SR), r"find_delay\(\): the samples are object, not real numbers"),   # an answer
    'find_delay complex': (lambda: kudio.find_delay(COMPLEX, VOICE, SR), r"find_delay\(\): the samples are complex128"),  # that of the real part
    'find_delay max_seconds': (lambda: kudio.find_delay(NOISE, NOISE, SR, max_seconds=NAN), r"find_delay\(\): max_seconds .*, got nan"),  # ValueError of Python's
    'align': (lambda: kudio.align(NOISE, NOISE, SR, min_correlation=NAN), r"align\(\): min_correlation .*, got nan"),     # "below nan"
    # -- loudness -----------------------------------------------------------------------------
    'loudness block_size': (lambda: kudio.loudness(VOICE, SR, block_size=NAN), r"loudness\(\): block_size .*, got nan"),  # ValueError of Python's
    'loudness complex': (lambda: kudio.loudness(COMPLEX, SR), r"the samples are complex128, not real numbers"),          # that of the real part
    'loudness masked': (lambda: kudio.loudness(MASKED, SR), r"of the samples are masked"),                               # that of the mask's samples too
    'true_peak complex': (lambda: kudio.true_peak(COMPLEX, SR), r"true_peak\(\): the samples are complex128"),           # that of the real part
    'true_peak oversample': (lambda: kudio.true_peak(VOICE, SR, oversample=2.9), r"true_peak\(\): oversample must be >= 1, a whole number, got 2.9"),   # 2
    'normalize_lufs': (lambda: kudio.normalize_lufs(VOICE, SR, lufs=NAN), r"normalize_lufs\(\): lufs .*, got nan"),       # all NaN
    'loudness_over_time': (lambda: kudio.loudness_over_time(VOICE, SR, window=0.4, hop=NAN), r"loudness_over_time\(\): hop .*, got nan"),   # ValueError of Python's
    'audio_report masked': (lambda: kudio.audio_report(MASKED, SR), r"audio_report\(\): \d+ of the samples are masked"),  # the peak under the mask
    'highpass masked': (lambda: kudio.highpass(MASKED, SR), r"highpass\(\): \d+ of the samples are masked"),             # what is under the mask, filtered
    'gain masked': (lambda: kudio.gain(MASKED, 6.0), r"gain\(\): \d+ of the samples are masked"),                       # the same
    'trim masked': (lambda: kudio.trim_silence(MASKED), r"trim_silence\(\): \d+ of the samples are masked"),            # the same
    'loudness of no channels': (lambda: kudio.loudness(np.zeros((SR, 0)), SR), r"loudness\(\) takes \(frames,\) or \(frames, channels\), got shape \(16000, 0\)"),   # ValueError of numpy's
    'true_peak of no channels': (lambda: kudio.true_peak(np.zeros((SR, 0)), SR), r"true_peak\(\) takes \(frames,\) or \(frames, channels\), got shape \(16000, 0\)"),  # -inf
    'a fade of a number': (lambda: kudio.fade_in(0.5, SR), r"fade_in\(\) takes \(frames,\) or \(frames, channels\), got shape \(\)"),   # IndexError
    'a filter of a number': (lambda: kudio.highpass(np.float32(0.5), SR), r"highpass\(\) takes \(frames,\) or \(frames, channels\), got shape \(\)"),   # IndexError
    'trim_silence 1e30': (lambda: kudio.trim_silence((VOICE * 1e30).astype(np.float32)), r"trim_silence\(\): the samples go as high as .*, and the square of that is past what float32 holds"),   # all of it silence
    # -- one NaN is all of it -------------------------------------------------------------------
    'remove_dc': (lambda: kudio.remove_dc(one_bad(VOICE)), r"remove_dc\(\) got 1 sample\(s\) that are not finite"),       # all NaN
    'highpass': (lambda: kudio.highpass(one_bad(VOICE), SR), r"highpass\(\) got 1 sample\(s\) that are not finite"),      # all NaN
    'lowpass': (lambda: kudio.lowpass(one_bad(VOICE), SR, 3000.0), r"lowpass\(\) got 1 sample\(s\) that are not finite"),  # all NaN
    'bandpass': (lambda: kudio.bandpass(one_bad(VOICE, INF), SR), r"bandpass\(\) got 1 sample\(s\)"),                     # all NaN
    'bandstop': (lambda: kudio.bandstop(one_bad(VOICE), SR), r"bandstop\(\) got 1 sample\(s\)"),                          # all NaN
    'wavelet': (lambda: kudio.wavelet_low_pass_filter(one_bad(VOICE)), r"wavelet_low_pass_filter\(\) got 1 sample\(s\)"),  # all NaN
    'wavelet thres': (lambda: kudio.wavelet_low_pass_filter(VOICE, thres=NAN), r"wavelet_low_pass_filter\(\): thres .*, got nan"),   # the clip as it was
    'wavelet of no name, silent clip': (lambda: kudio.wavelet_low_pass_filter(np.zeros(SR), wavelet='db39'), r"wavelet_low_pass_filter\(\): wavelet 'db39' is not a discrete wavelet"),   # the clip as it was
    'wavelet of no name, no threshold': (lambda: kudio.wavelet_low_pass_filter(VOICE, thres=0, wavelet='morl'), r"wavelet_low_pass_filter\(\): wavelet 'morl' is not a discrete wavelet"),   # the clip as it was
    'trad_enhance': (lambda: kudio.trad_enhance(one_bad(VOICE), SR), r"trad_enhance\(\) got 1 sample\(s\)"),              # NaN to the end
    'mu_law': (lambda: kudio.mu_law(one_bad(VOICE, INF)), r"mu_law\(\) got 1 sample\(s\) that are not finite"),           # 1.0
    'a_law': (lambda: kudio.a_law(one_bad(VOICE)), r"a_law\(\) got 1 sample\(s\) that are not finite"),                   # NaN
    'bit_depth': (lambda: kudio.bit_depth(one_bad(VOICE, -INF)), r"bit_depth\(\) got 1 sample\(s\) that are not finite"),  # -1.0
    'mu_law complex': (lambda: kudio.mu_law(COMPLEX), r"mu_law\(\): the samples are complex128"),                         # that of the real part
    'spectral_enhance complex': (lambda: kudio.spectral_enhance(COMPLEX, SR), r"spectral_enhance\(\): the samples are complex128"),   # that of the real part
    'save_wave complex': (lambda: kudio.save_wave("never.wav", COMPLEX, SR), r"save_wave\(\): the samples are complex128"),   # the real part, written
    'estimate_noise_psd complex': (lambda: kudio.estimate_noise_psd(kudio.STFT(sr=SR).analyse(VOICE)), r"estimate_noise_psd\(\): the samples are complex64"),   # of the real parts
    'estimate_noise_psd empty': (lambda: kudio.estimate_noise_psd(POWER[:0]), r"estimate_noise_psd\(\) takes a power spectrogram \(frames, bins\), got shape \(0, 257\)"),   # ValueError of numpy's
    # -- settings of filters, joins and links -----------------------------------------------------
    'filter order': (lambda: kudio.highpass(VOICE, SR, order=2.5), r"highpass\(\): order must be >= 1, a whole number, got 2.5"),    # TypeError of scipy's
    'filter cutoff': (lambda: kudio.highpass(VOICE, SR, cutoff='80'), r"cutoff must be between 0 and Nyquist \(8000.0 Hz\) for sr=16000, got '80'"),   # 80
    'fade': (lambda: kudio.fade_in(VOICE, SR, duration=NAN), r"fade_in\(\): duration .* of 0 or more, got nan"),          # ValueError of Python's
    'crossfade': (lambda: kudio.crossfade(VOICE, VOICE, SR, INF), r"crossfade\(\): duration .*, got inf"),                # OverflowError
    'crossfade channels': (lambda: kudio.crossfade(VOICE, np.stack([VOICE] * 2, 1), SR), r"crossfade\(\) joins two clips of as many channels.*\(32000,\) and \(32000, 2\)"),   # ValueError of numpy's
    'splice channels': (lambda: kudio.splice([np.stack([VOICE] * 2, 1), VOICE], SR), r"crossfade\(\) joins two clips of as many channels"),   # ValueError of numpy's
    'splice one': (lambda: kudio.splice(VOICE, SR), r"splice\(\) takes a list of clips, and got one array of shape \(32000,\)"),   # IndexError
    'channel_spec band': (lambda: kudio.channel_spec({'band': 'no'}), r"channel_spec\(\): band is True for the telephone band or False, got 'no'"),   # the telephone band
    'channel_spec band range': (lambda: kudio.channel_spec({'band': (300, 3400)}), r"channel_spec\(\): band .*, got \(300, 3400\)"),   # the telephone band
    'spec_augment': (lambda: kudio.spec_augment(POWER, n_freq_masks=-1), r"spec_augment\(\): n_freq_masks is a whole number, 0 or more, got -1"),   # no masks
    'spec_augment width': (lambda: kudio.spec_augment(POWER, freq_mask_width=-3), r"spec_augment\(\): freq_mask_width .*, got -3"),   # ValueError of numpy's
    'seed': (lambda: kudio.add_noise_snr(VOICE, NOISE, 5.0, seed=1.5), r"add_noise_snr\(\): seed has to be .*, got 1.5"),   # TypeError of numpy's
    'random_gain one': (lambda: kudio.random_gain(VOICE, (1.0,), seed=0), r"random_gain\(\): db_range is two finite numbers of dB, lowest first, got \(1.0,\)"),   # +1 dB every time
    'random_gain reversed': (lambda: kudio.random_gain(VOICE, (6, -6), seed=0), r"random_gain\(\): db_range .*, got \(6, -6\)"),   # ValueError of numpy's
    'telephone band': (lambda: kudio.telephone(VOICE, SR, band=None), r"telephone\(\): band is a \(low, high\) pair in Hz, got None"),   # TypeError
    'independent_burst': (lambda: kudio.independent_burst('0.1'), r"loss must be a fraction in \[0, 1\), got '0.1'"),   # TypeError
    'seed of a split': (lambda: kudio.split_pairs([kudio.Pair('a', 'b')], seed='x'), r"split_pairs\(\): seed has to be .*, got 'x'"),   # ValueError of numpy's
    # -- the noise the enhancers are told of ----------------------------------------------------------
    'bias': (lambda: kudio.spectral_enhance(VOICE, SR, noise='minimum', noise_kwargs=dict(bias=-1)), r"noise estimator 'minimum'\(\): bias .* above 0, got -1"),   # the clip as it was
    'ad': (lambda: kudio.spectral_enhance(VOICE, SR, noise_kwargs=dict(ad=5)), r"noise estimator 'mcra'\(\): ad .* from 0 to 1, got 5"),   # the clip, but for 0.002
    'ass': (lambda: kudio.spectral_enhance(VOICE, SR, noise_kwargs=dict(ass=NAN)), r"noise estimator 'mcra'\(\): ass .*, got nan"),   # an answer
    'window': (lambda: kudio.make_noise_tracker('mcra', window=-1), r"noise estimator 'mcra'\(\): window must be >= 1"),   # an answer
    'n_frames': (lambda: kudio.make_noise_tracker('initial', n_frames=-3), r"noise estimator 'initial'\(\): n_frames must be >= 1"),   # as 1
    'n_fft': (lambda: kudio.spectral_enhance(VOICE, SR, n_fft='512'), r"spectral_enhance\(\): n_fft must be >= 1, a whole number of samples, got '512'"),   # TypeError
    'alpha': (lambda: kudio.spectral_enhance(VOICE, SR, alpha=None), r"logmmse: alpha has to be a finite number, got None"),   # TypeError
    'no estimators': (lambda: kudio.compare_enhancers(VOICE, SR, ['wiener'], noises=[], metrics=False), r"compare_enhancers\(\): noises is empty"),   # IndexError
    'lsd n_fft': (lambda: kudio.lsd(VOICE, VOICE, SR, n_fft=512.5), r"lsd\(\): n_fft must be a whole number of samples, got 512.5"),   # 512
    'lsd hop': (lambda: kudio.lsd(VOICE, VOICE, SR, hop_length=float('nan')), r"lsd\(\): hop_length must be a whole number of samples, got nan"),   # ValueError of Python's
    'quantile': (lambda: kudio.estimate_noise_psd(POWER, 'quantile', quantile=NAN), r"quantile must be between 0 and 1, got nan"),   # refused by 4.3.0 as well
    # -- lengths, and 0 is no default ---------------------------------------------------------------------
    'f0 hop': (lambda: kudio.f0(VOICE, SR, hop_length=0), r"f0\(\): hop_length must be >= 1, a whole number of samples, or None for 10 ms, got 0"),   # 10 ms
    'desired_length': (lambda: kudio.waveform_to_spectrogram(VOICE, desired_length=0), r"desired_length .* or None for all of them, got 0"),   # all of it
    'stream n_mels': (lambda: kudio.SpectrogramStream(SR, n_mels=False), r"SpectrogramStream\(\): n_mels .* or 0 for a linear axis, got False"),   # 0
    # -- spectrograms back to sound --------------------------------------------------------------------
    'log power NaN': (lambda: kudio.spectrogram_to_waveform(VOICE, np.where(LOGSPEC > 0, NAN, LOGSPEC)), r"spectrogram_to_waveform\(\): \d+ number\(s\) of the spectrogram are NaN or inf"),   # all NaN
    'log power complex': (lambda: kudio.spectrogram_to_waveform(VOICE, LOGSPEC + 0j), r"spectrogram_to_waveform\(\) takes real numbers, got complex"),   # 1e14 and more
    'denoise shapes': (lambda: kudio.denoise_spec2wav(DENOISED[0], DENOISED[1].T), r"denoise_spec2wav\(\) takes .*, got shapes \(1, 126, 257\) and \(126, 257\)"),   # ValueError of numpy's
    'denoise NaN': (lambda: kudio.denoise_spec2wav(DENOISED[0] * NAN, DENOISED[1]), r"denoise_spec2wav\(\) got \d+ sample\(s\) that are not finite"),   # all NaN
    'context of a complex': (lambda: kudio.stack_context(POWER + 0j, 2), r"stack_context\(\) takes real numbers, got complex"),   # the real part
    'context 2.5': (lambda: kudio.stack_context(POWER, 2.5), r"context must be >= 0, a whole number, got 2.5"),   # TypeError of numpy's
    'context True': (lambda: kudio.waveform_to_spectrogram(VOICE, forward_backward=True), r"context must be >= 0, a whole number, got True"),   # one frame each side
    'context of a corpus': (lambda: kudio.ContextFrames([POWER], float('nan')), r"context must be >= 0, a whole number, got nan"),   # ValueError of Python's
    'windows 2.5': (lambda: kudio.frame_windows(POWER, 2.5), r"n_frames must be >= 1, a whole number, got 2.5"),   # TypeError of numpy's
    'windows of complex': (lambda: kudio.frame_windows(POWER + 0j, 16), r"frame_windows\(\) takes real numbers, got complex"),   # the real part
    # -- matrices ------------------------------------------------------------------------------------------
    'transform one column': (lambda: kudio.Standardizer().fit(POWER).transform(POWER[:9, :1]), r"Standardizer.transform\(\) was fitted on frames 257 wide, and got shape \(9, 1\)"),   # spread over the columns
    'transform a number': (lambda: kudio.Standardizer().fit(POWER).transform(3.0), r"Standardizer.transform\(\) was fitted .*, and got shape \(\)"),   # 257 numbers
    'inverse': (lambda: kudio.Standardizer().fit(POWER).inverse(POWER[:, :1]), r"Standardizer.inverse\(\) was fitted on frames 257 wide"),   # spread over the columns
    'fit nothing': (lambda: kudio.Standardizer().fit(POWER[:0]), r"Standardizer.fit\(\) got no frames to learn from, shape \(0, 257\)"),   # fitted, all NaN
    'fit NaN': (lambda: kudio.Standardizer().fit(np.where(np.arange(257) == 3, NAN, POWER)), r"Standardizer.fit\(\) got \d+ number\(s\) that are not finite"),   # a column of NaN
    'fit complex': (lambda: kudio.Standardizer().fit(POWER + 0j), r"Standardizer.fit\(\) takes real numbers, got complex"),   # of the real part
    # -- streams and bytes --------------------------------------------------------------------------------------
    'wave_encode 16-bit floats': (lambda: wave_encode(40000.0 * VOICE), r"wave_encode\(\): the samples are float64 and go as high as"),   # wrapped round
    'wave_encode NaN': (lambda: wave_encode(one_bad(VOICE)), r"wave_encode\(\) got 1 sample\(s\) that are not finite"),    # 0
    'wave_encode 64-bit': (lambda: wave_encode(np.array([70000, 1], dtype=np.int64)), r"wave_encode\(\): the samples are int64"),   # wrapped round
    'wave_decode': (lambda: wave_decode(b"\0" * 7, 1), r"wave_decode\(\): 7 bytes are no whole number of frames of 1"),    # ValueError of numpy's
    'wave_decode channels': (lambda: wave_decode(b"\0" * 8, 0), r"wave_decode\(\): channels must be >= 1"),               # ZeroDivisionError
    'audio_info': (lambda: kudio.audio_info(np.zeros((4, 4, 4)), sr=SR), r"audio_info\(\) takes \(frames,\) or \(frames, channels\), got shape \(4, 4, 4\)"),   # an answer
    'files as an array': (lambda: kudio.check_input(np.array(['a.wav', 'b.wav'])), r"this takes a path, or a list of them, and got an array of shape \(2,\)"),   # ValueError of numpy's
    # -- a clip is (frames,) or (frames, channels), a channel or more (stage 9b) ------------
    'gain of a number': (lambda: kudio.gain(np.float32(0.25), 6.0), r"gain\(\) takes \(frames,\) or \(frames, channels\), got shape \(\)"),   # a number, 0.499
    'gain of three axes': (lambda: kudio.gain(np.zeros((4, 8, 2)), 6.0), r"gain\(\) takes .*, got shape \(4, 8, 2\)"),   # the array, scaled
    'normalize of no channel': (lambda: kudio.normalize(np.zeros((SR, 0))), r"normalize\(\) takes .*, got shape \(16000, 0\)"),   # (16000, 0)
    'mu_law of a number': (lambda: kudio.mu_law(np.float32(0.25)), r"mu_law\(\) takes .*, got shape \(\)"),   # one sample
    'vad of a number': (lambda: kudio.vad(np.float32(0.25), SR), r"vad\(\) takes .*, got shape \(\)"),   # []
    'find_delay of a number': (lambda: kudio.find_delay(np.float32(0.25), NOISE, SR), r"find_delay\(\) takes .*, got shape \(\)"),   # no delay, correlation 0
    'reverse of three axes': (lambda: kudio.reverse(np.zeros((4, 8, 2))), r"reverse\(\) takes .*, got shape \(4, 8, 2\)"),   # along the first axis
    'remove_dc of three axes': (lambda: kudio.remove_dc(np.zeros((4, 8, 2))), r"remove_dc\(\) takes .*, got shape \(4, 8, 2\)"),   # along the first axis
    'resample of no channel': (lambda: kudio.resample(np.zeros((1600, 0)), SR, 8000), r"resample\(\): expected .*, got shape \(1600, 0\)"),   # (1600, 0)
    'audio_info of a number': (lambda: kudio.audio_info(np.float32(0.25), sr=SR), r"audio_info\(\) takes .*, got shape \(\)"),   # TypeError of pathlib's
    'a stream of a number': (lambda: kudio.SpectrogramStream(SR).push(np.float32(0.25)), r"SpectrogramStream.push\(\) takes .*, got shape \(\)"),   # no columns
    'compare a number': (lambda: kudio.compare_enhancers(np.float32(0.5), SR, ['wiener'], metrics=False), r"compare_enhancers\(\) takes .*, got shape \(\)"),   # TypeError of Python's
    'wave_separate odd': (lambda: kudio.wave_separate(np.zeros(31999), 2), r"wave_separate\(\): 31999 interleaved samples are no whole number of frames of 2"),   # 16000 and 15999 samples
    # -- spectra and matrices (stage 9b) ----------------------------------------------------------------
    'negative power': (lambda: kudio.estimate_noise_psd(-POWER), r"estimate_noise_psd\(\): \d+ of the powers are below 0"),   # all 1e-12
    'dB for power': (lambda: kudio.estimate_noise_psd(10 * np.log10(POWER + 1e-12)), r"a spectrogram in dB is no power"),   # an estimate of the dB
    'spec_augment complex': (lambda: kudio.spec_augment(POWER + 0j, seed=0), r"spec_augment\(\) takes real numbers, got complex"),   # masked with the real part of the mean
    'context of objects': (lambda: kudio.stack_context(POWER.astype(object), 2), r"stack_context\(\) takes real numbers, got object"),   # taken
    'context of text': (lambda: kudio.stack_context(POWER.astype(str), 2), r"stack_context\(\) takes real numbers, got <U"),   # the text read as numbers
    'fit of objects': (lambda: kudio.Standardizer().fit(POWER.astype(object)), r"Standardizer.fit\(\) takes real numbers, got object"),   # fitted
    'windows of bytes': (lambda: kudio.frame_windows(POWER.astype(bytes), 16), r"frame_windows\(\) takes real numbers, got \|S"),   # the bytes read as numbers
    # -- rooms and curves of nothing (stage 9b) -----------------------------------------------------------
    'rt60 of silence': (lambda: kudio.rt60(np.zeros(SR), SR), r"rt60\(\): the impulse response is all zeros"),   # C50, C80 and DRR of inf
    'curve of a corrupt clip': (lambda: kudio.loudness_over_time(one_bad(VOICE), SR, window=0.4), r"loudness_over_time\(\) got 1 sample\(s\) that are not finite"),   # nan from it on, with a warning
    'range of a corrupt clip': (lambda: kudio.loudness_range(np.tile(one_bad(VOICE), 2), SR), r"loudness_range\(\) got 2 sample\(s\) that are not finite"),   # nan, with a warning
    # -- flags (stage 9b): test_input_contract.py has every one of them --------------------------------
    'zero_phase no': (lambda: kudio.highpass(VOICE, SR, zero_phase='no'), r"highpass\(\): zero_phase is True or False, got 'no'"),   # True
    'center 2': (lambda: kudio.f0(VOICE, SR, center=2), r"f0\(\): center is True or False, got 2"),   # True
    'trim nan': (lambda: kudio.apply_rir(VOICE, IR, trim=NAN), r"apply_rir\(\): trim is True or False, got nan"),   # True
    'mono no': (lambda: kudio.audio_info(VOICE, sr=SR, peak='no'), r"audio_info\(\): peak is True or False, got 'no'"),   # True
    'match_gain no': (lambda: kudio.lsd(VOICE, VOICE, SR, match_gain='no'), r"lsd\(\): match_gain is True or False, got 'no'"),   # True
    'fast no': (lambda: kudio.srmr(VOICE, SR, fast='no'), r"srmr\(\): fast is True or False, got 'no'"),   # True
    'set_params': (lambda: kudio.AudioDataManager().set_params(SR, 6), r"set_params\(\): ch has to be 1 or 2 channels, got 6"),   # 1
}


@pytest.mark.parametrize("name", REFUSED)
def test_it_is_refused_by_name(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    call, said = REFUSED[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=said):
            call()
    assert not list(tmp_path.iterdir())


TAKEN = {
    'any flatness': lambda: kudio.vad(VOICE, SR, flatness_max=INF),
    'no threshold': lambda: kudio.vad(VOICE, SR, threshold_db=-INF),
    'no gap splits': lambda: len(kudio.vad(VOICE, SR, min_silence=INF)) == 1,
    'no minimum': lambda: kudio.vad(VOICE, SR, min_speech=0, pad=0),
    'a band': lambda: kudio.vad(VOICE, SR, band=(0, 4000)),
    'no lines': lambda: kudio.tonal_lines(VOICE, SR, max_lines=0).lines == (),
    'no bound': lambda: kudio.find_delay(NOISE, NOISE, SR, max_seconds=INF).delay == 0,
    'a bound under a sample': lambda: kudio.find_delay(NOISE, NOISE, SR, max_seconds=0).delay == 0,
    'align anything': lambda: kudio.align(NOISE, NOISE[::-1], SR, min_correlation=0),
    'all silent windows': lambda: curve().spans_below(INF),
    'a hop for the enhancer': lambda: kudio.spectral_enhance(VOICE, SR, hop_length=0),
    'no shift': lambda: np.array_equal(kudio.pitch_shift(VOICE, SR, 0), VOICE),
    'a linear axis': lambda: kudio.SpectrogramStream(SR, n_mels=0).n_bins == 257,
    'and none given': lambda: kudio.SpectrogramStream(SR, n_mels=None).n_bins == 257,
    'one frame': lambda: kudio.Standardizer().fit(POWER).transform(POWER[0]).shape == (257,),
    'a batch': lambda: kudio.Standardizer().fit(POWER).transform(POWER[None]).shape == (1,) + POWER.shape,
    'integers': lambda: kudio.Standardizer().fit(np.arange(20).reshape(5, 4)).transform([[1, 2, 3, 4]]),
    'statistics of one number': lambda: kudio.Standardizer(mean=0.0, std=1.0).transform(POWER),
    'a switch': lambda: [kudio.channel_spec({'band': b, 'bits': 8}) for b in (True, False, None, 0, 1, np.True_)],
    'a stack of clips': lambda: len(kudio.splice(np.stack([NOISE, NOISE]), SR)),
    'no masks': lambda: np.array_equal(kudio.spec_augment(POWER, n_freq_masks=0, n_time_masks=0), POWER),
    'a list to mask': lambda: kudio.spec_augment(POWER.tolist(), seed=0),
    'a threshold of nothing': lambda: np.array_equal(kudio.wavelet_low_pass_filter(VOICE, thres=0), VOICE),
    'nothing masked': lambda: kudio.loudness(np.ma.masked_array(VOICE, mask=False), SR),
    'a generator': lambda: kudio.rir(SR, seed=np.random.default_rng(3)),
    'what it can be': lambda: kudio.make_noise_tracker('mcra', ass=0, ad=1, ap=0.5, delta=1, window=1),
    'no context': lambda: np.array_equal(kudio.waveform_to_spectrogram(VOICE, forward_backward=False), LOGSPEC),
    'a whole float of context': lambda: kudio.stack_context(POWER, 2.0).shape == (len(POWER), 257 * 5),
    'no frames to stack': lambda: kudio.stack_context(POWER[:0], 2).shape == (0, 257 * 5),
    'a mask to stack': lambda: kudio.stack_context(np.eye(3, dtype=bool), 1).shape == (3, 9),
    'a log power of nothing': lambda: np.isfinite(kudio.spectrogram_to_waveform(VOICE, np.where(LOGSPEC < -5, -INF, LOGSPEC))).all(),
    'a list as a clip': lambda: kudio.audio_info(VOICE.tolist(), sr=SR).frames == len(VOICE),
    'a range of one gain': lambda: np.allclose(kudio.random_gain(VOICE, (3.0, 3.0), seed=0), kudio.gain(VOICE, 3.0)),
    'a NaN keeps to its sample': lambda: np.isnan(kudio.gain(one_bad(VOICE), 3.0)).sum() == 1
                                         and np.isnan(kudio.fade_in(one_bad(VOICE), SR)).sum() == 1,
}


@pytest.mark.parametrize("name", TAKEN)
def test_it_is_taken(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert TAKEN[name]() is not False


# -- what is said of what there is nothing of -----------------------------------------------------

def test_a_room_of_nothing_is_no_room():
    """Its C50, C80 and DRR were all +inf, of no energy over none. A room that is all
    direct sound is one, of no reverberation."""
    with pytest.raises(kudio.FeatureError, match=r"^rt60\(\): the impulse response is all "
                                                 r"zeros"):
        kudio.rt60(np.zeros(SR), SR)
    dry = kudio.rt60(np.r_[1.0, np.zeros(SR)], SR)
    assert dry.drr == dry.c50 == INF


def test_the_curve_of_a_corrupt_clip_is_not_that_of_silence():
    """Its maximum was -inf and its range 0: it is refused, as every other measurement
    of it is."""
    with pytest.raises(kudio.FeatureError, match=r"^loudness_over_time\(\) got 1 sample"):
        curve(np.where(np.arange(len(VOICE)) == 0, NAN, VOICE))
    silent = curve(np.zeros(len(VOICE)))
    assert silent.max == silent.min == -INF and silent.range_lu == 0


def test_a_rate_that_is_not_whole_is_kept():
    """16000.9 came back as 16000."""
    for found in (kudio.rt60(IR, 16000.9), kudio.find_delay(NOISE, NOISE, 16000.9),
                  kudio.audio_report(VOICE, 16000.9), curve_at(16000.9),
                  kudio.SpectrogramStream(16000.9), kudio.StreamEnhancer(16000.9)):
        assert found.sr == 16000.9
    assert kudio.rt60(IR, 16000.0).sr == 16000 and isinstance(kudio.rt60(IR, 16000.0).sr, int)


def curve_at(sr):
    return kudio.loudness_over_time(VOICE, sr, window=0.4)


def test_a_clip_too_short_for_a_line_is_said_to_be():
    with pytest.warns(UserWarning, match=r"tonal_lines\(\): the clip has 60 sample\(s\), too few"):
        assert kudio.tonal_lines(VOICE[SR:SR + 60], SR).lines == ()


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_16_bit_samples_held_in_32_bits_are_said_to_be(dtype):
    """They read as 96 dB down: a loudness of -inf, and a file of zeros."""
    y = (32767 * 0.4 * np.sin(2 * np.pi * 440 * T)).astype(np.int32)
    held = y if dtype == np.int32 else (y.astype(np.int64) + 2 ** 31).astype(np.uint32)
    for call in (lambda: kudio.loudness(held, SR), lambda: kudio.mu_law(held),
                 lambda: kudio.gain(held, 0.0)):
        with pytest.warns(UserWarning, match=rf"the samples are {dtype.__name__} and go no further "
                                             rf"than 1310\d, -104.3 dBFS .*pass them as int16"):
            call()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kudio.loudness(y.astype(np.int16), SR)
        kudio.loudness(y * 65536, SR)
        kudio.loudness(0 * y, SR)


# -- bytes ----------------------------------------------------------------------------------------

def test_float_audio_comes_back_from_its_bytes():
    """0.4 of full scale came back as 0."""
    for y in (VOICE, VOICE.astype(np.float32), np.stack([VOICE, -VOICE], axis=1)):
        back = wave_decode(wave_encode(y), 1 if y.ndim == 1 else 2).squeeze()
        assert np.max(np.abs(back / 32768 - y)) <= 0.5 / 32768 + 1e-9
    for y in (np.array([0, 128, 255], np.uint8), np.array([-2 ** 31, 0, 2 ** 30], np.int32)):
        assert wave_decode(wave_encode(y), 1).ravel().tolist() == \
            [-32768, 0, 32512] if y.dtype == np.uint8 else [-32768, 0, 16384]


def test_16_bit_numbers_are_their_own_bytes():
    numbers = np.array([0, 1, -1, 32767, -32768])
    for dtype in (np.int16, np.int32, np.int64):
        assert wave_decode(wave_encode(numbers.astype(dtype)), 1).ravel().tolist() == numbers.tolist()


def test_what_is_past_full_scale_is_held_there_and_said():
    with pytest.warns(UserWarning, match=r"wave_encode\(\): 1 sample\(s\) are past full scale, the "
                                         r"highest at 1.5"):
        assert wave_decode(wave_encode(np.array([0.0, 1.5, -1.0])), 1).ravel().tolist() == \
            [0, 32767, -32768]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert wave_decode(wave_encode(np.array([1.0, -1.0])), 1).ravel().tolist() == [32767, -32768]


# -- files and folders --------------------------------------------------------------------------------

def test_a_folder_with_a_dot_in_its_name_is_copied_to(tmp_path):
    """A date, taken for the extension of a file: nothing was copied and [] came back."""
    kudio.save_wave(tmp_path / "a.wav", NOISE, SR)
    assert [p.name for p in kudio.copy_waves(tmp_path / "session.2026-10-11", tmp_path / "a.wav")] \
        == ["a.wav"]


def test_the_two_arguments_the_other_way_round_are_refused(tmp_path):
    """[] came back, and the folder to copy from was the one that was to be copied to."""
    kudio.save_wave(tmp_path / "a.wav", NOISE, SR)
    for first in (tmp_path / "a.wav", tmp_path / "none.wav"):
        with pytest.raises(ValueError, match=r"^copy_waves\(\): the folder to copy to comes first"):
            kudio.copy_waves(first, tmp_path / "out")
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_an_empty_config_is_the_defaults_and_is_said_to_be(tmp_path):
    (tmp_path / "cfg.yaml").write_text("")
    with pytest.warns(UserWarning, match="cfg.yaml' is empty; using the defaults"):
        found = kudio.KudioConfig.load_config(tmp_path / "cfg.yaml")
    assert found == kudio.KudioConfig(**kudio.KudioConfig.default_config()) and len(found) > 1


def _inputs(tmp_path):
    for name in ("clean", "noise"):
        (tmp_path / name).mkdir()
        kudio.save_wave(tmp_path / name / "a.wav", NOISE, SR)
    return str(tmp_path / "clean"), str(tmp_path / "noise")


def test_all_of_the_outputs_can_be_written_as_noise_too(tmp_path):
    """A share of 1.0 wrote none of them, and 0.99 all."""
    syx = Synthesizer(*_inputs(tmp_path), out_path=str(tmp_path / "out"), snr_ratio=0)
    assert len(syx.syn(seed=0, is_silence=True, p_silence=1.0)) == 2
    assert len(syx.syn(seed=0, is_silence=True, p_silence=0)) == 1


def test_a_seed_is_kept_to_the_last_digit(tmp_path):
    syx = Synthesizer(*_inputs(tmp_path), out_path=str(tmp_path / "out"), snr_ratio=0)
    for seed, kept in ((2 ** 70 + 1, 2 ** 70 + 1), (np.int64(7), 7), (3.0, 3), (True, 1)):
        syx.syn(seed=seed)
        assert syx.seed == kept and type(syx.seed) is int


def test_a_clip_of_one_frame_comes_back_as_long_as_its_frames():
    """One frame was (257,) against (257, 1), 257 frames of sound."""
    with pytest.warns(UserWarning, match="200 sample"):
        short = kudio.denoise_wav2spec(VOICE[SR:SR + 200])
    for spec, phase in (short, kudio.denoise_wav2spec(VOICE[SR:SR + 3000])):
        assert len(kudio.denoise_spec2wav(spec, phase)) == 256 * (phase.shape[1] - 1)


@pytest.mark.parametrize("given, said", [
    (dict(seed=1.5), r"seed must be >= 0, a whole number, got 1.5"),
    (dict(seed='1'), r"seed must be >= 0, a whole number, got '1'"),
    (dict(is_silence=True, p_silence=2), r"p_silence .* from 0 to 1, got 2"),
    (dict(is_silence=True, p_silence=NAN), r"p_silence .*, got nan"),
    (dict(rdn_choice_num=-3), r"rdn_choice_num is a whole number of clean files, or 0 .*, got -3")])
def test_a_run_that_is_none_empties_nothing(tmp_path, given, said):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("from an earlier run")
    syx = Synthesizer(*_inputs(tmp_path), out_path=str(tmp_path / "out"), snr_ratio=0)
    with pytest.raises(ValueError, match=said):
        syx.syn(**{'seed': 0, **given})
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["keep.txt"]


def test_a_hop_that_does_not_fit_is_refused_once_and_not_once_a_file(tmp_path):
    clean, _ = _inputs(tmp_path)
    for given in (dict(hop_length=511), dict(n_fft=0), dict(window='nonsense')):
        with pytest.raises(ValueError):
            kudio.enhance_folder(clean, tmp_path / "out", **given)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("call, given, said", [
    (kudio.convert_folder, dict(subtype='PCM_99'), r"subtype is one of what a WAV file holds"),
    (kudio.convert_folder, dict(trim_db=NAN), r"trim_db has to be a finite number of dB above 0"),
    (kudio.convert_folder, dict(peak=0), r"peak has to be a finite number above 0, got 0"),
    (kudio.convert_folder, dict(lufs='x'), r"lufs has to be a finite number of LUFS, got 'x'"),
    (kudio.convert_folder, dict(mono='no'), r"mono is True or False, got 'no'"),
    (kudio.enhance_folder, dict(subtype='pcm 16'), r"subtype is one of what a WAV file holds"),
    (kudio.enhance_folder, dict(report='no'), r"report is True or False, got 'no'")],
    ids=lambda v: getattr(v, '__name__', None) or (str(v) if isinstance(v, dict) else ''))
def test_a_setting_that_fails_every_file_fails_once(tmp_path, call, given, said):
    """Three files and a subtype misspelt were three failures, one a file, and the folder
    made for them."""
    (tmp_path / "in").mkdir()
    for k in range(3):
        kudio.save_wave(tmp_path / "in" / f"{k}.wav", NOISE, SR)
    with pytest.raises(ValueError, match=rf"^{call.__name__}\(\): {said}"):
        call(tmp_path / "in", tmp_path / "out", **given)
    assert not (tmp_path / "out").exists()


# -- recording ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("given, said", [
    (dict(seconds=1, sr=-1), "sr must be a positive whole rate"),
    (dict(seconds=NAN), r"record\(\): seconds .*, got nan"),
    (dict(seconds=1, channels=0), r"record\(\): channels must be >= 1")])
def test_a_take_that_is_none_is_not_recorded(monkeypatch, given, said):
    from kudio.util import check

    def opened():
        pytest.fail("the device was asked")

    monkeypatch.setattr(check, "_sounddevice", opened)
    with pytest.raises(ValueError, match=said):
        kudio.record(**given)


@pytest.mark.parametrize("make, given, said", [
    (Recorder, dict(rate=0), "rate must be a positive whole rate"),
    (Recorder, dict(channels=0), r"Recorder\(\): channels must be >= 1"),
    (kudio.LocalStreamReader, dict(rate=NAN), "rate must be a positive whole rate"),
    (kudio.LocalStreamReader, dict(frame_size=0), r"LocalStreamReader\(\): frame_size must be >= 1"),
    (kudio.StreamRecorder, dict(sr=-1), "sr must be a positive whole rate"),
    (kudio.StreamRecorder, dict(max_seconds=0), r"StreamRecorder\(\): max_seconds .* above 0, got 0")])
def test_a_stream_that_is_none_is_not_opened(monkeypatch, make, given, said):
    from kudio.util import check

    def opened():
        pytest.fail("the device was asked")

    monkeypatch.setattr(check, "_sounddevice", opened)
    with pytest.raises(ValueError, match=said):
        make(**given)


# -- the command line ---------------------------------------------------------------------------------

@pytest.mark.parametrize("argv, rc", [
    (["room", "--make", "{dir}/room.wav", "--rt60", "inf"], 2), (["pitch", "{wav}", "--hop", "0"], 2),
    (["vad", "{wav}", "--threshold-db", "nan"], 2),
    (["align", "{wav}", "{wav}", "--max-seconds", "inf"], 0),
    (["compare", "{wav}", "--reference", "{one}", "--methods", "wiener", "--no-metrics"], 2)],
    ids=lambda v: str(v[0]) if isinstance(v, list) else "")
def test_the_command_line_says_it_in_one_line(tmp_path, capsys, argv, rc):
    """room, align and compare printed a traceback, room and align with the exit status of a
    finding; pitch took 10 ms and vad found no speech."""
    from kudio.cli import main
    kudio.save_wave(tmp_path / "a.wav", VOICE, SR)
    kudio.save_wave(tmp_path / "one.wav", VOICE[:1], SR)
    assert main([part.format(wav=tmp_path / "a.wav", one=tmp_path / "one.wav", dir=tmp_path)
                 for part in argv]) == rc
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert rc == 0 or (err.startswith(f"kudio {argv[0]}: ") and len(err.strip().splitlines()) == 1)


# -- the plots -----------------------------------------------------------------------------------------

def test_as_many_rows_as_there_are_files(tmp_path):
    """Without subplots= there were no axes to draw on, and with 1 they were no matrix."""
    pytest.importorskip("matplotlib").use("Agg")
    import matplotlib.pyplot as plt
    from kudio.util.visual import WaveVisualizer
    for k in range(3):
        kudio.save_wave(tmp_path / f"{k}.wav", VOICE, SR)
    files = sorted(tmp_path.glob("*.wav"))
    for subplots, n in ((None, 1), (None, 3), (1, 1), (3, 2)):
        drawn = WaveVisualizer(subplots=subplots)
        drawn.plot(files[:n])
        assert np.shape(drawn.ax) == (subplots or n, 2)
        assert [drawn.ax[k][0].get_title() for k in range(n)] == [str(k) for k in range(n)]
        plt.close(drawn.fig)
    with pytest.raises(ValueError, match="3 files do not go into 1 row"):
        WaveVisualizer(subplots=1).plot(files)
    plt.close("all")


@pytest.mark.parametrize("what", ["nan", "empty"])
def test_a_spectrum_of_no_sound_is_not_drawn(tmp_path, what):
    """It was an empty pair of axes."""
    pytest.importorskip("matplotlib").use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    from kudio.util import visual
    y = one_bad(VOICE) if what == "nan" else VOICE[:0]
    sf.write(tmp_path / "bad.wav", y, SR, subtype="FLOAT")
    before = len(plt.get_fignums())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(kudio.FeatureError, match=r"^plot_PSD\(\)"):
            visual.plot_PSD(tmp_path / "bad.wav")
    figure, ax = plt.subplots()
    with pytest.raises(kudio.FeatureError, match=r"^plot_power_spectral_density\(\)"):
        visual.plot_power_spectral_density(ax, y, SR)
    plt.close(figure)
    assert len(plt.get_fignums()) == before
