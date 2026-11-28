# -*- coding: utf-8 -*-
"""The phase vocoder: a clip made longer or shorter that keeps its pitch.

Private: :func:`kudio.time_stretch` and :func:`kudio.pitch_shift` are what to call.

The clip's spectrum, frame by frame, is read at another speed and turned back into
samples [FG66]_, [LD99]_. Output frame *t* stands at ``t * rate`` among the frames of
the clip. Its magnitudes are what they are there, between the frame before and the
frame after in proportion; its phases go on from those of the output frame before by
what they turn between those two. So each bin goes on at the frequency it had, and
what is lost is how the bins stand to each other: a click is smeared over a frame,
and noise, whose frames no longer add up in step, comes out about 3 dB down.

A frame is 2048 samples and the hop a quarter of it, zeros past the ends of the clip,
as in librosa, whose ``time_stretch`` this is but for two things.

The phases are kept in float64. Through a clip they add up to a million radians, and
in float32, where librosa 0.11 keeps those of float32 audio, what a frame turns by is
rounded to what a number of that size resolves. It is the frequency of each bin that
is rounded, by another amount in each, and again every time the sum outgrows a power
of two: after five minutes at 48 kHz tones are up to 1.5 cents off, and in the seconds
in which they step they come apart.

And the end. The last output frames stand past the last frame of the clip, with none
after it. Its magnitudes are held there, and its phases turn on as they last did. librosa
0.11 takes a frame of zeros for the one after, which fades the end of a clip that was
speeded up; 1.0 draws the magnitudes on in a line and lets the phases stand, so the
overlapping frames work against each other (``tests/test_vocoder.py``).

References
----------
.. [FG66] J. L. Flanagan and R. M. Golden, "Phase vocoder", Bell System Technical
   Journal, vol. 45, no. 9, pp. 1493-1509, 1966,
   doi:10.1002/j.1538-7305.1966.tb01706.x.
.. [LD99] J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification
   of audio", IEEE Trans. Speech and Audio Processing, vol. 7, no. 3, pp. 323-332,
   1999, doi:10.1109/89.759041.
"""
from __future__ import annotations

import numpy as np

__all__ = ['SIZE', 'HOP', 'stretch']

#: samples in a frame, and from one to the next
SIZE, HOP = 2048, 512
#: how many output frames are made at a time: 256 KB of them, and 64 take a third longer
_BLOCK = 16


def stretch(y: np.ndarray, rate: float, length: int) -> np.ndarray:
    """*y*, 1-D, floating point and not empty, read *rate* times as fast, *length*
    samples of it: float64 of float64 and float32 of anything shorter."""
    from scipy.fft import irfft, rfft
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SIZE) / SIZE)
    out = np.zeros(length, np.float32 if y.dtype.itemsize < 8 else np.float64)
    # one frame past the clip, for a clip with no more than one of its own
    padded = np.pad(y, (SIZE // 2, SIZE // 2 + HOP))
    frames = np.lib.stride_tricks.sliding_window_view(padded, SIZE)
    last = len(y) // HOP                                     # the clip's last frame
    n_frames = min(int(np.ceil((last + 1) / rate)), -(-(length + SIZE // 2) // HOP))

    # A frame is four pieces a hop long, and the output is built up as rows of a hop:
    # row t has piece k of frame t - k, the oldest frame first, whatever the blocks.
    # The last three rows of a block have more to come, from the next.
    pieces = SIZE // HOP
    squared = (window ** 2).reshape(pieces, HOP)
    rows, energy = np.zeros((2, pieces - 1, HOP))
    phase = np.zeros(SIZE // 2 + 1)
    for start in range(0, n_frames, _BLOCK):
        position = np.arange(start, min(n_frames, start + _BLOCK)) * rate
        here = np.minimum(position.astype(np.int64), last)
        there = np.minimum(here + 1, last)
        turning = np.minimum(here, max(last - 1, 0))
        which, at = np.unique([here, there, turning, turning + 1], return_inverse=True)
        here, there, turning, turned = at.reshape(4, -1)
        spectrum = rfft(window * frames[which * HOP], axis=1)
        size, angle = np.abs(spectrum), np.angle(spectrum)

        part = np.minimum(position - which[here], 1)[:, np.newaxis]
        size = (1 - part) * size[here] + part * size[there]
        if not start:
            phase = angle[here[0]]
        angles = np.cumsum(np.vstack([phase, angle[turned] - angle[turning]]), axis=0)
        phase = angles[-1]

        spectrum = np.empty(size.shape, np.complex128)
        spectrum.real, spectrum.imag = size * np.cos(angles[:-1]), size * np.sin(angles[:-1])
        made = (window * irfft(spectrum, n=SIZE, axis=1)).reshape(-1, pieces, HOP)
        rows = np.concatenate([rows, np.zeros((len(made), HOP))])
        energy = np.concatenate([energy, np.zeros((len(made), HOP))])
        for k in range(pieces - 1, -1, -1):
            rows[k:k + len(made)] += made[:, k]
            energy[k:k + len(made)] += squared[k]
        # what no later frame reaches, and after the last block all of it
        done = len(made) if start + _BLOCK < n_frames else len(rows)
        _put(out, start * HOP - SIZE // 2, rows[:done].ravel(), energy[:done].ravel())
        rows, energy = rows[done:], energy[done:]
    return out


def _put(out: np.ndarray, at: int, total: np.ndarray, energy: np.ndarray) -> None:
    """Into *out*, from sample *at* on, what the windowed frames came to over what
    their windows did."""
    np.divide(total, energy, out=total, where=energy > np.finfo(np.float64).tiny)
    first, stop = max(0, -at), min(len(total), len(out) - at)
    if stop > first:
        with np.errstate(over='ignore'):            # the caller looks for what is not finite
            out[at + first:at + stop] = total[first:stop]
