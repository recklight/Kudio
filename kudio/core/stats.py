# -*- coding: utf-8 -*-
"""Confidence intervals and paired tests for results over a test set: the
mean of a per-utterance score, the difference between two systems scored on
the same utterances, and an error rate pooled over a corpus.

A number in a results table is a mean over the utterances of one test set,
and what a reader needs beside it is how far it would move on another set
drawn the same way. The intervals here answer that by resampling the test
set, and the test asks whether a difference between two systems is larger
than the luck of the draw:

* :func:`bootstrap_ci`, an interval for the mean of a per-item score;
* :func:`paired_ci`, an interval for the mean of system minus baseline, the
  two scored on the same items;
* :func:`rate_ci` and :func:`paired_rate_ci`, the same for a rate pooled
  over a corpus, total errors over total reference words, and for the
  difference of two systems' rates;
* :func:`permutation_test`, a paired sign-flip test of the mean difference;
* :func:`holm`, Holm's adjustment when several systems are tested against
  one baseline.

>>> ci = kudio.paired_ci(pesq_a, pesq_base, groups=speakers)            # doctest: +SKIP
>>> test = kudio.permutation_test(pesq_a, pesq_base, groups=speakers)   # doctest: +SKIP
>>> print(f"{ci.estimate:+.2f} [{ci.low:+.2f}, {ci.high:+.2f}]",
...       f"p = {test.pvalue:.4f}")                                    # doctest: +SKIP

**What is resampled.** Without *groups* each item is drawn on its own,
which assumes the items are independent. Utterances of one speaker are not:
an enhancer that suits a voice suits it in every utterance, and a
recogniser that stumbles on an accent stumbles in every sentence. The mean
then moves with the speakers drawn as well as with the utterances, and
resampling utterances misses that part. With *groups*, one label per item,
I resample whole groups with replacement and keep each group's items
together. Field and Welsh [FW07] show that this cluster bootstrap
estimates the variance consistently under both the random-effect and the
transformation model of clustered data. In kudio's simulation
(``tools/validation/stats.py``), speakers of 10 to 30 utterances with an
intra-class correlation of 0.3, 95% intervals that resampled utterances
held the true mean in about half the data sets, 52% at 10 speakers and 50%
at 30; resampling speakers, 89% and 92%. For a WER pooled over 30 speakers
of 3 to 12 utterances, each speaker with an error rate of its own, it was
67% against 93%. Each of these shares comes from 2000 simulated data sets,
so it is good to about 1 point near a half and 0.5 near 95% (one standard
error). Group by what the claim is to generalize over: speakers as a rule,
but a recording session or a sentence read by many talkers can be a group
too. Where two factors are both drawn at random and crossed, every speaker
reading the same sentences, resampling speakers alone leaves out what the
sentences add to the variance; Owen's pigeonhole bootstrap [O07]
resamples the two separately, and kudio does not do that.

**How many groups.** Resampling can only reshuffle the groups at hand, and
a bootstrap interval for the mean of k independent values falls short of
its level when k is small. In the same simulation a 95% interval over 10
speakers held the true mean 89% of the time, over 5 83% and over 2 50%,
close to the 90%, 85% and 50% of 10, 5 or 2 independent values. Under
:data:`FEW_UNITS` groups, or as many items when there are no groups, the
intervals warn. VoiceBank-DEMAND's test set has two speakers: an interval
over them runs between the two speakers' own means, and the honest reading
is that two speakers cannot say how a third would fare. Report the number
of groups with every interval.

**Error rates.** A WER, CER or DER is a ratio of sums, total errors over
total reference words or seconds, and not a mean of per-utterance rates,
which would weigh a two-word utterance as much as a forty-word one.
:func:`rate_ci` therefore adds up the errors and the totals of every
resample afresh and divides, as Bisani and Ney do for the WER [BN04]. An
item whose total is 0, an empty reference, adds its errors and nothing to
the total, as it does in :func:`kudio.wer`. Two systems scored against the
same references share the totals, so the difference of their rates is the
summed difference of their errors over the summed totals, and
:func:`paired_rate_ci` resamples that.

**The interval.** The percentile interval takes the alpha/2 and
1 - alpha/2 quantiles of the resampled values, with numpy's linear
interpolation. BCa, the default, shifts both ends to correct for the bias
and the skew of the resampled values [E87], as Efron and Tibshirani set it
out ([ET93] Ch. 14). z0 comes from the share of resampled values below
the estimate, counting a value equal to it as half, as
``scipy.stats.bootstrap`` does; the acceleration from the jackknife,
``sum(d**3) / (6 * sum(d**2)**1.5)`` with ``d`` the mean of the
leave-one-out estimates minus each of them. With groups the jackknife
leaves out one group at a time, so that it runs over the same units the
bootstrap draws. I compute the estimate less the estimate without unit j as
``(num_j - estimate * den_j) / (D - den_j)``, which subtracts no two nearly
equal ratios. On skewed data BCa keeps closer to its level: on 20 lognormal
values the percentile interval held the mean 86% of the time and BCa 88%;
at 200 values both held it 94%, and on normal data the two hardly differ.

The resamples are drawn with ``rng.integers(0, n, (n_resamples, n))``, the
call ``scipy.stats.bootstrap`` makes, so with the same seed the two draw the
same resamples, and the percentile ends equal ``np.percentile`` over
scipy's ``bootstrap_distribution`` exactly. scipy up to 1.16 reads its own
ends that way; 1.18 reads them with ``scipy.stats.quantile``, which lands
within about 1e-15 of the width. The BCa ends agree with scipy's to 1.3e-14
of the interval's width (2.4e-14 with scipy 1.18), except where some
resamples hold each item once, 3.8% of them at 5 items and 0.24% at 8.
Their mean is the estimate, but summed in another order it can miss it in
the last bit, and scipy then counts each of them below or above the
estimate as the rounding falls. I count them as equal to it, within the
rounding the sum allows; at 5 items that moved scipy's BCa ends by up to
7.7% of the width.

**Cases with no interval.** I follow the rule kudio's metrics follow:
input that is malformed raises ValueError, its message starting with the
function's name, and input that is well formed but has no interval gives
nan with one warning. The warning ends "; the interval is nan" or "; the
p-value is nan" rather than the metrics' "; scored nan", since what is
missing is not a score. With fewer than 2 groups, or fewer than 2 items
when there are no groups, there is no interval. Where every resample gives
the estimate, all values equal for instance, the interval is the estimate
at both ends, with a warning: that the items at hand do not vary says
nothing about how far the mean could move on others, and a WER of 0 over
500 utterances does not make the WER of the next 500 certain to be 0.
Where BCa has no interval, the estimate lying beyond every resampled value
or the skew correction running past its range, the percentile interval
takes its place with a warning, and ``method`` in the result says
``'percentile'``. A NaN among the values is refused, as the metrics refuse
NaN: it is usually an item a metric could not score, and dropping it from
one list and not the other would pair the rest wrongly.

**The test.** :func:`permutation_test` tests the null hypothesis that the
two systems are exchangeable: that swapping which output is the system's
and which the baseline's leaves the scores' distribution unchanged. A swap
flips the sign of that item's difference, so the null distribution of the
mean difference comes from flipping signs. Without groups every item flips
on its own, which is right when the items are independent. With groups a
group's items flip together, which for the mean difference is the same as
flipping the sign of each group's sum; it keeps whatever ties a speaker's
utterances together and asks only that the systems be exchangeable speaker
by speaker. On the clustered data above with no true difference, flipping
utterances one by one rejected at 0.05 in 48% of the data sets at 10
speakers and 46% at 30; flipping speakers, in 5.3% and 4.6%.

When ``2**units <= n_resamples``, units being the groups or else the items,
the test runs through every sign pattern and the p-value is the share of
them whose statistic is at least as extreme as the observed one, the
observed pattern included; ``scipy.stats.permutation_test`` gives the same
p-values to the last bit on 2 to 16 items, in scipy 1.13 and in 1.18, but
for the exception at the end of this paragraph. Past
that it draws *n_resamples* patterns at random and takes
``(b + 1) / (n_resamples + 1)``, b being the drawn patterns at least as
extreme. Phipson and Smyth [PS10] show why the observed pattern belongs in
the count: without it the p-value is too small by about 1/n_resamples and
can be 0. Two-sided, "as extreme" means as large in absolute value. scipy
doubles the smaller one-sided p-value instead; over every pattern the two
agree, since the null of a sign flip is symmetric, and over random patterns
they are two estimates of the same p. A pattern whose statistic equals the
observed one counts as at least as extreme even when rounding puts it a
hair below: within scipy's tolerance of 100 eps times the observed value,
or within the rounding a sum of these terms allows, whichever is larger.
Where the observed difference is 0, scipy's tolerance is 0 as well, and
its one-sided p-value follows the rounding; on differences of a few tenths
kudio's then matches an enumeration in exact fractions and scipy's does
not.

Over G groups the smallest two-sided p-value is 2 / 2**G: 0.0625 at 5
groups, so no data can bring that test under 0.05. Over patterns drawn at
random it is 1 / (n_resamples + 1). The test warns where its smallest
p-value is above 0.05, and where it draws fewer than 1000 patterns.

**Several systems.** With five systems each tested at 0.05 against one
baseline, the chance of at least one false finding is 23% if the five tests
are independent, and up to 25% if they are not. :func:`holm` adjusts the
p-values by Holm's step-down procedure [H79], which keeps that chance at
the level whatever the dependence among the tests, so it holds for systems
scored on the same utterances, and it rejects everything Bonferroni's
correction rejects and possibly more.

What to watch for:

* *system*, *baseline* and *groups* pair up by position: item i of each
  has to be the same utterance.
* An interval and a test on the same data can disagree near the edge, the
  interval just excluding 0 with p a little above 0.05 or the other way
  round; they are built differently. Quote each for what it is.
* The seed fixes the resamples, so a result can be reproduced. Another seed
  moves the ends of an interval from 10,000 resamples by about 1% of its
  width (one standard deviation), and by about 2% at the long end of
  skewed data.
* The interval covers the draw of the test set, not of the system: two
  training runs of one model differ as well, and resampling utterances says
  nothing about that.

The sources::

    [E87]  B. Efron, "Better bootstrap confidence intervals", Journal of the
           American Statistical Association 82(397), pp. 171-185, 1987.
           doi:10.1080/01621459.1987.10478410
    [ET93] B. Efron and R. J. Tibshirani, "An Introduction to the
           Bootstrap", Chapman & Hall, 1993. doi:10.1007/978-1-4899-4541-9
    [BN04] M. Bisani and H. Ney, "Bootstrap estimates for confidence
           intervals in ASR performance evaluation", Proc. ICASSP 2004,
           vol. 1, pp. I-409-412. doi:10.1109/ICASSP.2004.1326009
    [FW07] C. A. Field and A. H. Welsh, "Bootstrapping clustered data",
           Journal of the Royal Statistical Society Series B 69(3),
           pp. 369-390, 2007. doi:10.1111/j.1467-9868.2007.00593.x
    [O07]  A. B. Owen, "The pigeonhole bootstrap", The Annals of Applied
           Statistics 1(2), pp. 386-411, 2007. doi:10.1214/07-AOAS122
    [PS10] B. Phipson and G. K. Smyth, "Permutation p-values should never be
           zero: calculating exact p-values when permutations are randomly
           drawn", Statistical Applications in Genetics and Molecular
           Biology 9(1), Article 39, 2010. doi:10.2202/1544-6115.1585
    [H79]  S. Holm, "A simple sequentially rejective multiple test
           procedure", Scandinavian Journal of Statistics 6, pp. 65-70,
           1979.

``tools/validation/stats.py`` reruns every comparison with scipy and every
figure of the simulation above. An interval from 10,000 resamples of 824
utterances takes about 50 ms on a desktop CPU.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from kudio.core._policy import warn

__all__ = ['ConfidenceInterval', 'PermutationResult', 'bootstrap_ci', 'paired_ci',
           'rate_ci', 'paired_rate_ci', 'permutation_test', 'holm', 'FEW_UNITS']

#: Fewer groups than this, or fewer items when there are no groups, and an
#: interval warns that it says little; the module docstring has why.
FEW_UNITS = 10

#: Fewer resamples than this, and an interval warns that its ends rest on a
#: handful of them.
FEW_RESAMPLES = 1000

#: How many values one step of the resampling gathers at most, 8 MB of
#: float64, whatever the size of the test set.
_CHUNK = 1 << 20

_METHODS = ('bca', 'percentile')
_ALTERNATIVES = ('two-sided', 'greater', 'less')


class ConfidenceInterval(NamedTuple):
    """A bootstrap confidence interval and what it was computed from."""

    #: the mean, rate or difference over all the items
    estimate: float
    #: lower end of the interval
    low: float
    #: upper end of the interval
    high: float
    #: the level asked for, 0.95 for a 95% interval
    confidence: float
    #: items scored
    n: int
    #: groups the items were resampled in, None when they were resampled one
    #: by one
    n_groups: Optional[int]
    #: resamples drawn
    n_resamples: int
    #: ``'bca'`` or ``'percentile'``: the method that gave the ends, which is
    #: ``'percentile'`` where BCa had no interval to give
    method: str


class PermutationResult(NamedTuple):
    """A paired sign-flip test and what it was computed from."""

    #: the mean difference, system minus baseline; the difference of the two
    #: pooled rates when totals were given
    statistic: float
    #: the p-value
    pvalue: float
    #: items scored
    n: int
    #: groups whose items flipped together, None when every item flipped on
    #: its own
    n_groups: Optional[int]
    #: sign patterns the p-value counts over: all ``2**units`` of them when
    #: *exact*, else the number drawn
    n_resamples: int
    #: whether every sign pattern was run through
    exact: bool
    #: ``'two-sided'``, ``'greater'`` or ``'less'``
    alternative: str


# --- reading the input --------------------------------------------------------

#: why a NaN among per-item values is refused
_NAN_HINT = ("A metric gives NaN for an item it could not score, and dropping it "
             "from one list and not the other would pair the rest wrongly; drop "
             "those items from every list passed here, groups included, and say "
             "how many were dropped")


def _vector(values, what: str, name: str, hint: str = _NAN_HINT) -> np.ndarray:
    """*values* as a one-dimensional float64 array of finite numbers; a
    ``(n, 1)`` or ``(1, n)`` array is read as the n values it holds."""
    if isinstance(values, np.ma.MaskedArray) and np.ma.getmaskarray(values).any():
        masked = int(np.count_nonzero(np.ma.getmaskarray(values)))
        raise ValueError(f"{name}(): {what} is a masked array with {masked} masked "
                         f"value(s), which np.asarray would read at whatever the mask "
                         f"hides; drop those items from every list first")
    try:
        values = np.asarray(values)
    except ValueError as e:                     # rows of different lengths
        raise ValueError(f"{name}(): {what} has to be one number per item, in a "
                         f"one-dimensional sequence; {e}") from None
    if values.ndim > 1 and sum(size != 1 for size in values.shape) <= 1:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"{name}(): {what} has to be one number per item, in a "
                         f"one-dimensional sequence; got shape {values.shape}")
    if values.dtype.kind == 'O':
        # astype(float) would read '0.5' as 0.5 and None as nan without a word
        items = values.tolist()
        for item in items:
            if isinstance(item, (str, bytes)) or not isinstance(item, (numbers.Real,
                                                                        np.bool_)):
                raise ValueError(f"{name}(): {what} has to hold real numbers, got "
                                 f"{item!r} among them")
        values = np.asarray(items, dtype=np.float64)
    if values.dtype.kind not in 'biuf':
        raise ValueError(f"{name}(): {what} has to hold real numbers, got an array "
                         f"of {values.dtype}")
    values = values.astype(np.float64)
    bad = int(np.count_nonzero(~np.isfinite(values)))
    if bad:
        raise ValueError(f"{name}(): {bad} of the {values.size} {what} "
                         f"{'is' if bad == 1 else 'are'} NaN or infinite. {hint}")
    return values


def _same_length(first: np.ndarray, second: np.ndarray, what: Tuple[str, str],
                 name: str) -> None:
    """Refuse two per-item lists of different lengths; *what* names one item
    of each."""
    if first.size != second.size:
        raise ValueError(f"{name}(): {_plural(first.size, what[0])} and "
                         f"{_plural(second.size, what[1])}; they pair up by position, "
                         f"one of each per item")


def _in_range(values: np.ndarray, what: str, name: str) -> np.ndarray:
    """*values*, refused where sums over resamples of them could pass the
    largest float64: a resample of groups can hold the largest group n
    times, so n**2 times the largest value has to stay finite."""
    if values.size:
        largest = float(np.max(np.abs(values)))
        if not math.isfinite(largest * values.size * values.size):
            raise ValueError(f"{name}(): {what} reach {largest:.3g}, and summed over "
                             f"resamples of {values.size} items they would pass the "
                             f"largest float64, 1.8e308; scale them down first")
    return values


def _difference(system: np.ndarray, baseline: np.ndarray, name: str) -> np.ndarray:
    """system - baseline, refused where it or its sums leave float64."""
    with np.errstate(over='ignore', invalid='ignore'):
        difference = system - baseline
    return _in_range(difference, 'the differences', name)


def _counts(values, what: str, name: str) -> np.ndarray:
    """Error counts or totals: finite and not negative."""
    values = _vector(values, what, name)
    negative = values < 0
    if np.any(negative):
        raise ValueError(f"{name}(): {what} cannot be negative, got "
                         f"{float(values[np.argmax(negative)])!r} among them")
    return values


def _group_index(groups, n: int, name: str) -> Tuple[np.ndarray, int]:
    """For each item, the number of its group, counted in the order the groups
    first appear; and how many groups there are. Numbering by first
    appearance makes groups of one item each draw the same resamples as no
    groups at all.

    A list is read label by label, as given: np.asarray would turn [1, '1']
    into two equal strings, and a tuple label into a row of a 2-d array.
    """
    if isinstance(groups, (str, bytes)):
        raise ValueError(f"{name}(): groups has to be one label per item, got a "
                         f"single {type(groups).__name__}")
    if isinstance(groups, np.ndarray):
        if groups.ndim > 1 and sum(size != 1 for size in groups.shape) <= 1:
            groups = groups.reshape(-1)
        if groups.ndim != 1:
            raise ValueError(f"{name}(): groups has to be one label per item, in a "
                             f"one-dimensional sequence; got shape {groups.shape}")
        labels = groups.tolist()
    else:
        try:
            labels = list(groups)
        except TypeError:
            raise ValueError(f"{name}(): groups has to be one label per item, got "
                             f"{type(groups).__name__}") from None
    if len(labels) != n:
        raise ValueError(f"{name}(): {_plural(len(labels), 'group label')} for "
                         f"{_plural(n, 'item')}; every item needs the label of its "
                         f"group, a speaker id say")
    first: Dict[Any, int] = {}
    index = np.empty(n, dtype=np.intp)
    for i, label in enumerate(labels):
        # a NaN is not equal to itself, so each NaN would be a group of its
        # own; pd.NA cannot even say whether it is
        try:
            missing = label is None or bool(label != label)
        except TypeError:
            missing = True
        except ValueError:                       # an array, which answers per element
            missing = False
        if missing:
            raise ValueError(f"{name}(): item {i} has {label!r} for its group; every "
                             f"item needs the label of its group, a speaker id say")
        try:
            index[i] = first.setdefault(label, len(first))
        except TypeError:
            raise ValueError(f"{name}(): group labels have to be numbers, text or "
                             f"tuples of them, got {label!r}") from None
    return index, len(first)


def _confidence(value, name: str) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not 0.0 < float(value) < 1.0):
        raise ValueError(f"{name}(): confidence has to be a level strictly between 0 "
                         f"and 1, 0.95 for a 95% interval; got {value!r}")
    return float(value)


def _resamples(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or value < 1):
        raise ValueError(f"{name}(): n_resamples has to be a positive whole number, "
                         f"got {value!r}")
    return int(value)


def _generator(seed, name: str) -> np.random.Generator:
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}(): seed has to be None, a non-negative integer or "
                         f"a numpy Generator; {e}") from None


def _rounding(units: int, magnitude: float) -> float:
    """How far apart two sums of the same *units* terms, added in different
    orders, can land: (units - 1) eps times the sum of the terms'
    magnitudes, with room to spare."""
    return 4.0 * units * np.finfo(float).eps * magnitude


def _plural(count: int, word: str) -> str:
    return f"{count} {word if count == 1 else word + 's'}"


# --- the bootstrap ------------------------------------------------------------

def _resampled(num: np.ndarray, den: np.ndarray, n_resamples: int,
               rng: np.random.Generator) -> np.ndarray:
    """sum(num) / sum(den) over *n_resamples* resamples of the units, each
    drawn with replacement."""
    units = num.size
    ones = bool(np.all(den == 1.0))
    rows = max(1, _CHUNK // units)
    out = np.empty(n_resamples)
    with np.errstate(divide='ignore', invalid='ignore'):
        for start in range(0, n_resamples, rows):
            stop = min(start + rows, n_resamples)
            # a call per chunk draws the same numbers as one call for all of
            # them, and one call for all of them is what scipy.stats.bootstrap
            # makes
            pick = rng.integers(0, units, (stop - start, units))
            top = num[pick].sum(axis=-1)
            out[start:stop] = top / units if ones else top / den[pick].sum(axis=-1)
    return out


def _acceleration(num: np.ndarray, den: np.ndarray) -> Optional[float]:
    """BCa's acceleration from the jackknife that leaves out one unit at a
    time ([ET93] Ch. 14), or None when leaving one out can leave no total
    to divide by."""
    rest = den.sum() - den
    if np.any(rest <= 0):
        return None
    theta = num.sum() / den.sum()
    lift = (num - theta * den) / rest           # estimate - estimate without j
    spread = lift - lift.mean()                 # mean of the jackknife - each
    square = float(np.sum(spread ** 2))
    if square == 0.0:
        return 0.0
    return float(np.sum(spread ** 3)) / (6.0 * square ** 1.5)


def _bca_levels(theta_b: np.ndarray, theta: float, num: np.ndarray, den: np.ndarray,
                alpha: float, tie: float) -> Union[Tuple[float, float], str]:
    """The levels BCa reads the two ends at, or why it has none."""
    from scipy.special import ndtr, ndtri

    below = (np.count_nonzero(theta_b < theta - tie)
             + np.count_nonzero(theta_b <= theta + tie))
    z0 = float(ndtri(below / (2 * theta_b.size)))
    if not math.isfinite(z0):
        return "the estimate lies beyond every resampled value"
    a = _acceleration(num, den)
    if a is None:
        return ("the totals all sit in one group or item, and the jackknife "
                "without it has nothing to divide by")
    z_alpha = float(ndtri(alpha))
    levels = []
    for z_end in (z_alpha, -z_alpha):
        lift = z0 + z_end
        rest = 1 - a * lift
        if not rest > 0:
            return (f"the skew correction runs past its range (acceleration "
                    f"{a:.3g}, bias {z0:.3g})")
        levels.append(float(ndtr(z0 + lift / rest)))
    return levels[0], levels[1]


def _few_units(name: str, units: int, kind: str) -> str:
    return (f"{name}(): {units} {kind} are too few for this interval to mean much. "
            f"Resampling can only reshuffle these {units}, and in kudio's "
            f"simulations a 95% interval over 2 held the true value in half the "
            f"data sets, over 5 in 83% and over 10 in 89%. Report the number of "
            f"{kind} beside it")


def _interval(numerator: np.ndarray, denominator: np.ndarray, estimate: float, groups,
              confidence, n_resamples, method, seed, name: str,
              unscorable: Optional[str] = None) -> ConfidenceInterval:
    """The interval every function above shares: the ratio of sums, over
    groups when *groups* is given and over items otherwise."""
    confidence = _confidence(confidence, name)
    n_resamples = _resamples(n_resamples, name)
    if method not in _METHODS:
        raise ValueError(f"{name}(): method has to be one of {_METHODS}, got {method!r}")
    rng = _generator(seed, name)
    n = numerator.size
    if groups is None:
        num, den, n_groups = numerator, denominator, None
    else:
        index, n_groups = _group_index(groups, n, name)
        num = np.bincount(index, weights=numerator, minlength=n_groups)
        den = np.bincount(index, weights=denominator, minlength=n_groups)

    def result(low, high, used=method):
        return ConfidenceInterval(float(estimate), float(low), float(high), confidence,
                                  n, n_groups, n_resamples, used)

    nan = float('nan')
    units = num.size
    if unscorable is None and units < 2:
        if n == 0:
            unscorable = "there are no items"
        elif n == 1:
            unscorable = "there is 1 item, and every resample of it is that item again"
        else:
            unscorable = (f"all {n} items are in one group, and every resample of "
                          f"one group is that group again")
    if unscorable is not None:
        warn(f"{name}(): {unscorable}; the interval is nan")
        return result(nan, nan)
    kind = 'groups' if n_groups is not None else 'items'
    ones = bool(np.all(den == 1.0))
    total = units if ones else den.sum()
    theta = num.sum() / total
    theta_b = _resampled(num, den, n_resamples, rng)
    empty = int(np.count_nonzero(~np.isfinite(theta_b)))
    if empty:
        warn(f"{name}(): {empty} of the {n_resamples} resamples drew only {kind} "
             f"whose totals are 0, and have no rate; the interval is nan")
        return result(nan, nan)
    # a resample holding each unit once gives the estimate, but summed in
    # another order it can miss it in the last bits
    tie = _rounding(units, float(np.sum(np.abs(num))) / total + abs(theta))
    if np.all(np.abs(theta_b - theta) <= tie):
        warn(f"{name}(): every resample gives the estimate, so the interval has no "
             f"width; that these {units} {kind} do not vary says nothing about how "
             f"far the value could move on others")
        return result(estimate, estimate)
    if units < FEW_UNITS:
        warn(_few_units(name, units, kind))
    if n_resamples < FEW_RESAMPLES:
        warn(f"{name}(): with {_plural(n_resamples, 'resample')} each end of the "
             f"interval rests on the few beyond it, and another seed would move it; "
             f"use at least {FEW_RESAMPLES} (the default is 10000)")
    alpha = (1 - confidence) / 2
    levels: Tuple[float, float] = (alpha, 1 - alpha)
    used = method
    if method == 'bca':
        bca = _bca_levels(theta_b, theta, num, den, alpha, tie)
        if isinstance(bca, str):
            warn(f"{name}(): BCa has no interval here: {bca}; gave the percentile "
                 f"interval instead")
            used = 'percentile'
        else:
            levels = bca
    low = np.percentile(theta_b, levels[0] * 100)
    high = np.percentile(theta_b, levels[1] * 100)
    return result(low, high, used)


def bootstrap_ci(values, *, groups=None, confidence: float = 0.95,
                 n_resamples: int = 10000, method: str = 'bca',
                 seed=0) -> ConfidenceInterval:
    """Confidence interval for the mean of a per-item score.

    >>> ci = kudio.bootstrap_ci(pesq_scores, groups=speakers)      # doctest: +SKIP
    >>> print(f"{ci.estimate:.2f} [{ci.low:.2f}, {ci.high:.2f}]")    # doctest: +SKIP

    The estimate is the mean over all the items, so with groups of
    different sizes a large group weighs more, as it does in the mean
    itself; each resample divides the summed scores of the groups it drew
    by the number of items in them.

    :param values: one score per item: PESQ per utterance, say, or 0/1 for
        an utterance recognised wholly right.
    :param groups: one label per item, a speaker id say, to resample whole
        groups instead of single items; numbers or text. None (default)
        resamples the items one by one.
    :param confidence: the level, strictly between 0 and 1: 0.95 (default)
        for a 95% interval.
    :param n_resamples: how many resamples to draw, 10000 by default; under
        1000 it warns.
    :param method: ``'bca'`` (default), the bias-corrected and accelerated
        interval, or ``'percentile'``.
    :param seed: seeds numpy's default generator, 0 by default, so that a
        result can be reproduced; None for fresh entropy, or a
        ``numpy.random.Generator``, which is used and moved on.
    :raises ValueError: for values or labels that are not one per item, a
        value that is NaN or infinite, and parameters out of range.
    :returns: a :class:`ConfidenceInterval`. Its ends are nan, with a
        warning, for fewer than 2 groups, or fewer than 2 items when there
        are no groups.
    """
    name = 'bootstrap_ci'
    values = _in_range(_vector(values, 'values', name), 'values', name)
    estimate = float(np.mean(values)) if values.size else float('nan')
    return _interval(values, np.ones(values.size), estimate, groups, confidence,
                     n_resamples, method, seed, name)


def paired_ci(system, baseline, *, groups=None, confidence: float = 0.95,
              n_resamples: int = 10000, method: str = 'bca',
              seed=0) -> ConfidenceInterval:
    """Confidence interval for the mean difference between two systems scored
    on the same items, system minus baseline.

    >>> ci = kudio.paired_ci(pesq_a, pesq_base, groups=speakers)   # doctest: +SKIP

    Each resample draws items, or groups, and takes both systems' scores of
    what it drew, which is what makes the interval narrower than one from
    two independent intervals: what an utterance does to both systems alike
    cancels in the difference.

    :param system: one score per item for the system.
    :param baseline: one score per item for the baseline, item i the same
        utterance as in *system*.
    :param groups: as :func:`bootstrap_ci`.
    :param confidence: as :func:`bootstrap_ci`.
    :param n_resamples: as :func:`bootstrap_ci`.
    :param method: as :func:`bootstrap_ci`.
    :param seed: as :func:`bootstrap_ci`.
    :raises ValueError: as :func:`bootstrap_ci`, and for *system* and
        *baseline* of different lengths.
    :returns: a :class:`ConfidenceInterval` of the mean of system - baseline.
    """
    name = 'paired_ci'
    system = _vector(system, 'system scores', name)
    baseline = _vector(baseline, 'baseline scores', name)
    _same_length(system, baseline, ('system score', 'baseline score'), name)
    difference = _difference(system, baseline, name)
    estimate = float(np.mean(difference)) if difference.size else float('nan')
    return _interval(difference, np.ones(difference.size), estimate, groups, confidence,
                     n_resamples, method, seed, name)


def _rate_interval(errors: np.ndarray, totals: np.ndarray, groups, confidence,
                   n_resamples, method, seed, name: str) -> ConfidenceInterval:
    _same_length(errors, totals, ('error count', 'total'), name)
    _in_range(errors, 'errors', name)
    _in_range(totals, 'totals', name)
    total = float(totals.sum())
    if errors.size and total == 0.0:
        return _interval(errors, totals, float('nan'), groups, confidence, n_resamples,
                         method, seed, name,
                         unscorable="the totals add up to 0, so there is no rate")
    estimate = float(errors.sum()) / total if errors.size else float('nan')
    return _interval(errors, totals, estimate, groups, confidence, n_resamples, method,
                     seed, name)


def rate_ci(errors, totals, *, groups=None, confidence: float = 0.95,
            n_resamples: int = 10000, method: str = 'bca',
            seed=0) -> ConfidenceInterval:
    """Confidence interval for an error rate pooled over a corpus,
    ``sum(errors) / sum(totals)``: a WER, a CER or a DER.

    >>> counts = [kudio.word_errors(r, h) for r, h in zip(refs, hyps)]   # doctest: +SKIP
    >>> ci = kudio.rate_ci([c.errors for c in counts],
    ...                    [c.reference_length for c in counts],
    ...                    groups=speakers)                              # doctest: +SKIP

    Every resample adds up the errors and the totals of the items, or
    groups, it drew and divides the one by the other, so the estimate is the
    corpus rate :func:`kudio.wer` gives, not a mean of per-utterance rates.

    :param errors: errors per item: substitutions, deletions and insertions
        for a WER, or missed, false-alarm and confused seconds for a DER.
        Not negative; they need not be whole numbers.
    :param totals: what each item's errors are counted against: reference
        words, characters or seconds. Not negative; an item with a total of
        0 adds its errors and nothing else.
    :param groups: as :func:`bootstrap_ci`.
    :param confidence: as :func:`bootstrap_ci`.
    :param n_resamples: as :func:`bootstrap_ci`.
    :param method: as :func:`bootstrap_ci`.
    :param seed: as :func:`bootstrap_ci`.
    :raises ValueError: as :func:`bootstrap_ci`, and for negative counts or
        lists of different lengths.
    :returns: a :class:`ConfidenceInterval` of the rate, as a fraction.
        nan, with a warning, where the totals add up to 0, and where a
        resample draws only items or groups whose totals are 0.
    """
    name = 'rate_ci'
    errors = _counts(errors, 'errors', name)
    totals = _counts(totals, 'totals', name)
    return _rate_interval(errors, totals, groups, confidence, n_resamples, method, seed,
                          name)


def paired_rate_ci(system_errors, baseline_errors, totals, *, groups=None,
                   confidence: float = 0.95, n_resamples: int = 10000,
                   method: str = 'bca', seed=0) -> ConfidenceInterval:
    """Confidence interval for the difference of two systems' pooled error
    rates on the same items, system minus baseline.

    >>> ci = kudio.paired_rate_ci(errors_a, errors_base, words,
    ...                           groups=speakers)                       # doctest: +SKIP

    Both systems are counted against the same references, so they share the
    totals, and the difference of the two rates is ``sum(system_errors -
    baseline_errors) / sum(totals)``; every resample recomputes that. A
    negative difference is a system with fewer errors.

    :param system_errors: errors per item for the system.
    :param baseline_errors: errors per item for the baseline, item i the
        same utterance.
    :param totals: reference words, characters or seconds per item, as
        :func:`rate_ci` takes them.
    :param groups: as :func:`bootstrap_ci`.
    :param confidence: as :func:`bootstrap_ci`.
    :param n_resamples: as :func:`bootstrap_ci`.
    :param method: as :func:`bootstrap_ci`.
    :param seed: as :func:`bootstrap_ci`.
    :raises ValueError: as :func:`rate_ci`.
    :returns: a :class:`ConfidenceInterval` of the difference of the rates.
    """
    name = 'paired_rate_ci'
    system_errors = _counts(system_errors, 'system errors', name)
    baseline_errors = _counts(baseline_errors, 'baseline errors', name)
    totals = _counts(totals, 'totals', name)
    _same_length(system_errors, baseline_errors,
                 ('system error count', 'baseline error count'), name)
    return _rate_interval(system_errors - baseline_errors, totals, groups, confidence,
                          n_resamples, method, seed, name)


# --- the sign-flip test -------------------------------------------------------

def _flipped(sums: np.ndarray, denominator: float, exact: bool, count: int,
             rng: np.random.Generator) -> np.ndarray:
    """The statistic under *count* sign patterns: every pattern, in order,
    when *exact*, else patterns drawn at random."""
    units = sums.size
    rows = max(1, _CHUNK // units)
    shifts = np.arange(units, dtype=np.int64)
    out = np.empty(count)
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        if exact:
            bits = (np.arange(start, stop, dtype=np.int64)[:, None] >> shifts) & 1
        else:
            bits = rng.integers(0, 2, (stop - start, units), dtype=np.int8)
        out[start:stop] = ((1.0 - 2.0 * bits) * sums).sum(axis=-1) / denominator
    return out


def permutation_test(system, baseline, *, groups=None, totals=None,
                     n_resamples: int = 10000, alternative: str = 'two-sided',
                     seed=0) -> PermutationResult:
    """Paired sign-flip test of the mean difference between two systems
    scored on the same items, system minus baseline.

    >>> kudio.permutation_test(pesq_a, pesq_base, groups=speakers)       # doctest: +SKIP
    >>> kudio.permutation_test(errors_a, errors_base, totals=words,
    ...                        groups=speakers, alternative='less')      # doctest: +SKIP

    The null hypothesis is that the two systems are exchangeable, item by
    item or, with *groups*, group by group; the module docstring has why a
    group's items flip together. With every sign pattern run through, the
    p-value is exact; with patterns drawn at random it is
    ``(b + 1) / (n_resamples + 1)``.

    :param system: one score, or error count, per item for the system.
    :param baseline: the same for the baseline, item i the same utterance.
    :param groups: one label per item, a speaker id say, whose items flip
        together. None (default) flips every item on its own.
    :param totals: reference words, characters or seconds per item, when
        the scores are error counts: the statistic is then the difference of
        the two pooled rates. Their sum is the same under every flip, so the
        p-value is that of the summed error counts. None (default) tests the
        mean difference.
    :param n_resamples: the most sign patterns to evaluate, 10000 by
        default. When the ``2**units`` patterns of all the units, groups or
        else items, are no more than this, every one is run through and the
        p-value is exact; otherwise this many are drawn at random.
    :param alternative: ``'two-sided'`` (default); ``'greater'`` for a
        system that scores higher than the baseline; ``'less'`` for one that
        scores lower, which is better for an error rate.
    :param seed: as :func:`bootstrap_ci`; an exact test draws nothing.
    :raises ValueError: for scores, labels or totals that are not one per
        item, a value that is NaN or infinite, negative totals, and
        parameters out of range.
    :returns: a :class:`PermutationResult`. nan, with a warning, for no items
        or totals that add up to 0. Where no possible p-value is under 0.05,
        5 groups two-sided for instance, it warns and still gives the
        p-value.
    """
    name = 'permutation_test'
    system = _vector(system, 'system scores', name)
    baseline = _vector(baseline, 'baseline scores', name)
    _same_length(system, baseline, ('system score', 'baseline score'), name)
    n_resamples = _resamples(n_resamples, name)
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"{name}(): alternative has to be one of {_ALTERNATIVES}, "
                         f"got {alternative!r}")
    rng = _generator(seed, name)
    difference = _difference(system, baseline, name)
    n = difference.size
    if totals is None:
        denominator = float(n)
    else:
        totals = _counts(totals, 'totals', name)
        _same_length(difference, totals, ('score', 'total'), name)
        denominator = float(_in_range(totals, 'totals', name).sum())
    if groups is None:
        sums, n_groups = difference, None
    else:
        index, n_groups = _group_index(groups, n, name)
        sums = np.bincount(index, weights=difference, minlength=n_groups)
    nan = float('nan')
    if n == 0 or denominator == 0.0:
        why = "there are no items" if n == 0 else "the totals add up to 0"
        warn(f"{name}(): {why}; the p-value is nan")
        return PermutationResult(nan, nan, n, n_groups, 0, False, alternative)
    statistic = float(difference.sum()) / denominator
    units = sums.size
    exact = (1 << units) <= n_resamples
    count = (1 << units) if exact else n_resamples
    kind = 'group' if n_groups is not None else 'item'
    if exact:
        smallest = math.ldexp(2.0 if alternative == 'two-sided' else 1.0, -units)
        if smallest > 0.05:
            warn(f"{name}(): {_plural(units, kind)} "
                 f"{'allows' if units == 1 else 'allow'} {1 << units} sign patterns, "
                 f"so no {alternative} p-value under {smallest:.3g} is possible and "
                 f"the test cannot reject at 0.05 whatever the data")
    elif count < FEW_RESAMPLES:
        smallest = 1 / (count + 1)
        cannot = (", so the test cannot reject at 0.05 whatever the data"
                  if smallest > 0.05 else "")
        warn(f"{name}(): with {_plural(count, 'sign pattern')} drawn at random the "
             f"p-value rests on the draw: another seed moves it, and none comes out "
             f"under 1/{count + 1} = {smallest:.3g}{cannot}; use at least "
             f"{FEW_RESAMPLES} (the default is 10000)")
    null = _flipped(sums, denominator, exact, count, rng)
    # the observed value in the arithmetic of the null, as its all-plus pattern
    observed = float(sums.sum()) / denominator
    tolerance = max(100 * np.finfo(float).eps * abs(observed),
                    _rounding(units, float(np.sum(np.abs(sums))) / denominator))
    if alternative == 'greater':
        hits = np.count_nonzero(null >= observed - tolerance)
    elif alternative == 'less':
        hits = np.count_nonzero(null <= observed + tolerance)
    else:
        hits = np.count_nonzero(np.abs(null) >= abs(observed) - tolerance)
    pvalue = hits / count if exact else (hits + 1) / (count + 1)
    return PermutationResult(statistic, min(1.0, float(pvalue)), n, n_groups, count,
                             exact, alternative)


# --- several comparisons ------------------------------------------------------

def holm(pvalues: Union[Sequence[float], np.ndarray, Mapping[Any, float]]):
    """Holm's adjusted p-values [H79], for several tests read together:
    several systems each against one baseline, say.

    >>> kudio.holm([0.01, 0.04, 0.03, 0.005])                          # doctest: +SKIP
    array([0.03, 0.06, 0.06, 0.02])
    >>> kudio.holm({'A': 0.01, 'B': 0.04})                             # doctest: +SKIP
    {'A': 0.02, 'B': 0.04}

    Ranked from the smallest, the k-th of m p-values is multiplied by
    m - k + 1, each is raised to the largest of those before it, and none
    goes above 1. A test whose adjusted p-value is at or under the level is
    rejected, and the chance of rejecting any true null among them stays at
    the level, however the tests depend on each other. Equal p-values get
    equal adjusted ones.

    :param pvalues: the p-values, in any order: a sequence, an array, or a
        dict from a name to its p-value.
    :raises ValueError: for a p-value outside 0 to 1, NaN, or not a number.
    :returns: the adjusted p-values in the order given, as a float64 array,
        or as a dict with the same keys when a dict was given.
    """
    if isinstance(pvalues, Mapping):
        keys = list(pvalues)
        adjusted = holm([pvalues[key] for key in keys])
        return dict(zip(keys, adjusted.tolist()))
    name = 'holm'
    p = _vector(pvalues, 'p-values', name,
                hint="A test without a p-value cannot be adjusted with the others; "
                     "leave it out, and say so")
    outside = (p < 0) | (p > 1)
    if np.any(outside):
        raise ValueError(f"{name}(): p-values lie between 0 and 1, got "
                         f"{float(p[np.argmax(outside)])!r} among them")
    m = p.size
    order = np.argsort(p, kind='stable')
    scaled = (m - np.arange(m)) * p[order]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted
