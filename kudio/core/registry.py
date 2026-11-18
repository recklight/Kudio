# -*- coding: utf-8 -*-
"""What each metric is, which way is better, what changes its number, and
what to cite for it.

A number from a metric is only half a result. "PESQ 2.9" can be raw P.862 or
a MOS-LQO, narrowband or wideband; an LSD can be on the natural log or in
dB; a min DCF means nothing without its prior. :data:`METRIC_INFO` keeps,
for every metric kudio computes, the facts a table or a methods section needs
next to the number, and :func:`cite` turns a list of metric names into the
BibTeX for kudio and for the works those metrics come from::

    >>> kudio.metric_info('segsnr').function                # doctest: +SKIP
    'kudio.segmental_snr'
    >>> print(kudio.cite(['pesq', 'stoi', 'si_sdr']))       # doctest: +SKIP
    @misc{kudio,
      author = {RL},
    ...

Names. Every name :func:`kudio.score` takes is here, with the same spelling,
and so is every public metric function under its own name. Where the two
differ, both are entries: ``segsnr`` and ``segmental_snr``, ``is`` and
``itakura_saito``, ``csig`` and ``composite``. ``stoi``, ``estoi`` and
``sdr`` exist only as names of :func:`kudio.score`, which runs pystoi and
mir_eval for them.

References. Every BibTeX entry is copied from a footnote of the guide or of
the validation page: authors, title, venue, volume, issue, pages, year, DOI
and arXiv number as the footnote gives them, and nothing it does not say.
tests/test_registry.py holds each entry to every footnote that cites the
same title. A metric cites what its definition comes from and, where kudio
follows someone's code, the work that code belongs to; a metric that is a
line of arithmetic, such as ``snr``, cites nothing.

What I left out. Functions that return the counts or the parts behind a
registered metric (``word_errors``, ``char_errors``, ``diarization_errors``,
``f0_errors``) are not entries of their own; the metric is. Curves that are
read on the way to a number (``schroeder_curve``, ``loudness_over_time``) are
not either, and nor are the functions that change audio rather than measure
it (``normalize_lufs``, ``match_loudness``, ``rir``, ``apply_rir``). The
numbers of :func:`kudio.audio_report` (level, clipping, DC offset, noise
floor, bandwidth) are a health check that catches a broken file, not a score
anyone reports, and :func:`kudio.speech_ratio` is how much of a clip a VAD
calls speech. :func:`kudio.find_delay` measures a delay so that a pair can
be lined up before it is scored. ``det_curve`` is a curve and is here all
the same, because a DET plot is reported and cited like a number.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from kudio._version import __version__

__all__ = ['MetricInfo', 'METRIC_INFO', 'BIBTEX', 'metric_info', 'cite']

#: where the validation page lives, as the docs link it
_VALIDATION = 'validation.md#'


@dataclass(frozen=True)
class MetricInfo:
    """One metric: how to call it, how to read it, and what to cite.

    :param name: the name to pass, as :func:`kudio.score` spells it where
        score takes it, and the function's name otherwise.
    :param label: what a table heading calls it.
    :param function: the dotted path of the function that computes it, such
        as ``'kudio.pesq'``.
    :param field: where the number sits in what *function* returns: an
        attribute of its result (``'high'`` of :func:`kudio.csii` for
        ``csii_high``), or for :func:`kudio.score` the key of its dict.
        ``None`` when the function returns the number, or a record of
        numbers this entry describes as a whole.
    :param better: ``'higher'`` or ``'lower'``; ``None`` for a measurement
        with no better direction, such as a loudness or a room's decay.
    :param range: ``(lo, hi)`` of the scale the number is on at the
        function's defaults, ``None`` for an open end. Settings that move it
        are named in *variants*.
    :param unit: ``'dB'``, ``'MOS'``, ``'bits'`` and so on; ``''`` for a
        ratio, a fraction or a distance with no unit.
    :param variants: the settings and conventions that change the number,
        so that two figures can be told apart before they are compared.
    :param refs: keys of :data:`BIBTEX`, in the order :func:`cite` gives
        them.
    :param validation: the section of the docs' validation page that says
        what the metric was compared with, ``'validation.md#pesq'``; ``None``
        where that page has no section for it.
    """

    name: str
    label: str
    function: str
    field: Optional[str]
    better: Optional[str]
    range: Tuple[Optional[float], Optional[float]]
    unit: str
    variants: str
    refs: Tuple[str, ...]
    validation: Optional[str]


# --- the works the metrics cite ------------------------------------------------

def _entry(kind: str, key: str, **fields: str) -> str:
    """One BibTeX entry, fields in the order given.

    *author* is written as the references write it, ``'A, B and C'``, and
    joined here with BibTeX's ``and``. *month* is one of BibTeX's month
    macros and goes in bare; every other value goes in braces.
    """
    lines = [f"@{kind}{{{key},"]
    for name, value in fields.items():
        if name == 'author':
            value = ' and '.join(part for chunk in value.split(', ')
                                 for part in chunk.split(' and '))
        if name == 'month':
            lines.append(f"  {name} = {value},")
        else:
            lines.append(f"  {name} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


# Standards, evaluation plans and other documents with an organisation for an
# author keep their whole title in braces, since their capitals are part of
# the name; papers protect only their acronyms and proper names. Accents and
# dashes are written as LaTeX spells them, so that the entries are plain
# ASCII: classic BibTeX sorts and abbreviates {\"u} correctly where it can
# trip over UTF-8, and print(kudio.cite(...)) works on a cp950 console.
_ENTRIES = (
    # SI-SDR and the improvement over the mixture
    _entry('inproceedings', 'leroux2019sdr',
           author='J. {Le Roux}, S. Wisdom, H. Erdogan and J. R. Hershey',
           title='{SDR} -- half-baked or well done?',
           booktitle='Proc. ICASSP', pages='626--630', year='2019',
           doi='10.1109/ICASSP.2019.8683855'),
    _entry('article', 'luo2019convtasnet',
           author='Y. Luo and N. Mesgarani',
           title='{Conv-TasNet}: surpassing ideal time-frequency magnitude masking '
                 'for speech separation',
           journal='IEEE/ACM Trans. Audio, Speech, Lang. Process.',
           volume='27', number='8', pages='1256--1266', year='2019',
           doi='10.1109/TASLP.2019.2915167'),
    _entry('article', 'kolbaek2017upit',
           author='M. Kolb{\\ae}k, D. Yu, Z.-H. Tan and J. Jensen',
           title='Multitalker speech separation with utterance-level permutation '
                 'invariant training of deep recurrent neural networks',
           journal='IEEE/ACM Trans. Audio, Speech, Lang. Process.',
           volume='25', number='10', pages='1901--1913', year='2017',
           doi='10.1109/TASLP.2017.2726762'),
    _entry('article', 'vincent2006bss',
           author="E. Vincent, R. Gribonval and C. F{\\'e}votte",
           title='Performance measurement in blind audio source separation',
           journal='IEEE Trans. Audio, Speech, Lang. Process.',
           volume='14', number='4', pages='1462--1469', year='2006',
           doi='10.1109/TSA.2005.858005'),
    # PESQ
    _entry('techreport', 'itut2001p862',
           author='{ITU-T}',
           title='{Perceptual evaluation of speech quality (PESQ): An objective '
                 'method for end-to-end speech quality assessment of narrow-band '
                 'telephone networks and speech codecs}',
           institution='ITU-T', type='ITU-T Recommendation', number='P.862',
           month='feb', year='2001'),
    _entry('techreport', 'itut2003p8621',
           author='{ITU-T}',
           title='{Mapping function for transforming P.862 raw result scores to '
                 'MOS-LQO}',
           institution='ITU-T', type='ITU-T Recommendation', number='P.862.1',
           month='nov', year='2003'),
    _entry('techreport', 'itut2007p8622',
           author='{ITU-T}',
           title='{Wideband extension to Recommendation P.862 for the assessment '
                 'of wideband telephone networks and speech codecs}',
           institution='ITU-T', type='ITU-T Recommendation', number='P.862.2',
           month='nov', year='2007'),
    _entry('techreport', 'itut2005p862amd2',
           author='{ITU-T}',
           title='{Revised Annex A -- Reference implementations and conformance '
                 'testing for ITU-T Recs P.862, P.862.1 and P.862.2}',
           institution='ITU-T', type='ITU-T Recommendation',
           number='P.862 Amendment 2', month='nov', year='2005'),
    # STOI and ESTOI
    _entry('article', 'taal2011stoi',
           author='C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen',
           title='An algorithm for intelligibility prediction of time-frequency '
                 'weighted noisy speech',
           journal='IEEE Trans. Audio, Speech, Lang. Process.',
           volume='19', number='7', pages='2125--2136', year='2011',
           doi='10.1109/TASL.2011.2114881'),
    _entry('article', 'jensen2016estoi',
           author='J. Jensen and C. H. Taal',
           title='An algorithm for predicting the intelligibility of speech masked '
                 'by modulated noise maskers',
           journal='IEEE/ACM Trans. Audio, Speech, Lang. Process.',
           volume='24', number='11', pages='2009--2022', year='2016',
           doi='10.1109/TASLP.2016.2585878'),
    # Hu and Loizou's measures, the composite, and the code they follow
    _entry('article', 'hu2008quality',
           author='Y. Hu and P. C. Loizou',
           title='Evaluation of objective quality measures for speech enhancement',
           journal='IEEE Trans. Audio, Speech, Lang. Process.',
           volume='16', number='1', pages='229--238', year='2008',
           doi='10.1109/TASL.2007.911054'),
    _entry('inproceedings', 'hu2006composite',
           author='Y. Hu and P. C. Loizou',
           title='Evaluation of objective measures for speech enhancement',
           booktitle='Proc. Interspeech', year='2006',
           doi='10.21437/Interspeech.2006-84'),
    _entry('book', 'loizou2013enhancement',
           author='P. C. Loizou',
           title='{Speech Enhancement: Theory and Practice}',
           edition='2nd', publisher='CRC Press', year='2013',
           doi='10.1201/b14529'),
    _entry('article', 'kitawaki1988coding',
           author='N. Kitawaki, H. Nagabuchi and K. Itoh',
           title='Objective quality evaluation for low-bit-rate speech coding '
                 'systems',
           journal='IEEE J. Sel. Areas Commun.',
           volume='6', number='2', pages='242--248', year='1988',
           doi='10.1109/49.601'),
    _entry('inproceedings', 'klatt1982wss',
           author='D. Klatt',
           title='Prediction of perceived phonetic distance from critical-band '
                 'spectra: a first step',
           booktitle='Proc. ICASSP', volume='7', pages='1278--1281', year='1982',
           doi='10.1109/ICASSP.1982.1171512'),
    # NCM and CSII
    _entry('article', 'ma2009ncm',
           author='J. Ma, Y. Hu and P. C. Loizou',
           title='Objective measures for predicting speech intelligibility in '
                 'noisy conditions based on new band-importance functions',
           journal='J. Acoust. Soc. Am.',
           volume='125', number='5', pages='3387--3405', year='2009',
           doi='10.1121/1.3097493'),
    _entry('article', 'kates2005csii',
           author='J. M. Kates and K. H. Arehart',
           title='Coherence and the speech intelligibility index',
           journal='J. Acoust. Soc. Am.',
           volume='117', number='4', pages='2224--2237', year='2005',
           doi='10.1121/1.1862575'),
    _entry('techreport', 'ansi1997s35',
           author='{ANSI}',
           title='{American National Standard Methods for Calculation of the '
                 'Speech Intelligibility Index}',
           institution='American National Standards Institute',
           address='New York', type='ANSI', number='S3.5-1997', year='1997'),
    # SIIB and SIIB^Gauss, and the estimator SIIB takes the information with.
    # Phys. Rev. E numbers its papers rather than their pages, and the
    # number goes where the pages would.
    _entry('article', 'vankuyk2018siib',
           author='S. {Van Kuyk}, W. B. Kleijn and R. C. Hendriks',
           title='An instrumental intelligibility metric based on information '
                 'theory',
           journal='IEEE Signal Process. Lett.',
           volume='25', number='1', pages='115--119', year='2018',
           doi='10.1109/LSP.2017.2774250'),
    _entry('article', 'vankuyk2018evaluation',
           author='S. {Van Kuyk}, W. B. Kleijn and R. C. Hendriks',
           title='An evaluation of intrusive instrumental intelligibility metrics',
           journal='IEEE/ACM Trans. Audio, Speech, Lang. Process.',
           volume='26', number='11', pages='2153--2166', year='2018',
           doi='10.1109/TASLP.2018.2856374'),
    _entry('article', 'kraskov2004mi',
           author='A. Kraskov, H. St{\\"o}gbauer and P. Grassberger',
           title='Estimating mutual information',
           journal='Phys. Rev. E',
           volume='69', number='6', pages='066138', year='2004',
           doi='10.1103/PhysRevE.69.066138'),
    # LSD and MCD
    _entry('inproceedings', 'saijo2025urgent',
           author='K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, '
                  'A. Kumar, M. Sach, Y. Fu, W. Wang, T. Fingscheidt and '
                  'S. Watanabe',
           title='{Interspeech 2025 URGENT} speech enhancement challenge',
           booktitle='Proc. Interspeech', pages='858--862', year='2025',
           doi='10.21437/Interspeech.2025-1363'),
    _entry('inproceedings', 'liu2022nvsr',
           author='H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang',
           title='Neural vocoder is all you need for speech super-resolution',
           booktitle='Proc. Interspeech', pages='4227--4231', year='2022',
           doi='10.21437/Interspeech.2022-11017'),
    _entry('inproceedings', 'fukada1992mcep',
           author='T. Fukada, K. Tokuda, T. Kobayashi and S. Imai',
           title='An adaptive algorithm for mel-cepstral analysis of speech',
           booktitle='Proc. ICASSP', volume='1', pages='137--140', year='1992',
           doi='10.1109/ICASSP.1992.225953'),
    _entry('article', 'salvador2007fastdtw',
           author='S. Salvador and P. Chan',
           title='Toward accurate dynamic time warping in linear time and space',
           journal='Intelligent Data Analysis',
           volume='11', number='5', pages='561--580', year='2007',
           doi='10.3233/IDA-2007-11508'),
    # measures with no reference signal
    _entry('article', 'falk2010srmr',
           author='T. H. Falk, C. Zheng and W.-Y. Chan',
           title='A non-intrusive quality and intelligibility measure of '
                 'reverberant and dereverberated speech',
           journal='IEEE Trans. Audio, Speech, Lang. Process.',
           volume='18', number='7', pages='1766--1774', year='2010',
           doi='10.1109/TASL.2010.2052247'),
    _entry('inproceedings', 'santos2014srmrnorm',
           author='J. F. Santos, M. Senoussaoui and T. H. Falk',
           title='An improved non-intrusive intelligibility metric for noisy and '
                 'reverberant speech',
           booktitle='Proc. IWAENC', pages='55--59', year='2014',
           doi='10.1109/IWAENC.2014.6953337'),
    _entry('inproceedings', 'kim2008wada',
           author='C. Kim and R. M. Stern',
           title='Robust signal-to-noise ratio estimation based on waveform '
                 'amplitude distribution analysis',
           booktitle='Proc. Interspeech', pages='2598--2601', year='2008',
           doi='10.21437/Interspeech.2008-644'),
    _entry('inproceedings', 'reddy2022dnsmos',
           author='C. K. A. Reddy, V. Gopal and R. Cutler',
           title='{DNSMOS P.835}: A non-intrusive perceptual objective speech '
                 'quality metric to evaluate noise suppressors',
           booktitle='Proc. ICASSP', pages='886--890', year='2022',
           doi='10.1109/ICASSP43922.2022.9746108'),
    _entry('inproceedings', 'reddy2021dnsmos',
           author='C. K. A. Reddy, V. Gopal and R. Cutler',
           title='{DNSMOS}: A non-intrusive perceptual objective speech quality '
                 'metric to evaluate noise suppressors',
           booktitle='Proc. ICASSP', pages='6493--6497', year='2021',
           doi='10.1109/ICASSP39728.2021.9414878'),
    _entry('article', 'dubey2024dns',
           author='H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler, '
                  'A. Ju, M. Zohourian, M. Tang, M. Golestaneh and '
                  'R. Aichner',
           title='{ICASSP} 2023 {Deep Noise Suppression Challenge}',
           journal='IEEE Open Journal of Signal Processing',
           volume='5', pages='725--737', year='2024',
           doi='10.1109/OJSP.2024.3378602',
           eprint='2303.11510', archiveprefix='arXiv'),
    _entry('techreport', 'itur2015bs1770',
           author='{ITU-R}',
           title='{Algorithms to measure audio programme loudness and true-peak '
                 'audio level}',
           institution='ITU-R', type='Recommendation ITU-R', number='BS.1770-4',
           month='oct', year='2015'),
    _entry('techreport', 'ebu2023tech3342',
           author='{EBU}',
           title='{Loudness range: a measure to supplement EBU R 128 loudness '
                 'normalization}',
           institution='EBU', type='EBU Tech', number='3342',
           month='nov', year='2023'),
    # rooms
    _entry('article', 'schroeder1965decay',
           author='M. R. Schroeder',
           title='New method of measuring reverberation time',
           journal='J. Acoust. Soc. Am.',
           volume='37', number='3', pages='409--412', year='1965',
           doi='10.1121/1.1909343'),
    _entry('techreport', 'iso2009acoustics',
           author='{ISO}',
           title='{Acoustics --- Measurement of room acoustic parameters --- '
                 'Part 1: Performance spaces}',
           institution='ISO', type='ISO', number='3382-1:2009', year='2009'),
    # recognizers
    _entry('misc', 'jurafsky2026slp',
           author='D. Jurafsky and J. H. Martin',
           title='{Speech and Language Processing}',
           note='3rd ed., draft of 19 August 2026', year='2026',
           url='https://web.stanford.edu/~jurafsky/slp3/'),
    # speaker verification and calibration
    _entry('inproceedings', 'martin1997det',
           author='A. Martin, G. Doddington, T. Kamm, M. Ordowski and M. Przybocki',
           title='The {DET} curve in assessment of detection task performance',
           booktitle='Proc. Eurospeech', pages='1895--1898', year='1997',
           doi='10.21437/Eurospeech.1997-504'),
    _entry('inproceedings', 'brummer2011bosaris',
           author='N. Br{\\"u}mmer and E. de Villiers',
           title='The {BOSARIS Toolkit}: Theory, Algorithms and Code for '
                 'Surviving the New {DCF}',
           booktitle="NIST SRE'11 Analysis Workshop", address='Atlanta',
           month='dec', year='2011', eprint='1304.2865', archiveprefix='arXiv'),
    _entry('article', 'brummer2006cllr',
           author='N. Br{\\"u}mmer and J. du Preez',
           title='Application-independent evaluation of speaker detection',
           journal='Computer Speech and Language',
           volume='20', number='2--3', pages='230--275', year='2006',
           doi='10.1016/j.csl.2005.08.001'),
    _entry('misc', 'nist2008sre',
           author='{NIST}',
           title='{The NIST Year 2008 Speaker Recognition Evaluation Plan}',
           note='3 April 2008', year='2008'),
    _entry('misc', 'nist2010sre',
           author='{NIST}',
           title='{The NIST Year 2010 Speaker Recognition Evaluation Plan}',
           note='version r6, 21 April 2010', year='2010'),
    _entry('misc', 'nist2016sre',
           author='{NIST}',
           title='{NIST 2016 Speaker Recognition Evaluation Plan}',
           note='4 August 2016', year='2016'),
    _entry('misc', 'nist2019cts',
           author='{NIST}',
           title='{NIST 2019 Speaker Recognition Evaluation: CTS Challenge}',
           note='22 July 2019', year='2019'),
    _entry('misc', 'nist2019av',
           author='{NIST}',
           title='{NIST 2019 Speaker Recognition Evaluation Plan}',
           note='16 August 2019, the audio-visual evaluation', year='2019'),
    # diarization
    _entry('misc', 'nist2009rt09',
           author='{NIST}',
           title='{The 2009 (RT-09) Rich Transcription Meeting Recognition '
                 'Evaluation Plan}',
           year='2009'),
    _entry('misc', 'ryant2019dihard',
           author='N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy '
                  'and M. Liberman',
           title='{Second DIHARD Challenge Evaluation Plan}',
           note='version 1.1', year='2019'),
    # pitch trackers
    _entry('inproceedings', 'chu2009ffe',
           author='W. Chu and A. Alwan',
           title='Reducing {F0} frame error of {F0} tracking algorithms under noisy '
                 'conditions with an unvoiced/voiced classification frontend',
           booktitle='Proc. ICASSP', pages='3969--3972', year='2009',
           doi='10.1109/ICASSP.2009.4960497'),
    _entry('inproceedings', 'drugman2011srh',
           author='T. Drugman and A. Alwan',
           title='Joint robust voicing detection and pitch estimation based on '
                 'residual harmonics',
           booktitle='Proc. Interspeech', pages='1973--1976', year='2011',
           doi='10.21437/Interspeech.2011-519'),
    _entry('article', 'rabiner1976pitch',
           author='L. R. Rabiner, M. J. Cheng, A. E. Rosenberg and C. A. McGonegal',
           title='A comparative performance study of several pitch detection '
                 'algorithms',
           journal='IEEE Trans. Acoust., Speech, Signal Process.',
           volume='24', number='5', pages='399--418', year='1976',
           doi='10.1109/TASSP.1976.1162846'),
)


def _key(entry: str) -> str:
    return entry[entry.index('{') + 1:entry.index(',')]


#: BibTeX for every work a metric in :data:`METRIC_INFO` cites, by key.
BIBTEX: Mapping[str, str] = MappingProxyType({_key(e): e for e in _ENTRIES})


# --- the metrics ---------------------------------------------------------------

def _metric(name: str, label: str, better: Optional[str],
            range: Tuple[Optional[float], Optional[float]], unit: str,
            variants: str, refs: Sequence[str] = (), *,
            function: Optional[str] = None, field: Optional[str] = None,
            section: Optional[str] = None) -> MetricInfo:
    return MetricInfo(name=name, label=label,
                      function=function or f'kudio.{name}', field=field,
                      better=better, range=range, unit=unit, variants=variants,
                      refs=tuple(refs),
                      validation=None if section is None else _VALIDATION + section)


def _also(info: MetricInfo, name: str, label: Optional[str] = None,
          field: Optional[str] = None) -> MetricInfo:
    """*info* under :func:`kudio.score`'s name for it."""
    return dataclasses.replace(info, name=name, label=label or info.label,
                               field=field)


_PESQ = ('itut2001p862', 'itut2003p8621', 'itut2007p8622', 'itut2005p862amd2')
#: The default MOS-LQO is P.862.1's at 8 kHz and P.862.2's at 16 kHz. Their
#: mappings take the raw -0.5..4.5 to 1.0168..4.5486 and to 1.0427..4.6439
#: (the formula is beside quality._PESQ_RANGE), so the score lies between the
#: first's floor and the second's ceiling, here rounded outwards: a bound
#: rounded to the nearest, 1.02, would put a real narrowband floor of 1.017
#: outside its own range.
_PESQ_LQO = (1.016, 4.644)
_HL = ('hu2008quality', 'loizou2013enhancement')
_SNR_SECTION = 'snr-si-sdr-stoi-and-sdr'
_HL_SECTION = 'hu-and-loizous-measures-and-the-composite'
_CSII_SECTION = 'ncm-and-csii'
#: What a copy of the reference scores on SIIB and SIIB^Gauss: 80 / 15 stacks
#: a second times 420 eigenchannels of -1/2 log2(1 - 0.75^2) bits, 1335.7625
#: bit/s, rounded outwards as _PESQ_LQO is.
_SIIB_RANGE = (0.0, 1335.77)
_SIIB_VARIANTS = (
    "None, but the rate and the length. Any rate but 16 kHz is taken to "
    "16 kHz with the filter SIIB.m's resample designs; through a sharper "
    "one, as pySIIB has, SIIB moved by up to 5.1 bit/s and SIIB^Gauss by up "
    "to 1.6 on kudio's test speech at 8 and 20 kHz. The frames scored are those "
    "within 40 dB of the reference's loud end, which a DC offset moves, "
    "since nothing takes it out. More speech scores more: the authors "
    "advise 20 s or more, and under 5.45 s of speech is nan.")
_DET_SECTION = 'eer-minimum-dcf-and-the-det-curve'
_CALIBRATION_SECTION = 'calibration-c_llr-min-c_llr-the-actual-dcf-and-c_primary'
_DIARIZATION_SECTION = 'diarization-der-jer-and-the-detection-error-rate'
_F0_SECTION = 'f0-errors'
_SEPARATION_SECTION = 'separation-si-sdri-pit-and-bss-eval'
_HL_FRAMES = ("Loizou's MATLAB code: LPC of order 10 at 8 kHz and 16 at 16 kHz on "
              "30 ms frames, the mean over the best 95% of frames. 8 and 16 kHz "
              "only.")


def _build() -> Dict[str, MetricInfo]:
    segmental = _metric(
        'segmental_snr', 'segSNR', 'higher', (-10.0, 35.0), 'dB',
        "frame_length and hop_length, 512 and 256 samples by default and "
        "always in score(), so 32 ms at 16 kHz and 10.7 ms at 48 kHz. Each "
        "frame held to [-10, 35] dB; silent frames count.",
        section=_SNR_SECTION)
    itakura = _metric(
        'itakura_saito', 'IS', 'lower', (0.0, 100.0), '',
        "The log term as comp_is.m has it, the reciprocal of Hu and Loizou's "
        "printed eq. 5; each frame held to [0, 100]. " + _HL_FRAMES,
        _HL, section=_HL_SECTION)
    cepstral = _metric(
        'cepstral_distance', 'CEP', 'lower', (0.0, 10.0), 'dB',
        "Cepstral coefficients 1 to p of the LPC envelopes, c0 left out; "
        "each frame held to [0, 10] dB. " + _HL_FRAMES,
        ('hu2008quality', 'kitawaki1988coding', 'loizou2013enhancement'),
        section=_HL_SECTION)
    fw_segmental = _metric(
        'fw_segmental_snr', 'fwSegSNR', 'higher', (-10.0, 35.0), 'dB',
        "25 bands up to 3.8 kHz at both rates, both spectra first scaled to "
        "unit area over every bin; each frame held to [-10, 35] dB. 8 and "
        "16 kHz only.",
        _HL, section=_HL_SECTION)
    composite = _metric(
        'composite', 'CSIG, CBAK, COVL', 'higher', (1.0, 5.0), 'MOS',
        "Which PESQ goes in (mode, scale, or a pesq= of your own): wideband "
        "MOS-LQO at 16 kHz and raw narrowband P.862 at 8 kHz by default, as "
        "published 16 kHz tables have it; mode='nb', scale='raw' is the "
        "variant the regression was fitted on, 0.85 higher on COVL on "
        "VoiceBank-DEMAND's noisy test set. score() always takes the default. "
        "8 and 16 kHz only.",
        ('hu2006composite',) + _HL + _PESQ, section=_HL_SECTION)
    csii = _metric(
        'csii', 'CSII', 'higher', (0.0, 1.0), '',
        "CSII.m's 16 critical bands up to 3.7 kHz at any rate from 8 kHz, at "
        "three levels of the reference's 30 ms frames: high at or above its "
        "RMS, mid down to 10 dB under it, low everything further down, with "
        "no lower limit where Kates and Arehart stop at 30 dB.",
        ('ma2009ncm', 'kates2005csii', 'ansi1997s35', 'loizou2013enhancement'),
        section=_CSII_SECTION)

    infos = [
        # --- against the clean reference, sample for sample
        _metric('snr', 'SNR', 'higher', (None, None), 'dB',
                "None. Level, polarity and delay all count against it.",
                section=_SNR_SECTION),
        _metric('si_sdr', 'SI-SDR', 'higher', (None, None), 'dB',
                "None. Both signals are taken to zero mean first, as "
                "Conv-TasNet's SI-SNR is; torchmetrics' SI-SDR keeps the mean "
                "unless zero_mean=True.",
                ('leroux2019sdr', 'luo2019convtasnet'), section=_SNR_SECTION),
        _also(segmental, 'segsnr'),
        segmental,
        _metric('sdi', 'SDI', 'lower', (0.0, None), '',
                "None. The error energy of the whole output over the clean "
                "energy, 10 ** (-snr / 10); not the speech-distortion index of "
                "the noise-reduction literature, which passes the clean speech "
                "through the filter alone.",
                section=_SNR_SECTION),
        _metric('pesq', 'PESQ', 'higher', _PESQ_LQO, 'MOS',
                "mode ('nb' is P.862, 'wb' P.862.2; 'auto' is wideband at "
                "16 kHz and narrowband at 8 kHz) and scale ('lqo' MOS-LQO, or "
                "'raw' P.862, -0.5 to 4.5). MOS-LQO runs 1.017 to 4.549 "
                "narrowband (P.862.1) and 1.043 to 4.644 wideband (P.862.2). "
                "Wideband scores are the 2005 code's, without P.862 "
                "Corrigendum 2 (2018). 8 and 16 kHz only.",
                _PESQ, section='pesq'),
        _metric('stoi', 'STOI', 'higher', (-1.0, 1.0), '',
                "None; pystoi's STOI, at its own 10 kHz.",
                ('taal2011stoi',), function='kudio.score', field='stoi',
                section=_SNR_SECTION),
        _metric('estoi', 'ESTOI', 'higher', (-1.0, 1.0), '',
                "None; pystoi's extended STOI, at its own 10 kHz.",
                ('jensen2016estoi',), function='kudio.score', field='estoi',
                section=_SNR_SECTION),
        _metric('sdr', 'SDR', 'higher', (None, None), 'dB',
                "None; mir_eval's bss_eval_sources, with its 512-tap "
                "distortion filter. Not comparable with SI-SDR.",
                ('vincent2006bss',), function='kudio.score', field='sdr',
                section=_SNR_SECTION),
        _metric('lsd', 'LSD', 'lower', (0.0, None), '',
                "URGENT 2025's settings by default: natural log, 32 ms frames "
                "every 16 ms, deg first scaled by its least-squares gain, eps "
                "1e-8. scale ('ln', 'log10', 'db'), n_fft, hop_length, "
                "match_gain and eps change it; ssr_eval's number at 44.1 kHz "
                "is scale='log10', n_fft=2048, hop_length=441, "
                "match_gain=False, eps=1e-12.",
                ('saijo2025urgent', 'liu2022nvsr'),
                section='log-spectral-distance'),
        # --- Hu and Loizou's measures and the composite
        _metric('llr', 'LLR', 'lower', (0.0, 2.0), '',
                "Each frame held to [0, 2]. " + _HL_FRAMES,
                _HL, section=_HL_SECTION),
        _also(itakura, 'is'),
        itakura,
        _also(cepstral, 'cep'),
        cepstral,
        _metric('wss', 'WSS', 'lower', (0.0, None), '',
                "Loizou's MATLAB code: 25 bands up to 3.8 kHz at both rates on "
                "30 ms frames, the mean over the best 95% of frames. Band "
                "energies are floored at -100 dB of full scale, which moves the "
                "score once the clean peak is under about -50 dBFS. 8 and "
                "16 kHz only.",
                ('hu2008quality', 'klatt1982wss', 'loizou2013enhancement'),
                section=_HL_SECTION),
        _also(fw_segmental, 'fwsegsnr'),
        fw_segmental,
        composite,
        _also(composite, 'csig', 'CSIG', field='csig'),
        _also(composite, 'cbak', 'CBAK', field='cbak'),
        _also(composite, 'covl', 'COVL', field='covl'),
        _metric('mcd', 'MCD', 'lower', (0.0, None), 'dB',
                "URGENT 2025's: frames of 1024 samples every 256 at every rate "
                "(n_fft, hop_length), SPTK's mel-cepstral analysis of the "
                "order and alpha its table gives for the rate, c0 included, "
                "FastDTW of radius 1, deg first scaled by its least-squares "
                "gain (match_gain=False for a pair that differs in timing). "
                "order and alpha, given together, replace the table's at any "
                "rate.",
                ('saijo2025urgent', 'fukada1992mcep', 'salvador2007fastdtw'),
                section='mel-cepstral-distortion'),
        # --- intelligibility
        _metric('ncm', 'NCM', 'higher', (0.0, 1.0), '',
                "None, but the rate: NCM.m's 20 bands run from 300 to 3400 Hz "
                "at 8 kHz and to 7400 Hz at 16 kHz, so one pair scores "
                "differently at the two. 8 and 16 kHz only.",
                ('ma2009ncm', 'ansi1997s35', 'loizou2013enhancement'),
                section=_CSII_SECTION),
        csii,
        _also(csii, 'csii_high', 'CSII high', field='high'),
        _also(csii, 'csii_mid', 'CSII mid', field='mid'),
        _also(csii, 'csii_low', 'CSII low', field='low'),
        _metric('siib', 'SIIB', 'higher', _SIIB_RANGE, 'bit/s',
                _SIIB_VARIANTS + " The noise that breaks ties in the "
                "estimator is drawn with a fixed seed, so a pair scores the "
                "same every time; SIIB.m draws it from MATLAB's generator in "
                "whatever state the session left it, which moves its score "
                "by a few parts in 1e5.",
                ('vankuyk2018siib', 'kraskov2004mi'), section='siib'),
        _metric('siib_gauss', 'SIIB^Gauss', 'higher', _SIIB_RANGE, 'bit/s',
                _SIIB_VARIANTS + " No random numbers enter it.",
                ('vankuyk2018evaluation', 'vankuyk2018siib'), section='siib'),
        # --- several sources
        _metric('si_sdr_improvement', 'SI-SDRi', 'higher', (None, None), 'dB',
                "None; the output's SI-SDR minus the mixture's, against the "
                "same source.",
                ('leroux2019sdr', 'luo2019convtasnet'),
                section=_SEPARATION_SECTION),
        _metric('sdr_improvement', 'SDRi', 'higher', (None, None), 'dB',
                "None; BSS Eval's SDR of the output minus the mixture's, as "
                "mir_eval computes it.",
                ('vincent2006bss', 'luo2019convtasnet'),
                section=_SEPARATION_SECTION),
        _metric('pit_si_sdr', 'SI-SDR (PIT)', 'higher', (None, None), 'dB',
                "The assignment with the highest mean SI-SDR over the sources, "
                "found exactly; per source, so papers report the mean. With "
                "the mixture, SI-SDRi too.",
                ('kolbaek2017upit', 'leroux2019sdr', 'luo2019convtasnet'),
                section=_SEPARATION_SECTION),
        _metric('bss_eval', 'SDR, SIR, SAR', 'higher', (None, None), 'dB',
                "permutation: mir_eval's own by default, the highest mean SIR; "
                "pit_si_sdr's to score a table with one assignment. With the "
                "mixture, SDRi too.",
                ('vincent2006bss',), section=_SEPARATION_SECTION),
        # --- no reference
        _metric('srmr', 'SRMR', 'higher', (0.0, None), '',
                "fast, norm (Santos et al.'s variant, the bands stopped at "
                "30 Hz as SRMR.m stops them), min_cf, max_cf and drop_silence. "
                "SRMRToolbox's SRMR.m step for step, which SRMRpy is not. 8 "
                "and 16 kHz only.",
                ('falk2010srmr', 'santos2014srmrnorm'), section='srmr'),
        _metric('wada_snr', 'WADA-SNR', 'higher', (-20.0, 100.0), 'dB',
                "None. The curve is computed from the model, speech gamma of "
                "shape 0.4 in Gaussian noise, rather than read off the "
                "published table; outside -20 to 100 dB the result is nan "
                "where the ports clamp.",
                ('kim2008wada',), section='wada-snr'),
        _metric('dnsmos', 'DNSMOS SIG, BAK, OVRL', 'higher', (1.0, 5.0), 'MOS',
                "personalized (the pDNSMOS model) and polyfit (False gives the "
                "raw outputs, not on the 1-5 scale). DNS-Challenge's model "
                "files, run as its dnsmos_local.py runs them.",
                ('reddy2022dnsmos', 'dubey2024dns'), section='dnsmos'),
        _metric('dnsmos_p808', 'DNSMOS P.808', 'higher', (1.0, 5.0), 'MOS',
                "librosa: '0.8' gives the mel spectrogram of librosa 0.8.1, "
                "which DNS-Challenge pins, '0.9' that of later librosa and of "
                "torchmetrics; up to 0.19 apart.",
                ('reddy2021dnsmos',), section='dnsmos'),
        _metric('loudness', 'Loudness', None, (None, None), 'LUFS',
                "block_size: 400 ms is the standard's, and anything else is no "
                "longer a BS.1770 measurement.",
                ('itur2015bs1770',)),
        _metric('loudness_range', 'LRA', None, (0.0, None), 'LU',
                "None; EBU Tech 3342's gates, -70 LUFS and 20 LU under the "
                "power mean, on the short-term loudness.",
                ('ebu2023tech3342', 'itur2015bs1770')),
        _metric('true_peak', 'True peak', None, (None, None), 'dBTP',
                "oversample: by default whatever brings the rate to at least "
                "192 kHz, as the standard asks.",
                ('itur2015bs1770',)),
        _metric('rt60', 'T30', None, (0.0, None), 's',
                "T30 by convention (Reverberation.seconds); T20 and EDT from "
                "other ranges of the same decay, and C50, C80 and DRR in dB. "
                "Measured over the whole band it is given, where ISO 3382-1 "
                "measures octave or one-third-octave bands.",
                ('schroeder1965decay', 'iso2009acoustics'), field='t30'),
        # --- recognizers
        _metric('wer', 'WER', 'lower', (0.0, None), '',
                "None. Nothing is normalised: case, punctuation and the way "
                "numbers are written count as they stand. A word is what "
                "str.split() separates; a corpus is pooled, its errors over all "
                "its reference words.",
                ('jurafsky2026slp',), section='wer-and-cer'),
        _metric('cer', 'CER', 'lower', (0.0, None), '',
                "spaces: False leaves whitespace out, as Kaldi's CER recipe "
                "does; True counts one space between words, as jiwer does. "
                "Nothing normalised; a corpus is pooled.",
                ('jurafsky2026slp',), section='wer-and-cer'),
        # --- speaker verification
        _metric('eer', 'EER', 'lower', (0.0, 1.0), '',
                "method: 'linear' between successive operating points, as the "
                "VoxSRC scorer and NIST's SRE 2016 script read it, or 'rocch', "
                "BOSARIS's convex hull, never above it.",
                ('martin1997det', 'brummer2011bosaris'), section=_DET_SECTION),
        _metric('min_dcf', 'minDCF', 'lower', (0.0, 1.0), '',
                "p_target, which has no default, c_miss and c_fa. As "
                "(c_miss, c_fa, p_target): SRE 2008 (10, 1, 0.01), SRE 2010 "
                "(1, 1, 0.001), SRE 2016 (1, 1, 0.01 and 0.005), VoxSRC "
                "(1, 1, 0.05). Normalized by the cost of deciding without the "
                "scores.",
                ('nist2008sre', 'nist2010sre', 'nist2016sre', 'nist2019cts',
                 'nist2019av'), section=_DET_SECTION),
        _metric('det_curve', 'DET curve', None, (0.0, 1.0), '',
                "None. Every operating point, tied scores one threshold.",
                ('martin1997det',), section=_DET_SECTION),
        _metric('cllr', 'C_llr', 'lower', (0.0, None), 'bits',
                "None. The scores are read as natural-log likelihood ratios.",
                ('brummer2006cllr',), section=_CALIBRATION_SECTION),
        _metric('min_cllr', 'min C_llr', 'lower', (0.0, 1.0), 'bits',
                "None. Tied scores share one LLR, as in PYLLR; BOSARIS's "
                "min_cllr.m splits them and reads lower.",
                ('brummer2006cllr',), section=_CALIBRATION_SECTION),
        _metric('act_dcf', 'actDCF', 'lower', (0.0, None), '',
                "p_target, c_miss and c_fa, as min_dcf; the threshold is "
                "log(beta) on natural-log LLRs.",
                ('nist2016sre', 'nist2019cts'), section=_CALIBRATION_SECTION),
        _metric('c_primary', 'C_Primary', 'lower', (0.0, None), '',
                "minimum averages min_dcf instead. The list is pooled as it "
                "comes, where SRE 2016 averages over partitions of it.",
                ('nist2016sre', 'nist2019cts'), section=_CALIBRATION_SECTION),
        # --- diarization
        _metric('der', 'DER', 'lower', (0.0, None), '',
                "collar, per side as md-eval.pl and dscore take it "
                "(pyannote.metrics' is the total), skip_overlap and uem. The "
                "speakers are mapped on all the time evaluated, as md-eval.pl "
                "maps them.",
                ('nist2009rt09',), section=_DIARIZATION_SECTION),
        _metric('jer', 'JER', 'lower', (0.0, 1.0), '',
                "uem. The speakers are paired by the smallest summed Jaccard "
                "error, as dscore pairs them; pyannote.metrics pairs by shared "
                "time.",
                ('ryant2019dihard',), section=_DIARIZATION_SECTION),
        _metric('detection_error_rate', 'Detection error rate', 'lower',
                (0.0, None), '',
                "collar and uem, as der; every speaker merged into one.",
                section=_DIARIZATION_SECTION),
        # --- pitch trackers
        _metric('gross_pitch_error', 'GPE', 'lower', (0.0, 1.0), '',
                "threshold, 0.2 of the reference and strict; over the frames "
                "voiced in both.",
                ('chu2009ffe', 'rabiner1976pitch'), section=_F0_SECTION),
        _metric('voicing_decision_error', 'VDE', 'lower', (0.0, 1.0), '',
                "None; over all frames, so frames unvoiced in both dilute it.",
                ('chu2009ffe', 'rabiner1976pitch'), section=_F0_SECTION),
        _metric('f0_frame_error', 'FFE', 'lower', (0.0, 1.0), '',
                "threshold, as gross_pitch_error; over all frames.",
                ('chu2009ffe', 'rabiner1976pitch'), section=_F0_SECTION),
        _metric('fine_pitch_error', 'FPE', 'lower', (0.0, 0.2), '',
                "threshold, as gross_pitch_error, which also bounds it. A "
                "fraction of the reference F0, where Drugman and Alwan print it "
                "times 100, in percent; a standard deviation that divides by "
                "the number of frames, where MATLAB's std divides by one fewer.",
                ('drugman2011srh', 'rabiner1976pitch'), section=_F0_SECTION),
        _metric('f0_rmse', 'F0 RMSE', 'lower', (0.0, None), 'cents',
                "None; over the frames voiced in both, gross errors counted "
                "at full size.",
                section=_F0_SECTION),
    ]
    return {info.name: info for info in infos}


#: Every metric kudio computes, by the name you pass for it.
METRIC_INFO: Mapping[str, MetricInfo] = MappingProxyType(_build())


def metric_info(name: str) -> MetricInfo:
    """The :class:`MetricInfo` of *name*, a name :func:`kudio.score` takes or
    the name of a metric function.

    >>> kudio.metric_info('csii_high').field                # doctest: +SKIP
    'high'

    :raises ValueError: for a name that is not in :data:`METRIC_INFO`.
    """
    if not isinstance(name, str):
        raise ValueError(f"metric_info(): the name has to be a string, got {name!r}")
    try:
        return METRIC_INFO[name]
    except KeyError:
        raise ValueError(f"metric_info(): unknown metric {name!r}; choose from "
                         f"{tuple(METRIC_INFO)}") from None


def _kudio_entry() -> str:
    # @misc rather than @software: plain.bst and IEEEtran.bst know no
    # @software and drop its version, and every style prints a note. No
    # year: the year a version came out is not known until it is out, and a
    # guessed one in a bibliography outlives the guess. The DOI is Zenodo's
    # concept DOI, which resolves to the newest release; the version in the
    # note says which one was used.
    return "\n".join(("@misc{kudio,",
                      "  author = {RL},",
                      "  title = {Kudio},",
                      f"  note = {{Version {__version__}}},",
                      "  doi = {10.5281/zenodo.24219274},",
                      "  url = {https://github.com/recklight/Kudio},",
                      "}"))


def cite(metrics: Union[str, Sequence[str], None] = None) -> str:
    """BibTeX for kudio and for the works behind *metrics*.

    >>> print(kudio.cite(['pesq', 'stoi']))                 # doctest: +SKIP

    Kudio's own entry comes first. Then each metric's references in the order
    :data:`METRIC_INFO` lists them, metric by metric in the order given, each
    work once however many of the metrics cite it. A metric that cites
    nothing, such as ``snr``, adds nothing.

    :param metrics: a name or a list of names, as :func:`kudio.score` takes
        them or as the functions are called; ``None`` gives kudio's entry
        alone.
    :raises ValueError: for a name that is not in :data:`METRIC_INFO`, all
        of them named at once, and for a set of names, whose order would
        change from run to run.
    :returns: the entries, a blank line between each two, ending in a
        newline, ready to save as a ``.bib`` file.
    """
    names = _names(metrics)
    unknown = [n for n in names if n not in METRIC_INFO]
    if unknown:
        raise ValueError(f"cite(): unknown metric(s) {unknown}; choose from "
                         f"{tuple(METRIC_INFO)}")
    keys: List[str] = []
    for name in names:
        keys.extend(k for k in METRIC_INFO[name].refs if k not in keys)
    return "\n\n".join([_kudio_entry()] + [BIBTEX[k] for k in keys]) + "\n"


def _names(metrics: Union[str, Iterable[str], None]) -> List[str]:
    if metrics is None:
        return []
    if isinstance(metrics, str):
        return [metrics]
    if isinstance(metrics, (bytes, bytearray)):
        raise ValueError(f"cite(): a metric name is text, got bytes {metrics!r}")
    if isinstance(metrics, (set, frozenset)):
        # a set of strings comes out in another order in every process, and
        # so would the bibliography
        raise ValueError(f"cite(): metrics has to be a name or a list of names, "
                         f"not a set, whose order changes from run to run; got "
                         f"{metrics!r}")
    try:
        names = list(metrics)
    except TypeError:
        raise ValueError(f"cite(): metrics has to be a name or a list of names, "
                         f"got {metrics!r}") from None
    odd = [n for n in names if not isinstance(n, str)]
    if odd:
        raise ValueError(f"cite(): metric names have to be strings, got {odd}")
    return names
