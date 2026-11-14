# -*- coding: utf-8 -*-
"""The metric registry: every metric has an entry, every entry points at a
real function, every reference is the one the docs give, every validation
anchor exists, and cite() and CITATION.cff say the same about kudio."""
import dataclasses
import importlib
import inspect
import json
import re
import unicodedata
from pathlib import Path

import pytest

import kudio
from kudio.core.evaluator import METRICS
from kudio.core.registry import BIBTEX, METRIC_INFO, cite, metric_info

ROOT = Path(__file__).resolve().parents[1]

_HL = ('hu2008quality', 'loizou2013enhancement')
_PESQ = ('itut2001p862', 'itut2003p8621', 'itut2007p8622', 'itut2005p862amd2')
_NIST = ('nist2008sre', 'nist2010sre', 'nist2016sre', 'nist2019cts', 'nist2019av')
_F0 = ('chu2009ffe', 'rabiner1976pitch')
_NO_END = (None, None)

#: every metric function by its own name, and the three score() names that
#: run another package, written out a second time: (better, unit, range, refs)
EXPECTED = {
    'snr': ('higher', 'dB', _NO_END, ()),
    'si_sdr': ('higher', 'dB', _NO_END, ('leroux2019sdr', 'luo2019convtasnet')),
    'segmental_snr': ('higher', 'dB', (-10.0, 35.0), ()),
    'sdi': ('lower', '', (0.0, None), ()),
    'pesq': ('higher', 'MOS', (1.016, 4.644), _PESQ),
    'stoi': ('higher', '', (-1.0, 1.0), ('taal2011stoi',)),
    'estoi': ('higher', '', (-1.0, 1.0), ('jensen2016estoi',)),
    'sdr': ('higher', 'dB', _NO_END, ('vincent2006bss',)),
    'lsd': ('lower', '', (0.0, None), ('saijo2025urgent', 'liu2022nvsr')),
    'llr': ('lower', '', (0.0, 2.0), _HL),
    'itakura_saito': ('lower', '', (0.0, 100.0), _HL),
    'cepstral_distance': ('lower', 'dB', (0.0, 10.0),
                          ('hu2008quality', 'kitawaki1988coding', 'loizou2013enhancement')),
    'wss': ('lower', '', (0.0, None),
            ('hu2008quality', 'klatt1982wss', 'loizou2013enhancement')),
    'fw_segmental_snr': ('higher', 'dB', (-10.0, 35.0), _HL),
    'composite': ('higher', 'MOS', (1.0, 5.0), ('hu2006composite',) + _HL + _PESQ),
    'mcd': ('lower', 'dB', (0.0, None),
            ('saijo2025urgent', 'fukada1992mcep', 'salvador2007fastdtw')),
    'ncm': ('higher', '', (0.0, 1.0), ('ma2009ncm', 'ansi1997s35', 'loizou2013enhancement')),
    'csii': ('higher', '', (0.0, 1.0),
             ('ma2009ncm', 'kates2005csii', 'ansi1997s35', 'loizou2013enhancement')),
    'siib': ('higher', 'bit/s', (0.0, 1335.77), ('vankuyk2018siib', 'kraskov2004mi')),
    'siib_gauss': ('higher', 'bit/s', (0.0, 1335.77),
                   ('vankuyk2018evaluation', 'vankuyk2018siib')),
    'si_sdr_improvement': ('higher', 'dB', _NO_END, ('leroux2019sdr', 'luo2019convtasnet')),
    'sdr_improvement': ('higher', 'dB', _NO_END, ('vincent2006bss', 'luo2019convtasnet')),
    'pit_si_sdr': ('higher', 'dB', _NO_END,
                   ('kolbaek2017upit', 'leroux2019sdr', 'luo2019convtasnet')),
    'bss_eval': ('higher', 'dB', _NO_END, ('vincent2006bss',)),
    'srmr': ('higher', '', (0.0, None), ('falk2010srmr', 'santos2014srmrnorm')),
    'wada_snr': ('higher', 'dB', (-20.0, 100.0), ('kim2008wada',)),
    'dnsmos': ('higher', 'MOS', (1.0, 5.0), ('reddy2022dnsmos', 'dubey2024dns')),
    'dnsmos_p808': ('higher', 'MOS', (1.0, 5.0), ('reddy2021dnsmos',)),
    'loudness': (None, 'LUFS', _NO_END, ('itur2015bs1770',)),
    'loudness_range': (None, 'LU', (0.0, None), ('ebu2023tech3342', 'itur2015bs1770')),
    'true_peak': (None, 'dBTP', _NO_END, ('itur2015bs1770',)),
    'rt60': (None, 's', (0.0, None), ('schroeder1965decay', 'iso2009acoustics')),
    'wer': ('lower', '', (0.0, None), ('jurafsky2026slp',)),
    'cer': ('lower', '', (0.0, None), ('jurafsky2026slp',)),
    'eer': ('lower', '', (0.0, 1.0), ('martin1997det', 'brummer2011bosaris')),
    'min_dcf': ('lower', '', (0.0, 1.0), _NIST),
    'det_curve': (None, '', (0.0, 1.0), ('martin1997det',)),
    'cllr': ('lower', 'bits', (0.0, None), ('brummer2006cllr',)),
    'min_cllr': ('lower', 'bits', (0.0, 1.0), ('brummer2006cllr',)),
    'act_dcf': ('lower', '', (0.0, None), ('nist2016sre', 'nist2019cts')),
    'c_primary': ('lower', '', (0.0, None), ('nist2016sre', 'nist2019cts')),
    'der': ('lower', '', (0.0, None), ('nist2009rt09',)),
    'jer': ('lower', '', (0.0, 1.0), ('ryant2019dihard',)),
    'detection_error_rate': ('lower', '', (0.0, None), ()),
    'gross_pitch_error': ('lower', '', (0.0, 1.0), _F0),
    'voicing_decision_error': ('lower', '', (0.0, 1.0), _F0),
    'f0_frame_error': ('lower', '', (0.0, 1.0), _F0),
    'fine_pitch_error': ('lower', '', (0.0, 0.2), ('drugman2011srh', 'rabiner1976pitch')),
    'f0_rmse': ('lower', 'cents', (0.0, None), ()),
}

#: the entries that are not a function returning its own number, as
#: (function, field); every other entry is (f'kudio.{name}', None)
CALLS = {
    'segsnr': ('kudio.segmental_snr', None),
    'is': ('kudio.itakura_saito', None),
    'cep': ('kudio.cepstral_distance', None),
    'fwsegsnr': ('kudio.fw_segmental_snr', None),
    'stoi': ('kudio.score', 'stoi'),
    'estoi': ('kudio.score', 'estoi'),
    'sdr': ('kudio.score', 'sdr'),
    'csig': ('kudio.composite', 'csig'),
    'cbak': ('kudio.composite', 'cbak'),
    'covl': ('kudio.composite', 'covl'),
    'csii_high': ('kudio.csii', 'high'),
    'csii_mid': ('kudio.csii', 'mid'),
    'csii_low': ('kudio.csii', 'low'),
    'rt60': ('kudio.rt60', 't30'),
}

#: public functions of the checked modules that are deliberately not entries
NOT_METRICS = {
    'score', 'eval_metrics',                  # they run metrics, and are none
    'word_errors', 'char_errors',             # the counts behind wer and cer
    'diarization_errors',                     # the parts of der
    'f0_errors',                              # the five F0 measures at once
    'load_rttm', 'find_dnsmos_model',         # they read files and find models
    'schroeder_curve', 'loudness_over_time',  # curves read on the way
    'normalize_lufs', 'match_loudness',       # they change audio
    'rir', 'apply_rir',
    'audio_report',                           # a health check of a file
    'vad', 'vad_split', 'vad_trim', 'vad_frames', 'speech_ratio',
    'find_delay', 'align',                    # they line a pair up
}

#: the modules whose public functions are checked against the registry
METRIC_MODULES = ['evaluator', 'quality', 'intelligibility', 'siib', 'melcepstrum',
                  'separation', 'srmr', 'wada', 'dnsmos', 'loudness', 'room',
                  'wer', 'verification', 'diarization', 'f0eval',
                  'report', 'vad', 'align']

#: the metrics the validation page has no section for
NO_VALIDATION_SECTION = {'loudness', 'loudness_range', 'true_peak', 'rt60'}


def _checkout_file(*parts):
    """A path that only a git checkout has, skipping elsewhere: the sdist
    carries the suite but not the docs."""
    path = ROOT.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{'/'.join(parts)} only exists in a checkout")
    return path


# --- the entries -----------------------------------------------------------------

def test_every_score_name_is_an_entry():
    missing = [m for m in METRICS if m not in METRIC_INFO]
    assert not missing, f"score() takes {missing}, which the registry lacks"


def test_every_metric_function_and_score_name_has_its_expected_reading():
    """Which way is better, the unit, the range and the references, written
    out here a second time, so that a slip in either place shows."""
    seen = set()
    for name, info in METRIC_INFO.items():
        own = name if name in EXPECTED else info.function.split('.')[-1]
        assert own in EXPECTED, f"{name} is in neither EXPECTED nor runs a function that is"
        seen.add(own)
        better, unit, span, refs = EXPECTED[own]
        assert (info.better, info.unit, info.range, info.refs) == (better, unit, span, refs), name
    assert seen == set(EXPECTED), set(EXPECTED) - seen


def test_every_entry_calls_what_it_should():
    for name, info in METRIC_INFO.items():
        assert (info.function, info.field) == CALLS.get(name, (f'kudio.{name}', None)), name


def test_every_public_function_of_a_checked_module_is_an_entry_or_left_out():
    """A metric added to one of these modules without an entry fails here,
    and so does one left out without saying so in NOT_METRICS."""
    unlisted = []
    for module_name in METRIC_MODULES:
        module = importlib.import_module(f'kudio.core.{module_name}')
        for name in module.__all__:
            if inspect.isfunction(getattr(module, name)) and \
                    name not in METRIC_INFO and name not in NOT_METRICS:
                unlisted.append(f'{module_name}.{name}')
    assert not unlisted, unlisted
    assert not NOT_METRICS & set(METRIC_INFO)


def test_names_and_functions():
    """An entry is a function under its own name, or a name score() takes
    that runs some other function."""
    for name, info in METRIC_INFO.items():
        assert info.name == name
        if info.function != f'kudio.{name}':
            assert name in METRICS, f"{name} runs {info.function} but score() " \
                                    f"has no such name"


def _resolve(path):
    assert path.startswith('kudio.'), path
    obj = kudio
    for part in path.split('.')[1:]:
        obj = getattr(obj, part)
    return obj


def _fields_of(result_type):
    if hasattr(result_type, '_fields'):
        return set(result_type._fields)
    return {f.name for f in dataclasses.fields(result_type)}


def test_each_entry_points_at_a_callable_in_the_installed_kudio():
    for name, info in METRIC_INFO.items():
        function = _resolve(info.function)
        assert callable(function), info.function
        if info.field is None:
            continue
        if info.function == 'kudio.score':
            # the dict score() returns, keyed by the name it was asked for
            assert info.field == name and name in METRICS, name
            continue
        # a function that returns a record: the field has to be one of its own
        returned = inspect.signature(function).return_annotation
        if isinstance(returned, str):
            returned = eval(returned, function.__globals__)
        assert info.field in _fields_of(returned), \
            f"{name}: {returned.__name__} has no field {info.field!r}"


def test_a_score_name_says_what_its_function_says():
    """``segsnr`` and ``segmental_snr``, ``csig`` and ``composite``: one
    metric, so one reading of it."""
    for name, info in METRIC_INFO.items():
        own = info.function.split('.')[-1]
        if own == name or own not in METRIC_INFO:
            continue
        other = METRIC_INFO[own]
        for attr in ('better', 'range', 'unit', 'variants', 'refs', 'validation'):
            assert getattr(info, attr) == getattr(other, attr), (name, attr)


def test_the_entries_are_well_formed():
    for name, info in METRIC_INFO.items():
        assert info.label, name
        assert info.better in ('higher', 'lower', None), name
        lo, hi = info.range
        assert lo is None or hi is None or lo < hi, name
        assert isinstance(info.unit, str), name
        assert info.variants and info.variants.endswith('.'), name
        assert isinstance(info.refs, tuple), name
        assert len(set(info.refs)) == len(info.refs), name
        unknown = [k for k in info.refs if k not in BIBTEX]
        assert not unknown, f"{name} cites {unknown}, which BIBTEX lacks"


def test_wss_is_not_described_by_lpc():
    """WSS reads band energies; the LPC settings belong to LLR, IS and CEP."""
    assert 'LPC' not in METRIC_INFO['wss'].variants
    for name in ('llr', 'itakura_saito', 'cepstral_distance'):
        assert 'LPC' in METRIC_INFO[name].variants, name


def test_bibtex_holds_what_the_registry_cites_and_nothing_else():
    cited = {k for info in METRIC_INFO.values() for k in info.refs}
    assert set(BIBTEX) == cited, set(BIBTEX) ^ cited


def test_the_registry_is_read_only():
    with pytest.raises(TypeError):
        METRIC_INFO['pesq'] = METRIC_INFO['snr']
    with pytest.raises(TypeError):
        BIBTEX['kudio'] = ''
    with pytest.raises(dataclasses.FrozenInstanceError):
        METRIC_INFO['pesq'].unit = 'dB'


def test_metric_info_takes_either_name():
    assert metric_info('pesq') is METRIC_INFO['pesq']
    assert metric_info('segsnr').function == 'kudio.segmental_snr'
    assert metric_info('csii_high').field == 'high'


def test_metric_info_refuses_an_unknown_name_and_lists_the_names():
    with pytest.raises(ValueError, match=r"^metric_info\(\): unknown metric 'PESQ'; "
                                         r"choose from \(.*'pesq'"):
        metric_info('PESQ')
    with pytest.raises(ValueError, match=r"^metric_info\(\): "):
        metric_info(['pesq'])


# --- BibTeX ---------------------------------------------------------------------

#: the fields each entry type has to carry, as BibTeX requires them; an
#: @misc needs none there, and gets an author and a title here
REQUIRED = {
    'article': {'author', 'title', 'journal', 'year'},
    'inproceedings': {'author', 'title', 'booktitle', 'year'},
    'book': {'author', 'title', 'publisher', 'year'},
    'techreport': {'author', 'title', 'institution', 'year'},
    'misc': {'author', 'title'},
}

_MONTHS = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
           'oct', 'nov', 'dec')
_MONTH_NAMES = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December')


def _braced(text, i):
    """The value of the braces opening at *i*, and where they close."""
    depth, j = 0, i
    while j < len(text):
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise AssertionError(f"unbalanced braces from {text[i:i + 40]!r}")


def _parse_bibtex(text):
    """``[(type, key, {field: value})]`` of a run of BibTeX entries, or an
    AssertionError saying where it stopped making sense."""
    head = re.compile(r'@([a-z]+)\{([A-Za-z0-9_:.-]+),')
    name_re = re.compile(r'([a-z]+)\s*=\s*')
    bare = re.compile(r'[A-Za-z0-9]+')
    entries, i = [], 0

    def skip_space(i, key):
        while i < len(text) and text[i].isspace():
            i += 1
        assert i < len(text), f"unbalanced braces: {key} never closes"
        return i

    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text):
            return entries
        m = head.match(text, i)
        assert m, f"no entry starts at {text[i:i + 40]!r}"
        kind, key = m.groups()
        fields, i = {}, m.end()
        while True:
            i = skip_space(i, key)
            if text[i] == '}':
                i += 1
                break
            m = name_re.match(text, i)
            assert m, f"{key}: no field at {text[i:i + 40]!r}"
            name, i = m.group(1), m.end()
            if text[i:i + 1] == '{':
                value, i = _braced(text, i)
            else:
                m = bare.match(text, i)
                assert m, f"{key}: {name} has no value"
                value, i = m.group(), m.end()
                assert value in _MONTHS or value.isdigit(), \
                    f"{key}: bare {name} {value!r} is no BibTeX macro"
            assert name not in fields, f"{key}: {name} twice"
            fields[name] = value
            i = skip_space(i, key)
            if text[i] == ',':
                i += 1
            else:
                assert text[i] == '}', f"{key}: no comma after {name}"
        entries.append((kind, key, fields))


def test_the_parser_finds_what_is_wrong():
    with pytest.raises(AssertionError, match='unbalanced'):
        _parse_bibtex("@misc{a,\n  title = {A {b},\n}")
    with pytest.raises(AssertionError, match='twice'):
        _parse_bibtex("@misc{a,\n  year = {1},\n  year = {2},\n}")
    with pytest.raises(AssertionError, match='no comma'):
        _parse_bibtex("@misc{a,\n  year = {1}\n  title = {t},\n}")
    assert _parse_bibtex('@misc{a,\n  title = {B{\\"u}r},\n  month = feb,\n}') == \
        [('misc', 'a', {'title': 'B{\\"u}r', 'month': 'feb'})]


def test_every_entry_parses_with_the_fields_its_type_needs():
    for key, text in BIBTEX.items():
        entries = _parse_bibtex(text)
        assert len(entries) == 1, key
        kind, parsed_key, fields = entries[0]
        assert parsed_key == key
        assert kind in REQUIRED, f"{key}: @{kind}"
        missing = REQUIRED[kind] - set(fields)
        assert not missing, f"{key} lacks {missing}"
        assert re.fullmatch(r'\d{4}', fields.get('year', '')), f"{key}: no year"
        # BibTeX's own "and" between names; a comma would make it read
        # "Last, First" and swap the parts of every name
        assert ', ' not in fields['author'], f"{key}: a comma in its authors"
        if 'month' in fields:
            # a macro, so that each style prints the month its own way
            assert re.search(rf"^  month = {fields['month']},$", text, re.M), \
                f"{key}: month in braces"


def test_every_entry_is_ascii():
    """LaTeX's spelling for accents and dashes: classic BibTeX handles it,
    and print(kudio.cite(...)) cannot fail on a legacy console."""
    for key, text in BIBTEX.items():
        assert text.isascii(), key


# --- the entries against the references the docs give ----------------------------

_LATEX = {'{\\"u}': 'ü', '{\\"o}': 'ö', '{\\ae}': 'æ', "{\\'e}": 'é'}


def _plain(value):
    """*value* as the docs print it: accents and dashes decoded, braces gone."""
    for tex, char in _LATEX.items():
        value = value.replace(tex, char)
    assert '\\' not in value, f"an escape this test cannot read in {value!r}"
    value = value.replace('---', '—').replace('--', '–')
    return value.replace('{', '').replace('}', '')


def _bare(title):
    """A title reduced to its letters and digits, for finding it."""
    return ''.join(c for c in unicodedata.normalize('NFKD', title.lower())
                   if c.isalnum())


def _as_the_docs_write(authors):
    names = [_plain(a) for a in authors.split(' and ')]
    if len(names) < 3:
        return ' and '.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def _word(value):
    """A pattern for *value* standing on its own: P.862 is not P.862.1."""
    return re.compile(r'(?<![\w.])' + re.escape(value) + r'(?!\.?\w)')


_TITLE = re.compile(r'"([^"]+)"|\*([^*]+)\*')
_DOI = re.compile(r'doi:(10\.\d{4,9}/\S+?)(?=[.,;]?(?:\s|$))')
_ARXIV = re.compile(r'arXiv:(\d{4}\.\d{4,5})')
_DATE_BEFORE = re.compile(r'\((?:(\d{2})/|(' + '|'.join(_MONTH_NAMES) + r') )(\d{4})\),?\s*$')


def _footnotes():
    """``(where, text)`` of every footnote on the English guide pages and the
    validation page."""
    guide = ROOT / 'docs' / 'guide'
    if not guide.is_dir():
        pytest.skip("docs/ only exists in a checkout")
    pages = [p for p in sorted(guide.glob('*.md')) if not p.name.endswith('.zh-TW.md')]
    pages.append(_checkout_file('docs', 'validation.md'))
    notes = []
    for page in pages:
        for number, line in enumerate(page.read_text(encoding='utf-8').splitlines(), 1):
            m = re.match(r'\[\^[^\]]+\]:\s*(.*)', line)
            if m:
                notes.append((f"{page.relative_to(ROOT).as_posix()}:{number}",
                               m.group(1)))
    return notes


def _citations():
    """``(where, note, before, title, after)`` for every work a footnote
    cites: the text since the previous title, the title, and the text up to
    the next one."""
    found = []
    for where, note in _footnotes():
        spans = [(m.start(), m.end(), m.group(1) or m.group(2))
                 for m in _TITLE.finditer(note)]
        for k, (start, end, title) in enumerate(spans):
            before = note[spans[k - 1][1] if k else 0:start]
            after = note[end:spans[k + 1][0] if k + 1 < len(spans) else len(note)]
            found.append((where, note, before, title, after))
    return found


def _unaccounted(after, venue, fields):
    """What the citation puts between *venue* and the year that *fields*
    does not hold: a place, a month."""
    start = after.index(venue) + len(venue)
    stop = after.find(fields['year'], start)
    known = {fields.get('address')}
    if 'month' in fields:
        known.add(_MONTH_NAMES[_MONTHS.index(fields['month'])])
    parts = [p.strip() for p in after[start:stop].split(',')]
    return [p for p in parts if p and p not in known]


def _mismatches(fields, note, before, title, after):
    """What in *fields* the citation does not say, and what it says that
    *fields* leaves out."""
    wrong = []
    around = before + after
    if _plain(fields['title']) != title:
        wrong.append(f"title {_plain(fields['title'])!r}")
    author = fields['author']
    if author.startswith('{'):
        if _plain(author) not in note:
            wrong.append(f"author {_plain(author)!r}")
    elif not before.rstrip().endswith(_as_the_docs_write(author) + ','):
        wrong.append(f"author {_as_the_docs_write(author)!r}")
    if not _word(fields['year']).search(around):
        wrong.append(f"year {fields['year']}")
    # a date just before the title, "(02/2001)" or "(November 2023)", is
    # the edition of a standard: its month and year
    date = _DATE_BEFORE.search(before)
    if date:
        month = _MONTHS[int(date.group(1)) - 1] if date.group(1) else \
            _MONTHS[_MONTH_NAMES.index(date.group(2))]
        if (fields.get('month'), fields['year']) != (month, date.group(3)):
            wrong.append(f"month and year {fields.get('month')} {fields['year']} "
                         f"where the footnote dates it {date.group(0).strip()}")
    elif 'month' in fields:
        written = f"{_MONTH_NAMES[_MONTHS.index(fields['month'])]} {fields['year']}"
        if written not in after:
            wrong.append(f"month {fields['month']}")
    if 'journal' in fields:
        issue = f"({fields['number'].replace('--', '-')})" if 'number' in fields else ''
        venue = f"{fields['journal']} {fields['volume']}"
        if venue + issue not in after:
            wrong.append(f"journal {venue + issue!r}")
        elif not issue and venue + '(' in after:
            wrong.append("no number where the footnote gives an issue")
    if 'booktitle' in fields:
        if fields['booktitle'] not in after:
            wrong.append(f"booktitle {fields['booktitle']!r}")
        else:
            wrong += [f"{p!r} left out" for p in
                      _unaccounted(after, fields['booktitle'], fields)]
    for name in ('institution', 'publisher'):
        if name in fields and fields[name] in after:
            wrong += [f"{p!r} left out" for p in _unaccounted(after, fields[name], fields)]
    if 'number' in fields and 'journal' not in fields and \
            not _word(fields['number']).search(before):
        wrong.append(f"number {fields['number']!r}")
    for name in ('institution', 'publisher', 'address', 'url', 'note'):
        if name in fields and fields[name] not in note:
            wrong.append(f"{name} {fields[name]!r}")
    # what the citation states has to be in the entry, and the other way round
    pages = re.search(r'pp\. (\d+)-(\d+)', after)
    stated = f"{pages.group(1)}--{pages.group(2)}" if pages else None
    if pages is None and 'journal' in fields:
        # a journal that numbers its papers rather than their pages: the
        # number stands between the issue and the year, "Phys. Rev. E
        # 69(6), 066138, 2004", and the entry keeps it as its pages
        article = re.search(re.escape(f"{fields['journal']} {fields['volume']}")
                            + r'(?:\([^)]*\))?, (\d+), ' + fields['year'], after)
        stated = article.group(1) if article else None
    if fields.get('pages') != stated:
        wrong.append(f"pages {fields.get('pages')}")
    volume = re.search(r'vol\. (\d+)', after)
    if volume and fields.get('volume') != volume.group(1):
        wrong.append(f"volume {fields.get('volume')} where the footnote says "
                     f"{volume.group(0)}")
    edition = re.search(r'(\d+(?:st|nd|rd|th)) ed\.', after)
    if fields.get('edition') and not edition:
        wrong.append(f"edition {fields['edition']}")
    if edition and fields.get('edition') != edition.group(1) and \
            edition.group(0) not in fields.get('note', ''):
        wrong.append(f"no edition where the footnote says {edition.group(0)}")
    doi = _DOI.search(after)
    if fields.get('doi') != (doi.group(1) if doi else None):
        wrong.append(f"doi {fields.get('doi')} where the docs give "
                     f"{doi.group(1) if doi else 'none'}")
    arxiv = _ARXIV.search(after)
    if fields.get('eprint') != (arxiv.group(1) if arxiv else None):
        wrong.append(f"eprint {fields.get('eprint')}")
    return wrong


def _matching(fields, citations):
    title = _bare(_plain(fields['title']))
    return [c for c in citations if _bare(c[3]) == title]


def test_every_entry_is_the_reference_the_docs_give():
    """Each entry has to match every footnote that cites the same title, so
    two pages that disagree about a work fail here as well."""
    citations = _citations()
    problems = []
    for key, text in BIBTEX.items():
        (_, _, fields), = _parse_bibtex(text)
        matches = _matching(fields, citations)
        if not matches:
            problems.append(f"{key}: no footnote cites {_plain(fields['title'])!r}")
        for where, note, before, title, after in matches:
            problems += [f"{key} against {where}: {w}"
                         for w in _mismatches(fields, note, before, title, after)]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize('key, field, value', [
    ('taal2011stoi', 'doi', '10.1109/TASL.2011.2114882'),
    ('taal2011stoi', 'pages', '2125--2137'),
    ('kitawaki1988coding', 'author', 'H. Nagabuchi and N. Kitawaki and K. Itoh'),
    ('brummer2006cllr', 'number', None),
    ('klatt1982wss', 'volume', None),
    ('loizou2013enhancement', 'edition', None),
    ('ansi1997s35', 'address', None),
    ('brummer2011bosaris', 'month', None),
    ('itut2001p862', 'month', 'mar'),
    ('itut2003p8621', 'number', 'P.862'),
    ('nist2010sre', 'note', 'version r5, 21 April 2010'),
    ('leroux2019sdr', 'title', '{SDR} - half-baked or well done?'),
    ('jurafsky2026slp', 'year', '2025'),
    ('kraskov2004mi', 'pages', None),
    ('kraskov2004mi', 'pages', '066139'),
])
def test_the_footnote_check_catches(key, field, value):
    """Each of these changes, made to an entry that passes, has to fail."""
    (_, _, fields), = _parse_bibtex(BIBTEX[key])
    citations = _matching(fields, _citations())
    assert citations and not any(_mismatches(fields, c[1], c[2], c[3], c[4])
                                 for c in citations)
    changed = dict(fields)
    if value is None:
        del changed[field]
    else:
        changed[field] = value
    assert all(_mismatches(changed, c[1], c[2], c[3], c[4]) for c in citations)


# --- the validation page ---------------------------------------------------------

def _slug(heading):
    """The id Python-Markdown's toc gives a heading, which mkdocs.yml leaves
    at its default slugify: accents dropped, punctuation but hyphens and
    underscores dropped, lower case, runs of spaces and hyphens one hyphen."""
    value = unicodedata.normalize('NFKD', heading).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def _sections(path):
    """``{heading: text under it}`` of a Markdown page, code fences skipped."""
    sections, heading, fenced = {}, None, False
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('```'):
            fenced = not fenced
            continue
        m = None if fenced else re.match(r'(#{1,6})\s+(.*?)\s*$', line)
        if m:
            heading = m.group(2)
            sections[heading] = heading
        elif heading is not None:
            sections[heading] += '\n' + line
    return sections


def test_every_validation_anchor_is_a_heading_of_the_english_page():
    mkdocs = _checkout_file('mkdocs.yml').read_text(encoding='utf-8')
    assert 'slugify' not in mkdocs, \
        "mkdocs.yml sets its own slugify; _slug has to follow it"
    headings = list(_sections(_checkout_file('docs', 'validation.md')))
    try:
        from markdown.extensions.toc import slugify
    except ImportError:
        pass
    else:
        assert [_slug(h) for h in headings] == [slugify(h, '-') for h in headings]
    ids = {_slug(h) for h in headings}
    for name, info in METRIC_INFO.items():
        if info.validation is None:
            continue
        page, _, anchor = info.validation.partition('#')
        assert page == 'validation.md', name
        assert anchor in ids, f"{name}: validation.md has no #{anchor}"


def _spoken(text):
    return re.sub(r'[_\-\s]+', ' ', text.lower())


def test_every_validation_section_names_its_metric():
    """The anchor has to lead to the section about this metric: the section
    names it, by its name, its function, its label or an acronym of the
    label."""
    by_id = {_slug(h): text for h, text in
             _sections(_checkout_file('docs', 'validation.md')).items()}
    for name, info in METRIC_INFO.items():
        if info.validation is None:
            continue
        text = _spoken(by_id[info.validation.partition('#')[2]])
        names = {name, info.function.split('.')[-1], info.label}
        names |= set(re.findall(r'\b[A-Z][A-Za-z0-9_]*[A-Z0-9]\b', info.label))
        assert any(re.search(r'(?<!\w)' + re.escape(_spoken(n)) + r'(?!\w)', text)
                   for n in names), f"{name}: {info.validation} does not name it"


def test_the_chinese_page_gives_its_sections_the_same_ids():
    text = _checkout_file('docs', 'validation.zh-TW.md').read_text(encoding='utf-8')
    ids = set(re.findall(r'^#{1,6} .*\{#([^}\s]+)\}\s*$', text, re.M))
    for name, info in METRIC_INFO.items():
        if info.validation is not None:
            assert info.validation.partition('#')[2] in ids, name


def test_only_the_measures_the_page_does_not_cover_lack_a_section():
    assert {n for n, i in METRIC_INFO.items() if i.validation is None} == \
        NO_VALIDATION_SECTION


# --- cite() ----------------------------------------------------------------------

def test_cite_alone_gives_kudio():
    """@misc with the version in a note: plain.bst and IEEEtran.bst know no
    @software and would drop the version."""
    text = cite()
    (kind, key, fields), = _parse_bibtex(text)
    assert (kind, key) == ('misc', 'kudio')
    assert fields == {'author': 'RL', 'title': 'Kudio',
                      'note': f'Version {kudio.__version__}',
                      'url': 'https://github.com/recklight/Kudio'}
    assert text.endswith('}\n')
    assert cite([]) == text


def test_cite_puts_kudio_first_then_each_work_once_in_the_order_met():
    names = ['csii_high', 'ncm', 'csii', 'stoi', 'csii_low']
    keys = [key for _, key, _ in _parse_bibtex(cite(names))]
    assert keys == ['kudio', 'ma2009ncm', 'kates2005csii', 'ansi1997s35',
                    'loizou2013enhancement', 'taal2011stoi']
    expected = ['kudio']
    for name in names:
        expected += [k for k in METRIC_INFO[name].refs if k not in expected]
    assert keys == expected


def test_cite_gives_the_entries_as_bibtex_has_them():
    text = cite(['pesq'])
    for key in METRIC_INFO['pesq'].refs:
        assert text.count(BIBTEX[key]) == 1
    assert len(_parse_bibtex(cite(list(METRIC_INFO)))) == len(BIBTEX) + 1


def test_cite_takes_a_name_or_a_list_and_either_spelling():
    assert cite('pesq') == cite(['pesq']) == cite(('pesq',))
    assert cite('is') == cite('itakura_saito')
    assert cite('csig') == cite('composite')
    assert cite('segsnr') == cite('segmental_snr') == cite()      # cites nothing


def test_cite_refuses_unknown_names_all_at_once():
    with pytest.raises(ValueError, match=r"^cite\(\): unknown metric\(s\) "
                                         r"\['nope', 'PESQ'\]; choose from"):
        cite(['pesq', 'nope', 'PESQ'])
    for bad in (3, [None], ['pesq', 1.5]):
        with pytest.raises(ValueError, match=r"^cite\(\): "):
            cite(bad)


def test_cite_refuses_a_set_and_bytes_by_name():
    """A set's order changes from run to run, and with it the bibliography."""
    for bad in ({'pesq', 'stoi'}, frozenset(['pesq'])):
        with pytest.raises(ValueError, match=r"^cite\(\): .*not a set"):
            cite(bad)
    with pytest.raises(ValueError, match=r"^cite\(\): a metric name is text, got bytes"):
        cite(b'pesq')


def test_cite_of_everything_is_ascii():
    cite(list(METRIC_INFO)).encode('ascii')


# --- CITATION.cff ------------------------------------------------------------------

def _pyproject(pattern):
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    m = re.search(pattern, text, re.M)
    assert m, pattern
    return m.group(1)


def test_citation_cff_names_this_version():
    path = ROOT / 'CITATION.cff'
    if not path.is_file():
        if not (ROOT / 'mkdocs.yml').is_file():
            pytest.skip("CITATION.cff only exists in a checkout")
        pytest.fail("CITATION.cff is missing from the checkout")
    yaml = pytest.importorskip('yaml')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert data['cff-version'] == '1.2.0'
    assert data['version'] == kudio.__version__
    assert data['type'] == 'software'
    assert data['license'] == 'MIT'
    # what cite() says about kudio, CITATION.cff says too; RL is a person
    # who goes by initials, which CFF spells as an alias
    (_, _, fields), = _parse_bibtex(cite())
    assert data['title'] == fields['title']
    assert data['authors'] == [{'alias': fields['author']}]
    assert fields['note'] == f"Version {data['version']}"
    assert data['repository-code'] == fields['url']
    # and what pyproject says
    assert data['abstract'] == _pyproject(r'^description = "(.*)"$')
    assert data['keywords'] == re.findall(r'"([^"]+)"',
                                          _pyproject(r'^keywords = \[(.*)\]$'))
    assert data['url'] == _pyproject(r'^Documentation = "(.*)"$')


def test_zenodo_json_says_what_citation_cff_says():
    """Zenodo reads .zenodo.json and ignores CITATION.cff when both are there,
    and it takes only part of the CFF schema, an author given by alias not
    among it; GitHub's "Cite this repository" reads CITATION.cff. The two
    have to say the same."""
    path = ROOT / '.zenodo.json'
    if not path.is_file():
        if not (ROOT / 'mkdocs.yml').is_file():
            pytest.skip(".zenodo.json only exists in a checkout")
        pytest.fail(".zenodo.json is missing from the checkout")
    yaml = pytest.importorskip('yaml')
    cff = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    zenodo = json.loads(path.read_text(encoding='utf-8'))
    assert zenodo['title'] == cff['title']
    assert zenodo['description'] == cff['abstract']
    assert zenodo['creators'] == [{'name': a['alias']} for a in cff['authors']]
    assert zenodo['license'] == cff['license'].lower()
    assert zenodo['keywords'] == cff['keywords']
    assert zenodo['upload_type'] == 'software' and zenodo['access_right'] == 'open'
    # the version comes from the release Zenodo archives, so none is written
    assert 'version' not in zenodo
