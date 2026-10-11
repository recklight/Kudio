# -*- coding: utf-8 -*-
"""Package surface: version, public names, deprecation aliases, packaging."""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

import kudio

# names the app / training layer import from the top level (must stay working)
TOP_LEVEL_NAMES = [
    'AudioBuffer', 'AudioDataManager', 'AudioEvaluate', 'CheckDevice',
    'FetchBuffer', 'LocalStreamReader', 'Recorder', 'RemoteStreamReader',
    'Synthesizer', 'Timer', 'check_input', 'file_load', 'save_wave',
    'play_audio', 'record', 'trad_enhance', 'wavelet_low_pass_filter',
    'warn', 'gre', 'pur', 'fGre',
    # v3 canonical feature names
    'waveform_to_spectrogram', 'spectrogram_to_waveform', 'file_to_spectrogram',
    'save_spectrogram_as_wave', 'mfcc', 'mfcc_from_files', 'logspec_from_files',
    'melspectrogram',
    # v3 effects
    'trim_silence', 'split_on_silence', 'time_stretch', 'pitch_shift',
    'add_noise_snr', 'spec_augment',
    # v3 metrics + exceptions
    'si_sdr', 'snr', 'segmental_snr', 'KudioError', 'AudioIOError',
]


def test_version_is_v3():
    assert kudio.__version__.startswith("3.")


def test_top_level_names_present():
    missing = [n for n in TOP_LEVEL_NAMES if not hasattr(kudio, n)]
    assert not missing, f"missing top-level names: {missing}"


def test_all_is_curated():
    for name in kudio.__all__:
        assert hasattr(kudio, name), name


def test_import_is_lightweight():
    """`import kudio` must not drag in matplotlib / pandas.

    Uses a subprocess so other test modules (which import pandas at collection
    time) don't pollute the check.
    """
    import subprocess
    import sys
    code = ("import kudio, sys; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib leaked'; "
            "assert 'pandas' not in sys.modules, 'pandas leaked'")
    subprocess.check_call([sys.executable, "-c", code])


def test_deprecated_aliases_warn():
    y = np.zeros(4096, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kudio.w2s(y)                      # deprecated alias of waveform_to_spectrogram
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_alias_result_matches_canonical():
    y = np.random.RandomState(0).randn(4096).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_array_equal(kudio.w2s(y), kudio.waveform_to_spectrogram(y))


def test_optional_dep_error_is_typed():
    # data2xlsx lives in an extra-gated submodule, not the top level
    assert not hasattr(kudio, 'data2xlsx')
    from kudio.exceptions import DependencyError
    assert issubclass(DependencyError, kudio.KudioError)


# =========================================================== packaging
#
# Everything below reads the project's own files rather than the installed
# metadata, because `pip install -e .` freezes that metadata at install time:
# an editable install three edits old still reports the old pins, so a test
# built on it would pass over exactly the staleness it is meant to catch.
#
# The files are read as text and picked apart with regexes. tomllib landed in
# 3.11 and 3.9 is this package's floor, and pulling `tomli` in for the test
# suite alone is a worse trade than a handful of patterns over a file this
# small.

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    """Text of a file that every form of this project carries."""
    path = ROOT / name
    assert path.is_file(), f"{name} is missing from {ROOT}"
    return path.read_text(encoding='utf-8')


def _read_from_checkout(name):
    """Text of a file that only a git checkout has, skipping elsewhere.

    The sdist deliberately carries the suite but not the CI workflow or the
    docs config, so these checks have nothing to look at when the tests run
    from an unpacked release.
    """
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} only exists in a checkout")
    return path.read_text(encoding='utf-8')


def _core_dependencies():
    block = re.search(r'^dependencies = \[\n(.*?)^\]', _read('pyproject.toml'),
                      re.S | re.M)
    assert block, "no core `dependencies` array in pyproject.toml"
    return block.group(1)


def _requirement(name):
    """The core requirement string for *name*, e.g. ``numpy>=1.24,<2.6``."""
    for spec in re.findall(r'"([^"]+)"', _core_dependencies()):
        if spec.lower().startswith(name.lower()):
            return spec
    raise AssertionError(f"{name} is not a core dependency any more")


def _notes_on(spec):
    """The comment lines that belong to requirement *spec*.

    A pin and the reason for it drift apart silently, so the reason has to be
    checkable: that means knowing which comment goes with which line. The ones
    directly above it, plus anything after the # on its own line.
    """
    lines = _core_dependencies().splitlines()
    index = next(i for i, line in enumerate(lines) if spec in line)
    notes = [lines[index].partition('#')[2]]
    while index > 0 and lines[index - 1].lstrip().startswith('#'):
        index -= 1
        notes.append(lines[index])
    return '\n'.join(notes)


def _extras():
    """``{extra: [requirement, ...]}`` from [project.optional-dependencies]."""
    section = re.search(r'^\[project\.optional-dependencies\]\n(.*?)(?=^\[)',
                        _read('pyproject.toml'), re.S | re.M)
    assert section, "no [project.optional-dependencies] table in pyproject.toml"
    extras = {}
    for line in section.group(1).splitlines():
        head = re.match(r'([A-Za-z][\w-]*) *= *\[(.*)\]', line)
        if head:
            extras[head.group(1)] = re.findall(r'"([^"]+)"', head.group(2))
    return extras


def _urls():
    section = re.search(r'^\[project\.urls\]\n(.*?)(?=^\[)',
                        _read('pyproject.toml'), re.S | re.M)
    assert section, "no [project.urls] table in pyproject.toml"
    return dict(re.findall(r'^(\w+) *= *"([^"]+)"', section.group(1), re.M))


def _workflow():
    return yaml.safe_load(_read_from_checkout('.github/workflows/ci.yml'))


def _installed_extras(run):
    """Extra names from every ``pip install ... .[a,b]`` in a shell step."""
    found = set()
    for match in re.finditer(r'pip install[^\n]*\.\[([^\]]+)\]', run):
        found |= {name.strip() for name in match.group(1).split(',')}
    return found


def test_numpy_is_capped_where_numba_caps_it():
    """librosa drags numba in, and numba's own numpy bound is the real ceiling.

    numpy 2.5 needs 3.12, so a cap set too low does nothing on 3.9-3.11 and
    then bites on 3.12 and 3.13: a flat resolver failure for anyone who needs
    the newer numpy, and a quiet downgrade for everyone else.
    """
    spec = _requirement('numpy')
    assert '<2.6' in spec, spec
    assert '>=1.24' in spec, spec              # scipy/librosa floor, not numba's

    # And the reason has to move with the pin: a comment naming a version
    # other than the one in force sends the next reader the wrong way.
    ceiling = re.search(r'<(\d+\.\d+)', spec).group(1)
    notes = _notes_on(spec)
    assert ceiling in notes, (
        f"the numpy pin stops at {ceiling} but its comment does not say so:\n"
        f"{notes}")


def test_published_urls_name_a_branch_that_exists():
    """The default branch here is master; /blob/main/ links are 404s.

    PyPI freezes these per release, so a wrong one cannot be taken back -- it
    is wrong for that version for good.
    """
    urls = _urls()
    wrong = sorted(k for k, v in urls.items()
                   if re.search(r'/(?:blob|tree|raw)/main/', v))
    assert not wrong, f"these point at a branch this repository has not got: {wrong}"
    assert urls['Changelog'].endswith('/blob/master/CHANGELOG.md'), urls['Changelog']


def test_the_sdist_takes_the_whole_tests_directory():
    """Shipping tests/test*.py and nothing else ships a suite that cannot run.

    Six modules import their fixtures from tests/conftest.py, which is not a
    test file and so falls outside setuptools' default rule. Grafting the
    directory is what keeps the sdist's suite collectable.
    """
    assert re.search(r'^\s*graft\s+tests\s*$', _read('MANIFEST.in'), re.M), (
        "MANIFEST.in does not graft tests/, so conftest.py is left out of the sdist")


def test_the_loudness_cross_check_has_somewhere_to_come_from():
    """kudio implements BS.1770 itself, so pyloudnorm is the only outside
    check that the numbers come out right. An extra nobody can install is an
    extra that skips everywhere, which is the same as not having the check."""
    extras = _extras()
    assert 'check' in extras, "no [check] extra for the cross-check dependencies"
    assert any(spec.startswith('pyloudnorm') for spec in extras['check']), extras['check']
    assert 'check' not in ' '.join(extras.get('dev', [])), (
        "[dev] has to stay installable anywhere; leave the cross-checks to CI")


def test_the_docs_toolchain_is_declared_somewhere_pip_can_read_it():
    """A clean checkout has to be able to build the docs, which means the
    theme and plugin mkdocs.yml selects need a dependency group to live in and
    not just a line in a comment for the reader to retype."""
    docs = ' '.join(_extras().get('docs', []))
    assert 'mkdocs-material' in docs, docs       # the theme mkdocs.yml selects
    assert 'mkdocstrings' in docs, docs          # the plugin mkdocs.yml selects


def test_ci_actions_are_off_the_deprecated_node_runtime():
    """checkout and setup-python below v7 run on Node 20, which is retired."""
    stale = re.findall(r'uses: *(actions/(?:checkout|setup-python)@v[1-6])\b',
                       _read_from_checkout('.github/workflows/ci.yml'))
    assert not stale, f"these action versions run on a retired Node: {sorted(set(stale))}"


def test_some_ci_leg_installs_the_optional_backends_and_runs_the_suite():
    """The mir_eval and pystoi branches of the evaluator only execute where
    those are installed. Without a leg that installs them, the mir_eval <0.9
    bound is a claim nothing in the project ever tries."""
    for name, job in _workflow()['jobs'].items():
        steps = job.get('steps', [])
        installed = set()
        for step in steps:
            installed |= _installed_extras(step.get('run', ''))
        if {'eval', 'check'} <= installed:
            assert any('pytest' in step.get('run', '') for step in steps), (
                f"job {name} installs the back-ends but never runs the suite")
            return
    pytest.fail("no CI job installs [eval] and [check] and then runs the suite")


def test_the_release_checks_the_artifacts_before_uploading_them():
    """Nothing can be taken back off PyPI, so every check belongs in front of
    the upload step rather than after it."""
    steps = _workflow()['jobs']['publish']['steps']
    upload = next(i for i, step in enumerate(steps)
                  if 'pypi-publish' in step.get('uses', ''))
    before = '\n'.join(step.get('run', '') for step in steps[:upload])

    assert 'twine check --strict' in before
    assert 'walk_packages' in before, "the built wheel is never import-walked"
    assert 'GITHUB_REF' in before, "the tag is never compared with _version.py"
