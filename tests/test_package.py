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
    # quality measures, with a reference and without
    'lsd', 'llr', 'itakura_saito', 'cepstral_distance', 'wss',
    'fw_segmental_snr', 'composite', 'CompositeScore', 'srmr', 'wada_snr',
    # intelligibility, and what a recognizer, a speaker verifier or a pitch
    # tracker got wrong
    'ncm', 'csii', 'CsiiScore', 'siib', 'siib_gauss',
    'wer', 'cer', 'word_errors', 'char_errors',
    'ErrorCounts', 'eer', 'min_dcf', 'det_curve', 'DetCurve',
    'gross_pitch_error', 'voicing_decision_error',
    'f0_frame_error', 'fine_pitch_error', 'f0_rmse', 'f0_errors', 'F0Errors',
    # separation
    'si_sdr_improvement', 'sdr_improvement', 'pit_si_sdr', 'PitScore',
    'bss_eval', 'BssEval',
    # calibration of a verifier's scores
    'cllr', 'min_cllr', 'act_dcf', 'c_primary',
    # mel-cepstral distortion
    'mcd',
    # who spoke when
    'der', 'diarization_errors', 'jer', 'detection_error_rate', 'load_rttm',
    # a listener's opinion, predicted
    'dnsmos', 'dnsmos_p808', 'DnsmosScore', 'find_dnsmos_model',
    # what each metric is, and what to cite for it
    'METRIC_INFO', 'MetricInfo', 'metric_info', 'BIBTEX', 'cite',
    # how a result was computed, in one line beside it
    'signature', 'parse_signature',
    # how far a result over a test set could move, and whether a gap is real
    'bootstrap_ci', 'paired_ci', 'rate_ci', 'paired_rate_ci', 'ConfidenceInterval',
    'permutation_test', 'PermutationResult', 'holm',
    # a whole test set, from the files to the table under a paper
    'score_dataset', 'DatasetScore', 'MetricSummary', 'REFERENCE_FREE',
]


def test_version_is_v4():
    assert kudio.__version__.startswith("4.")


def test_top_level_names_present():
    missing = [n for n in TOP_LEVEL_NAMES if not hasattr(kudio, n)]
    assert not missing, f"missing top-level names: {missing}"


def test_all_is_curated():
    for name in kudio.__all__:
        assert hasattr(kudio, name), name


def test_import_is_lightweight():
    """`import kudio` must not drag in matplotlib / pandas, or librosa, which
    no metric needs. Nor scipy.signal, left to the functions that use it, or pywt,
    which nothing in kudio imports. Nor tqdm, 50 ms on its own, which only the
    batch helpers' progress bars need.

    Uses a subprocess so other test modules (which import pandas at collection
    time) don't pollute the check.
    """
    import subprocess
    import sys
    code = ("import kudio, sys; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib leaked'; "
            "assert 'pandas' not in sys.modules, 'pandas leaked'; "
            "assert 'librosa' not in sys.modules, 'librosa leaked'; "
            "assert 'soxr' not in sys.modules, 'soxr leaked'; "
            "assert 'scipy.signal' not in sys.modules, 'scipy.signal leaked'; "
            "assert 'pywt' not in sys.modules, 'pywt leaked'; "
            "assert 'tqdm' not in sys.modules, 'tqdm leaked'")
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


def test_every_install_command_for_an_extra_quotes_it():
    """Unquoted, zsh reads `kudio[audio]` as a file pattern and stops with
    `no matches found`, so the hint, the README and the pages all quote it."""
    import re

    from kudio.exceptions import DependencyError
    assert str(DependencyError('sounddevice', 'audio')).endswith(
        'Install it with: pip install "kudio[audio]"')
    root = Path(__file__).resolve().parents[1]
    files = [root / "README.md", *(root / "docs").rglob("*.md"),
             *(root / "kudio").rglob("*.py")]
    bare = [f"{p.relative_to(root)}:{n}"
            for p in files
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"pip install (-U )?kudio\[", line)]
    assert not bare, bare


def _imports(*folder):
    """``{package: files}`` of every import under a folder of the project, read from the
    source: one inside a function no test reaches counts too."""
    import ast

    root = Path(__file__).resolve().parents[1]
    found = {}
    for path in sorted(root.joinpath(*folder).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            elif (isinstance(node, ast.Call) and node.args
                  and getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                  in ('import_module', '__import__', 'find_spec')):
                names = [getattr(node.args[0], 'value', None) or '']
            for name in names:
                found.setdefault(str(name).split('.')[0], set()).add(path.name)
    return found


#: what kudio imports and by which name pip knows it, and for what: None is always
DECLARED = {
    'numpy': ('numpy', None), 'scipy': ('scipy', None), 'soundfile': ('soundfile', None),
    'soxr': ('soxr', None), 'yaml': ('PyYAML', None), 'tqdm': ('tqdm', None),
    'sounddevice': ('sounddevice', 'audio'),
    'pesq': ('pesq', 'pesq'),
    'onnxruntime': ('onnxruntime', 'dnsmos'), 'matplotlib': ('matplotlib', 'viz'),
    'pandas': ('pandas', 'data'), 'openpyxl': ('openpyxl', 'data'),
    'audioread': ('audioread', 'compressed'),
}


def test_kudio_imports_what_it_declares_and_declares_what_it_imports():
    """A package that is there because another one brought it goes when that one
    goes."""
    import sys

    if not hasattr(sys, 'stdlib_module_names'):
        pytest.skip("which modules are Python's own is known from 3.10 on")
    found = set(_imports("kudio")) - set(sys.stdlib_module_names) - {'kudio', ''}
    assert found <= set(DECLARED)
    assert set(DECLARED) - found <= {'openpyxl'}                       # pandas' writer
    name = re.compile(r'[A-Za-z][\w-]*')
    core = {name.match(spec).group() for spec in re.findall(r'"([^"]+)"', _core_dependencies())}
    assert core == {pip for pip, extra in DECLARED.values() if extra is None}
    extras = _extras()
    for pip, extra in DECLARED.values():
        if extra is not None:
            assert pip in {name.match(spec).group() for spec in extras[extra]}, (pip, extra)


@pytest.mark.parametrize("folder", [("kudio",), ("tools", "docs")], ids="/".join)
def test_librosa_is_not_needed(folder):
    """Nor anything that came with it. Not by kudio, and not by the figures of the
    documentation, for which make_figures.py asks for the [viz] extra alone."""
    found = _imports(*folder)
    assert len(found) > 5
    assert not {'librosa', 'numba', 'llvmlite', 'sklearn', 'joblib', 'resampy', 'samplerate',
                'pooch', 'lazy_loader', 'msgpack', 'decorator'} & set(found)


@pytest.mark.parametrize("args", [('soxr',), ('pesq', 'pesq')])
def test_a_dependency_error_crosses_a_process_boundary_intact(args):
    """A pool worker hands its exception back pickled, and the first call to
    soxr can happen in a Synthesizer worker."""
    import pickle

    from kudio.exceptions import DependencyError
    raised = DependencyError(*args)
    back = pickle.loads(pickle.dumps(raised))
    assert type(back) is DependencyError
    assert (str(back), back.package, back.extra) == (str(raised), raised.package,
                                                     raised.extra)


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


def test_no_core_dependency_has_a_ceiling():
    """A cap set too low is a flat resolver failure for anyone who needs the newer
    release, and a quiet downgrade for everyone else. The one kudio had, on numpy,
    was numba's."""
    for spec in re.findall(r'"([^"]+)"', _core_dependencies()):
        assert re.fullmatch(r'[A-Za-z]+>=[\d.]+', spec), spec
    assert _requirement('numpy') == 'numpy>=1.24'


def test_the_extra_for_compressed_files_is_one_that_imports():
    """audioread 3.0.1 installs on Python 3.13 and does not import there: aifc is gone."""
    extras = _extras()
    assert extras['compressed'] == ["audioread>=3.0", "audioread>=3.1; python_version >= '3.13'"]
    assert 'compressed' in extras['all'][0]
    steps = _workflow()['jobs']['compressed']['steps']
    assert any('compressed' in _installed_extras(step.get('run', '')) for step in steps)


def test_librosa_is_there_to_be_measured_against_and_for_nothing_else():
    extras = _extras()
    assert 'librosa>=0.11' in extras['check']
    for name, specs in extras.items():
        assert name == 'check' or not any('librosa' in spec for spec in specs), name


def test_pywavelets_is_there_to_be_measured_against_and_for_nothing_else():
    """The wavelet filter has its own transform; PyWavelets is what it is held to."""
    extras = _extras()
    assert 'PyWavelets>=1.4' in extras['check']
    assert 'PyWavelets' not in _core_dependencies()
    for name, specs in extras.items():
        assert name == 'check' or not any('PyWavelets' in spec for spec in specs), name
    assert 'pywt' not in _imports('kudio')


def test_pystoi_is_there_to_be_measured_against_and_eval_is_empty():
    extras = _extras()
    assert 'pystoi>=0.3.3' in extras['check'] and extras['eval'] == []
    for name, specs in extras.items():
        assert name == 'check' or not any('pystoi' in spec for spec in specs), name
    assert 'pystoi' not in _imports('kudio')
    job = _workflow()['jobs']['librosa']
    assert job['env']['KUDIO_REQUIRE_PYSTOI'] == '1'
    assert 'tests/test_against_pystoi.py' in '\n'.join(s.get('run', '') for s in job['steps'])


def test_mir_eval_is_there_to_be_measured_against_and_for_nothing_else():
    extras = _extras()
    assert 'mir_eval>=0.7,<0.9' in extras['check']
    for name, specs in extras.items():
        assert name == 'check' or not any('mir_eval' in spec for spec in specs), name
    assert 'mir_eval' not in _imports('kudio')
    job = _workflow()['jobs']['librosa']
    assert job['env']['KUDIO_REQUIRE_MIR_EVAL'] == '1'
    assert 'tests/test_against_mir_eval.py' in '\n'.join(s.get('run', '') for s in job['steps'])


def test_the_comparison_with_pywavelets_runs_and_cannot_skip():
    job = _workflow()['jobs']['librosa']
    assert job['env']['KUDIO_REQUIRE_PYWAVELETS'] == '1'
    runs = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'tests/test_against_pywavelets.py' in runs and 'check' in _installed_extras(runs)


def test_the_oldest_leg_runs_on_the_floors():
    """Each of them, or a floor is a claim nothing in the project tries."""
    runs = '\n'.join(step.get('run', '') for step in _workflow()['jobs']['oldest']['steps'])
    pinned = dict(re.findall(r'"([A-Za-z0-9]+)==([\d.]+)', runs))
    extras = _extras()
    for spec in (re.findall(r'"([^"]+)"', _core_dependencies())
                 + extras['viz'] + extras['data'] + extras['audio']):
        name, floor = re.fullmatch(r'([A-Za-z0-9]+)>=([\d.]+)', spec).groups()
        assert pinned[name].rstrip('.') == floor, (spec, pinned.get(name))
    assert 'pytest' in runs


def test_a_release_waits_for_every_job_that_runs_the_suite():
    jobs = _workflow()['jobs']
    for name, job in jobs.items():
        needs = job.get('needs', [])
        for other in [needs] if isinstance(needs, str) else needs:
            assert other in jobs, f"{name} needs {other}, which is no job"
    suites = {name for name, job in jobs.items()
              if any('pytest' in step.get('run', '') for step in job.get('steps', []))}
    assert suites == set(jobs['publish']['needs']) == {'test', 'oldest', 'librosa', 'backends',
                                                       'compressed'}


def test_kudio_is_compared_with_both_librosas():
    """0.11 is what Python 3.11 and earlier get, 1.0 what 3.12 and later do. Where there
    is no librosa the comparison skips, so the job has to say it needs one."""
    job = _workflow()['jobs']['librosa']
    legs = job['strategy']['matrix']['include']
    assert {leg['librosa'] for leg in legs} == {'0.11', '1.0'}
    assert any(leg['os'].startswith('windows') for leg in legs)
    assert job['env']['KUDIO_REQUIRE_LIBROSA'] == '${{ matrix.librosa }}'
    # Windows' own shell goes by the last line of a step alone
    assert job['defaults']['run']['shell'] == 'bash' and job['env']['PYTHONUTF8'] == '1'


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


def test_the_docs_build_stays_on_mkdocs_1():
    """MkDocs 2.0 drops the plugin system Material and mkdocstrings are built
    on. Material 9 holds it back today; the cap in [docs] means the site does
    not depend on that staying true."""
    pins = [s.replace(' ', '') for s in _extras().get('docs', [])
            if re.match(r'mkdocs\s*[<>=!~]', s)]
    assert pins and '<2' in pins[0], f"[docs] lets pip install MkDocs 2: {pins}"


def test_a_mistyped_address_gets_the_english_404_page(tmp_path):
    """Each language's build writes 404.html at the root, the Chinese one
    last, and GitHub Pages shows that file for any address it does not have."""
    pytest.importorskip("markdown")             # the hooks run under the docs build
    import importlib.util

    spec = importlib.util.spec_from_file_location("kudio_docs_hooks",
                                                  ROOT / "tools" / "docs" / "hooks.py")
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    page = tmp_path / "404.html"
    page.write_text('<html lang="en">not found', encoding="utf-8")
    hooks.on_post_build({"site_dir": str(tmp_path), "theme": {"language": "en"}})
    page.write_text('<html lang="zh-TW">找不到', encoding="utf-8")
    hooks.on_post_build({"site_dir": str(tmp_path), "theme": {"language": "zh-TW"}})
    assert page.read_text(encoding="utf-8") == '<html lang="en">not found'


def test_the_documentation_link_is_where_the_site_is_published():
    """PyPI shows pyproject's Documentation URL; mkdocs.yml says which address
    the site is built for. Two places naming one address have to agree."""
    site_url = re.search(r'^site_url:\s*(\S+)\s*$', _read('mkdocs.yml'), re.M)
    assert site_url, "mkdocs.yml has no site_url"
    assert _urls().get('Documentation') == site_url.group(1), (
        _urls().get('Documentation'), site_url.group(1))


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


def test_live_audio_is_sounddevice_alone():
    """One binding of PortAudio and one numbering of the devices. On Linux it
    needs the system's PortAudio, which a CI leg installs to run the suite with it."""
    assert _extras()['audio'] == ['sounddevice>=0.4.5']
    assert 'pyaudio' not in _imports('kudio')
    for name, job in _workflow()['jobs'].items():
        runs = '\n'.join(step.get('run', '') for step in job.get('steps', []))
        if 'audio' in _installed_extras(runs):
            assert 'libportaudio2' in runs and 'pytest' in runs, name
            return
    pytest.fail("no CI job installs [audio]")


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


# -- helpers leaving the top level ----------------------------------------------
@pytest.mark.parametrize("name", ["error", "info", "warn", "red", "h1", "color_test"])
def test_console_colour_helpers_warn_from_the_top_level(name, capsys):
    with pytest.warns(DeprecationWarning, match=re.escape(f"kudio.util.{name}()")):
        getattr(kudio, name)("x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(kudio.util, name)("x")          # where they stay, silent


def test_upercase_points_at_uppercase():
    with pytest.warns(DeprecationWarning, match=r"kudio\.uppercase"):
        assert kudio.upercase(["a", 1]) == ["A"]
    with pytest.warns(DeprecationWarning, match=r"kudio\.util\.uppercase"):
        assert kudio.util.upercase(["a"]) == ["A"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert kudio.uppercase(["a"]) == ["A"]


# -- which except clause catches a refusal --------------------------------------
def test_a_feature_error_is_a_value_error_too():
    """vad refuses a rate with a FeatureError and lsd with a ValueError; which
    of the two it is depends on the module, and a caller writes one except."""
    y = np.zeros(1600)
    for refused in (lambda: kudio.vad(y, 0), lambda: kudio.lsd(y, y, 0)):
        with pytest.raises(ValueError):
            refused()
    with pytest.raises(kudio.KudioError):
        kudio.vad(y, 0)
    assert issubclass(kudio.FeatureError, kudio.KudioError)
    assert not issubclass(kudio.SynthesisError, ValueError)
    assert not issubclass(kudio.AudioIOError, ValueError)


def test_a_feature_error_comes_back_from_a_worker_process():
    import pickle
    back = pickle.loads(pickle.dumps(kudio.FeatureError("too short")))
    assert type(back) is kudio.FeatureError and str(back) == "too short"
