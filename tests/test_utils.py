# -*- coding: utf-8 -*-
import types

import pytest

from kudio import find_duplicate, lowercase, timer
from kudio.util.colors import color_test
from kudio.util.conv import data2xlsx, uppercase


def test_timer_decorator_passthrough():
    @timer
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_find_duplicate():
    counts = find_duplicate(['a', 'b', 'a'])
    assert counts == {'a': 2, 'b': 1}


def test_case_helpers():
    assert lowercase(['A', 'b', 1]) == ['a', 'b']
    assert uppercase(['a', 'B', None]) == ['A', 'B']


def test_color_test_runs():
    assert color_test('test') is True


def test_data2xlsx_new_and_append(tmp_path):
    pd = pytest.importorskip("pandas")   # data2xlsx needs the [data] extra
    pytest.importorskip("openpyxl")
    xlsx = tmp_path / "out" / "scores.xlsx"
    assert data2xlsx({'x': [1, 2], 'y': [3, 4]}, xlsx, 'sheet1') is None
    # append a second sheet to the existing file
    assert data2xlsx(pd.DataFrame({'z': [5]}), xlsx, 'sheet2') is None

    with pd.ExcelFile(xlsx) as xf:
        assert set(xf.sheet_names) == {'sheet1', 'sheet2'}
        df1 = pd.read_excel(xf, 'sheet1')
    assert df1['x'].tolist() == [1, 2]


# -- device enumeration --------------------------------------------------------

def test_list_devices_rejects_a_bad_kind():
    from kudio import list_devices

    with pytest.raises(ValueError, match="kind must be"):
        list_devices(kind='microphone')


def test_list_devices_shape_and_filtering():
    """Skips when there is no audio backend, e.g. on a headless CI runner."""
    from kudio import list_devices

    sd = pytest.importorskip("sounddevice")
    try:
        every = list_devices()
    except Exception as e:                      # no PortAudio host on this box
        pytest.skip(f"no audio backend: {e}")

    assert isinstance(every, list)
    for d in every:
        assert set(d) == {'index', 'name', 'max_input_channels',
                          'max_output_channels', 'default_samplerate', 'api'}
    apis = [api['name'] for api in sd.query_hostapis()]
    assert [d['api'] for d in every] == [apis[d['hostapi']] for d in sd.query_devices()]

    inputs = list_devices('input')
    assert all(d['max_input_channels'] > 0 for d in inputs)
    assert len(inputs) <= len(every)
    # indices index into sounddevice's own table -- the contract record() relies on
    assert [d['index'] for d in every] == list(range(len(sd.query_devices())))


# -- optional dependencies -----------------------------------------------------

def test_visual_names_the_viz_extra_without_matplotlib():
    """Importing the plotting helpers has to point at the extra that fixes it.

    Runs in a subprocess with matplotlib hidden from the import machinery, so
    the check means the same thing on a box that has the [viz] extra installed
    and on one that does not.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Hide:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'matplotlib' or name.startswith('matplotlib.'):\n"
        "            raise ModuleNotFoundError(name, name=name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Hide())\n"
        "from kudio.exceptions import DependencyError\n"
        "try:\n"
        "    import kudio.util.visual\n"
        "except DependencyError as e:\n"
        "    assert 'matplotlib' in str(e), str(e)\n"
        "    assert 'kudio[viz]' in str(e), str(e)\n"
        "else:\n"
        "    raise AssertionError('kudio.util.visual imported without matplotlib')\n"
    )
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# -- column widths on a new workbook -------------------------------------------
def test_data2xlsx_fits_the_columns_of_a_new_file(tmp_path):
    pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")
    xlsx = tmp_path / "new.xlsx"
    data2xlsx({'a_long_column_name': [1], 'b': ['fifteen letters']}, xlsx, 'S')
    sheet = openpyxl.load_workbook(xlsx)['S']
    assert sheet.column_dimensions['A'].width == len('a_long_column_name')
    assert sheet.column_dimensions['B'].width == len('fifteen letters')


# -- where the plotting helpers save -------------------------------------------

def test_save_dir_is_a_folder_and_each_plot_keeps_its_own_file(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np

    import kudio
    from kudio.util import visual

    clip = tmp_path / "clip.wav"
    t = np.arange(8000) / 8000
    kudio.save_wave(clip, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 8000)
    figs = tmp_path / "figs"
    visual.plot_PSD(clip, save_dir=figs, dpi=50)
    visual.plot_3_features(clip, save_dir=figs, dpi=50)
    visual.plot_4_features(clip, save_dir=figs, dpi=50)
    visual.basic_analyze(clip, save_dir=figs)
    history = types.SimpleNamespace(history={"accuracy": [0.5, 0.7], "val_accuracy": [0.4, 0.6],
                                             "loss": [1.0, 0.6], "val_loss": [1.1, 0.8]})
    visual.hist_plt(history, fig_dir=figs)
    assert sorted(p.name for p in figs.iterdir()) == [
        "clip_3_features.png", "clip_4_features.png", "clip_overview.png", "clip_psd.png",
        "model_history.png"]
    # a name with a suffix matplotlib writes is still the file itself
    visual.plot_PSD(clip, save_dir=tmp_path / "out" / "spectrum.pdf", dpi=50)
    assert (tmp_path / "out" / "spectrum.pdf").is_file()


def test_common_saves_a_bare_name_in_its_fig_dir(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from kudio.util import visual

    monkeypatch.chdir(tmp_path)
    figure = visual.Common(fig_dir="figs", is_save=True)
    figure.loss_plot([1.0, 0.5], [1.2, 0.7])
    figure.save("loss")
    figure.save(tmp_path / "elsewhere.png")
    figure.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["elsewhere.png", "figs"]
    assert [p.name for p in (tmp_path / "figs").iterdir()] == ["loss.png"]
