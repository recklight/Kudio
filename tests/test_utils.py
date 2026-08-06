# -*- coding: utf-8 -*-
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
                          'max_output_channels', 'default_samplerate'}

    inputs = list_devices('input')
    assert all(d['max_input_channels'] > 0 for d in inputs)
    assert len(inputs) <= len(every)
    # indices index into sounddevice's own table -- the contract record() relies on
    assert [d['index'] for d in every] == list(range(len(sd.query_devices())))
