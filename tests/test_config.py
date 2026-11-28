# -*- coding: utf-8 -*-
import pytest
import yaml

from kudio import KC, KudioConfig


def test_default_config():
    cfg = KudioConfig.load_config()
    assert cfg.waveform.default_rate == 16000
    assert cfg.f_mfe.n_mels == 40      # audio-only defaults
    assert 'train' not in cfg          # no training keys
    assert KC is KudioConfig


def test_attribute_access_and_update():
    cfg = KudioConfig({'a': {'b': 1}})
    assert cfg.a.b == 1
    cfg.update({'a': {'c': 2}})
    assert cfg.a.b == 1 and cfg.a.c == 2
    assert cfg.to_dict() == {'a': {'b': 1, 'c': 2}}


def test_write_and_load_roundtrip(tmp_path):
    cfg = KudioConfig({'train': {'clean': 'data/clean'}, 'snr': [-5, 0, 5]})
    f = tmp_path / "cfg.yaml"
    assert cfg.write_config(str(f)) is True
    assert cfg.write_config(str(f)) is False          # no overwrite by default
    assert cfg.write_config(str(f), exist_ok=True) is True

    loaded = KudioConfig.load_config(str(f))
    assert loaded.train.clean == 'data/clean'
    assert loaded.snr == [-5, 0, 5]


def test_load_config_from_dir(tmp_path):
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({'x': 1}), encoding='utf-8')
    cfg = KudioConfig.load_config(str(tmp_path))
    assert cfg.x == 1


def test_freeze():
    cfg = KudioConfig({'a': 1})
    cfg.freeze()
    try:
        cfg['new_key'] = 2
        raised = False
    except KeyError:
        raised = True
    assert raised
    cfg.unfreeze()
    cfg['new_key'] = 2
    assert cfg.new_key == 2


# -- a path that leads nowhere is said, not swallowed -------------------------
def test_a_config_path_that_is_not_there_warns(tmp_path):
    with pytest.warns(UserWarning, match="no config at"):
        cfg = KudioConfig.load_config(str(tmp_path / "cfg.yml"))
    assert cfg.waveform.default_rate == 16000
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "cfg.yaml").write_text("x: 1\n", encoding="utf-8")
    with pytest.warns(UserWarning, match=r"2 cfg\.yaml"):
        KudioConfig.load_config(str(tmp_path))


# -- a write that is refused leaves the file that was there ---------------------
def test_write_config_takes_numpy_numbers_and_paths(tmp_path):
    from pathlib import PureWindowsPath

    import numpy as np
    cfg = KudioConfig({'rate': np.int64(16000), 'gain': np.float64(0.5),
                       'peak': np.float32(0.25), 'trim': np.bool_(True),
                       'name': np.str_('run'), 'snr': [np.int8(-5), np.array(0), 5],
                       'clean': PureWindowsPath('data', 'clean'),
                       'nested': {'sr': np.uint16(8000)}})
    plain = KudioConfig({'rate': 16000, 'gain': 0.5, 'peak': 0.25, 'trim': True,
                         'name': 'run', 'snr': [-5, 0, 5], 'clean': 'data/clean',
                         'nested': {'sr': 8000}})
    assert cfg.write_config(str(tmp_path / "n.yaml"))
    assert plain.write_config(str(tmp_path / "p.yaml"))
    assert (tmp_path / "n.yaml").read_bytes() == (tmp_path / "p.yaml").read_bytes()
    assert KudioConfig.load_config(str(tmp_path / "n.yaml")) == plain


def test_one_number_used_twice_is_written_twice(tmp_path):
    """Not as an anchor and a reference to it, which is how yaml writes one
    object met twice."""
    import numpy as np
    rate = np.int64(16000)
    KudioConfig(a=rate, b=rate).write_config(str(tmp_path / "c.yaml"))
    assert (tmp_path / "c.yaml").read_text(encoding="utf-8") == "a: 16000\nb: 16000\n"


@pytest.mark.parametrize("value, said", [
    (object, r"^write_config\(\): the config holds <class 'object'>, and YAML has no plain "
             r"form for a type$"),
    (1 + 2j, r"the config holds \(1\+2j\), .* a complex$"),
])
def test_a_config_that_cannot_be_written_leaves_the_one_there(tmp_path, value, said):
    path = tmp_path / "cfg.yaml"
    cfg = KudioConfig({'a': 1})
    cfg.write_config(str(path))
    before = path.read_bytes()
    cfg['b'] = value
    with pytest.raises(ValueError, match=said):
        cfg.write_config(str(path), exist_ok=True)
    assert path.read_bytes() == before


def test_what_the_safe_dumper_writes_is_written_the_same(tmp_path):
    """Shared lists, text outside ASCII, None, dates and nesting included."""
    import datetime
    shared = [1, 2]
    cfg = KudioConfig({'z': None, 'a': {'b': [1.5, True, '文字', {'c': ()}]},
                       'when': datetime.date(2026, 1, 2), 'raw': b'ab', 'on': 'yes'})
    cfg['twice'] = [shared, shared]
    cfg.write_config(str(tmp_path / "c.yaml"))
    assert ((tmp_path / "c.yaml").read_text(encoding="utf-8")
            == yaml.safe_dump(cfg.to_dict(), allow_unicode=True))


# -- dicts, however deep in lists they sit --------------------------------------------
def test_a_dict_inside_lists_is_written(tmp_path):
    data = {'stages': [[{'snr': [0, 5]}, {'snr': [10]}], ({'a': [{'b': 1}]}, 2)],
            'plain': [[1, 2], [3]]}
    cfg = KudioConfig(data)
    assert cfg.to_dict() == data

    def no_config_inside(value):
        if isinstance(value, dict):
            assert type(value) is dict
            value = list(value.values())
        if isinstance(value, (list, tuple)):
            for item in value:
                no_config_inside(item)
    no_config_inside(cfg.to_dict())
    path = tmp_path / "c.yaml"
    cfg.write_config(str(path))
    assert yaml.safe_load(path.read_text(encoding="utf-8"))['stages'][0] == data['stages'][0]
    assert KudioConfig.load_config(str(path)).stages[0][1].snr == [10]


def test_a_config_given_as_a_value_is_written_as_a_dict(tmp_path):
    cfg = KudioConfig()
    cfg['runs'] = {'first': KudioConfig(a=1)}             # set as it is, not converted
    cfg['list'] = [{'second': KudioConfig(b=[KudioConfig(c=2)])}]
    assert cfg.to_dict() == {'runs': {'first': {'a': 1}}, 'list': [{'second': {'b': [{'c': 2}]}}]}
    cfg.write_config(str(tmp_path / "c.yaml"))
    assert yaml.safe_load((tmp_path / "c.yaml").read_text(encoding="utf-8")) == cfg.to_dict()
