# -*- coding: utf-8 -*-
import yaml

from kudio import KC, KudioConfig


def test_default_config():
    cfg = KudioConfig.load_config()
    assert cfg.waveform.default_rate == 16000
    assert cfg.f_mfe.n_mels == 40      # audio-only defaults in v3
    assert 'train' not in cfg          # training keys moved out of kudio
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
