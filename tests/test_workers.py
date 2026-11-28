# -*- coding: utf-8 -*-
"""A process that dies is an error that says so. With multiprocessing.Pool whoever waited
for its answer waited for ever, so what is asked here is asked of another process, which
has a deadline."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import kudio
from kudio._workers import Workers, chunk_of

# Python 3.12 warns of every fork in a process that has threads, and numpy's BLAS has
# them; the pools here start as Python starts them by default, as 4.3.0's Pool did
pytestmark = pytest.mark.filterwarnings(
    r"ignore:This process \(pid=\d+\) is multi-threaded, use of fork\(\) may lead to "
    r"deadlocks in the child:DeprecationWarning")


def _path(*folders) -> str:
    """PYTHONPATH with *folders* in front of what it was: an environment that finds its
    packages through it keeps them."""
    there = os.environ.get("PYTHONPATH")
    return os.pathsep.join([str(f) for f in folders] + ([there] if there else []))


#: in the workers, which import this before anything else, a file of this name is the end
_DIES = '''
import multiprocessing, os
import soundfile
real = soundfile.SoundFile.__init__


def init(self, file, *args, **kwargs):
    if str(file).endswith("die.wav") and multiprocessing.parent_process() is not None:
        os._exit(9)
    real(self, file, *args, **kwargs)


soundfile.SoundFile.__init__ = init
'''

_MAIN = '''
import os, sys, time
import numpy as np
import kudio


def die(x):
    if x[0] == 3:
        os._exit(9)
    return float(x[0])


def put(folder, names, n=1600):
    os.makedirs(folder, exist_ok=True)
    for k, name in enumerate(names):
        kudio.save_wave(os.path.join(folder, name + ".wav"),
                        0.1 * np.random.default_rng(k).standard_normal(n), 16000)


def map_waves():
    kudio.map_waves(die, [np.full(2 ** 17, float(k)) for k in range(40)], workers=4)


def load_waves():
    put("many", ["die"] + [f"w{k:03d}" for k in range(300)], 160)
    kudio.load_waves("many", use_pool=True, core_=3)


def synthesizer():
    put("clean", ["a"]); put("noise", ["die"])
    kudio.Synthesizer("clean", "noise", "out", snr_ratio=[0.5 * k for k in range(210)]).syn(
        mode="inc", seed=0, is_pool=True)


def score_dataset():
    names = ["a", "b", "die", "c", "d"]
    put("ref", names); put("deg", names)
    kudio.score_dataset("deg", reference="ref", metrics=["snr"], jobs=2, n_resamples=1000)


def workers_40():
    print(kudio.map_waves(abs, range(-200, 0), workers=40) == list(range(200, 0, -1)))


def workers_70():
    print(kudio.map_waves(abs, range(-200, 0), workers=70) == list(range(200, 0, -1)))


if __name__ == "__main__":
    # one after the other, each on a line of its own, in one process: a Python and a
    # kudio started once instead of once a case
    for name in sys.argv[1:]:
        print(f"<{name}>", end=" ", flush=True)
        try:
            globals()[name]()
            print("no error", flush=True)
        except kudio.KudioError as e:
            print(type(e).__name__, " ".join(str(e).split()), flush=True)
        except BaseException as e:
            print("another error:", type(e).__name__, e, flush=True)
'''

CASES = {'map_waves': ('KudioError', 'map_waves(): a worker process died (exit code 9) over '
                                     'item 4 of 40.'),
         'load_waves': ('AudioIOError', 'load_waves(): a worker process died (exit code 9) over '
                                        'item 1 of 301.'),
         'synthesizer': ('SynthesisError', 'Synthesizer: a worker process died (exit code 9) '
                                           'over items '),
         'score_dataset': ('KudioError', 'score_dataset(): a worker process died (exit code 9) '
                                         'over item')}


#: what the script runs, in its order: the pools that one wait cannot hold first, the
#: pools with a worker that dies after
IN_A_SCRIPT = ('workers_40', 'workers_70', *CASES)


@pytest.fixture(scope="module")
def script(tmp_path_factory):
    """``{case: what it said}`` of the cases run by _MAIN, in a process of its own with
    100 s to do them in: what still waits then is missing, and so is what came after.
    The progress bars and the warnings go to another pipe: tqdm before 4.66 does not
    read TQDM_DISABLE, and its bars broke each answer off its name."""
    folder = tmp_path_factory.mktemp("script")
    (folder / "site").mkdir()
    (folder / "site" / "sitecustomize.py").write_text(_DIES)
    (folder / "main.py").write_text(_MAIN)
    env = dict(os.environ, TQDM_DISABLE="1",
               PYTHONPATH=_path(folder / "site", Path(kudio.__file__).parents[1]))
    child = subprocess.Popen([sys.executable, "main.py", *IN_A_SCRIPT], cwd=str(folder), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = child.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        out, err = _kill(child)
    said = {}
    for line in out.splitlines():
        if line.startswith("<") and "> " in line:
            name, rest = line[1:].split("> ", 1)
            said[name] = rest
    return said, f"{out}\n{err}"


@pytest.mark.parametrize("name", CASES)
def test_a_worker_that_dies_is_an_error_and_not_a_wait(script, name):
    said, out = script
    assert name in said, f"{name} still waits after 100 s, or never began:\n{out}"
    error, message = CASES[name]
    assert said[name].startswith(f"{error} {message}"), out


def _kill(child):
    """It and the workers it waits for; what it said until then, and on stderr."""
    if os.name == 'nt':
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True)
    else:
        subprocess.run(["pkill", "-9", "-P", str(child.pid)], capture_output=True)
    child.kill()
    return child.communicate()


def test_a_script_without_its_main_is_told_so(tmp_path):
    """It is run again by each worker, which may not start any, and dies of it."""
    (tmp_path / "main.py").write_text(textwrap.dedent("""
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
        import kudio
        kudio.map_waves(len, ["a", "b", "c"], workers=2)
    """))
    env = dict(os.environ, TQDM_DISABLE="1", PYTHONPATH=_path(Path(kudio.__file__).parents[1]))
    child = subprocess.Popen([sys.executable, "main.py"], cwd=str(tmp_path), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = child.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        _kill(child)
        pytest.fail("it still waits")
    assert child.returncode != 0
    assert "a worker process died (exit code 1) as it started." in out
    assert "if __name__ == '__main__':" in out


# -- the workers themselves ---------------------------------------------------------------

def _double(x):
    return 2 * x


def _slow_first(x):
    time.sleep(0.2 if x == 0 else 0)
    return x


def _fives(x):
    if x == 5:
        raise kudio.FeatureError("five is right out")
    return x


def _unsendable(x):
    return lambda: x


class _Odd(Exception):
    """One that does not come back from being pickled."""

    def __init__(self, a, b):
        super().__init__(a, b)

    def __reduce__(self):
        return _Odd, ("one argument short",)


def _raises_odd(x):
    raise _Odd(1, 2)


def _end(x):
    raise SystemExit(4)


@pytest.fixture(scope="module")
def team():
    with Workers(3, "here") as workers:
        yield workers


@pytest.mark.parametrize("chunk", [1, 2, 7, 100])
def test_the_answers_come_in_the_order_of_the_items(team, chunk):
    assert list(team.imap(_double, range(25), chunk)) == [2 * k for k in range(25)]
    assert list(team.imap(_slow_first, range(9), chunk)) == list(range(9))
    assert list(team.imap(_double, [], chunk)) == []


def test_arrays_larger_than_a_pipe_holds_go_there_and_back(team):
    items = [np.full(300_000, float(k)) for k in range(7)]
    for k, out in enumerate(team.imap(_double, items)):
        assert out.shape == (300_000,) and out[0] == out[-1] == 2.0 * k


def test_what_a_worker_raises_is_raised_here_in_its_turn(team):
    seen = []
    with pytest.raises(kudio.FeatureError, match="five is right out") as caught:
        for x in team.imap(_fives, range(12)):
            seen.append(x)
    assert seen == [0, 1, 2, 3, 4]
    assert "in _fives" in str(caught.value.__cause__)


def test_what_will_not_pickle_is_said(team):
    with pytest.raises(Exception, match="_unsendable.*lambda"):
        list(team.imap(_unsendable, range(4)))
    with pytest.raises(RuntimeError, match="_Odd"):
        list(team.imap(_raises_odd, range(4)))


def test_a_function_that_will_not_pickle_leaves_the_team_at_work():
    with Workers(2, "here") as workers:
        with pytest.raises(Exception, match="[Pp]ickle"):
            list(workers.imap(lambda x: x, range(4)))


def test_a_worker_that_leaves_is_one_that_died():
    """SystemExit is no error of the function's. It ended the pool's worker too."""
    with pytest.raises(kudio.SynthesisError,
                       match=r"^there: a worker process died \(exit code 4\) over item 1 of 1\."):
        with Workers(2, "there", kudio.SynthesisError) as workers:
            list(workers.imap(_end, [0]))


def test_one_killed_before_it_is_asked_is_found_when_it_is():
    """It never had item 1, and was said to have died over it."""
    with pytest.raises(kudio.KudioError,
                       match=r"a worker process died \(exit code -?\d+\) before item 1 of 5\."):
        with Workers(2, "here") as workers:
            workers.answer()
            for worker in workers._team:
                worker.process.kill()
                worker.process.join()
            list(workers.imap(_double, range(5)))


def test_nothing_is_left_running():
    import multiprocessing
    before = set(multiprocessing.active_children())
    with Workers(2, "here") as workers:
        workers.answer()
        assert len(set(multiprocessing.active_children()) - before) == 2
    assert set(multiprocessing.active_children()) <= before


@pytest.mark.parametrize("n, workers, chunk", [(1, 8, 1), (31, 8, 1), (32, 8, 1), (33, 8, 2),
                                               (210, 8, 7), (10000, 8, 313)])
def test_every_worker_gets_four_goes(n, workers, chunk):
    assert chunk_of(n, workers) == chunk


def _naps(x):
    if x == 0:
        raise kudio.FeatureError("at once")
    time.sleep(60)
    return x


def test_an_error_does_not_wait_for_the_rest_to_finish():
    """The three that were still at work were each waited for 10 s."""
    started = time.monotonic()
    with pytest.raises(kudio.FeatureError, match="at once"):
        with Workers(4, "here") as workers:
            workers.answer()                       # so that all four are at work at once
            list(workers.imap(_naps, range(4)))
    assert time.monotonic() - started < 15


def _one_wait_holds():
    """How many processes a wait can hold the handles of."""
    if os.name == 'nt':
        import _winapi
        if not hasattr(_winapi, 'BatchedWaitForMultipleObjects'):
            return 31
    return None


@pytest.mark.parametrize("n", [40, 70])
def test_more_workers_than_one_wait_holds(script, n):
    """On Windows before Python 3.13 32 of them were 64 handles, and a ValueError of
    _winapi's. map_waves runs in the script, where its workers do not load pytest."""
    said, out = script
    assert said.get(f"workers_{n}") == "True", out
    assert Workers(n, "here").n == min(n, _one_wait_holds() or n)


def test_the_threads_are_shared_out_among_the_workers_there_are(monkeypatch, tmp_path):
    from kudio import _workers
    from kudio.core import dataset_score
    for side in ("ref", "deg"):
        (tmp_path / side).mkdir()
        for k in range(5):
            kudio.save_wave(tmp_path / side / f"{k}.wav", np.full(1600, 0.1 * (k + 1)), 16000)
    asked = []

    class Stop(Exception):
        pass

    def run(tasks, workers):
        asked.append(workers)
        raise Stop

    monkeypatch.setattr(_workers, "most", lambda n: min(n, 2))
    monkeypatch.setattr(dataset_score, "_run", run)
    with pytest.raises(Stop):
        kudio.score_dataset(tmp_path / "deg", reference=tmp_path / "ref", metrics=["snr"],
                            jobs=5, n_resamples=1000)
    assert asked == [2]
