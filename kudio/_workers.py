# -*- coding: utf-8 -*-
"""Work shared out among processes, and an error when one of them dies.

``multiprocessing.Pool`` starts another process in the place of one that was killed,
for want of memory say, and waits for the answer of the dead one for ever. An executor
notices, but up to Python 3.11 it can wait for ever too: for the process to take in what
it is being sent. Here every process has a pipe of its own and no lock is shared.
"""
from __future__ import annotations

import os
import pickle
import traceback
from multiprocessing import connection, get_context
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type

from kudio.exceptions import KudioError

__all__ = ['Workers', 'chunk_of', 'most']

#: what a worker says first
_HELLO = "hello"


def most(n: int) -> int:
    """How many processes there are when *n* are asked for. On Windows before Python
    3.13 ``connection.wait`` takes 63 handles at most, and each process is two of them,
    its pipe and itself: 31 there, as many as asked anywhere else."""
    n = int(n)
    if os.name == 'nt':
        import _winapi
        if not hasattr(_winapi, 'BatchedWaitForMultipleObjects'):
            return min(n, 31)
    return n


class _Remote(Exception):
    """The traceback of what was raised in a worker, as the cause of it here."""

    def __str__(self) -> str:
        return f"\n\"\"\"\n{self.args[0]}\"\"\""


def _sendable(error: BaseException) -> BaseException:
    """*error*, or what can be said of it when it will not pickle."""
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:                                                # noqa: BLE001
        return RuntimeError(f"{type(error).__name__}: {error}")


def _say(mine: Any, answer: Any) -> bool:
    """Send it, or that it will not pickle; False when nobody is there to hear it."""
    try:
        try:
            mine.send(answer)
        except ConnectionError:
            raise
        except Exception as e:                                       # noqa: BLE001
            mine.send((False, (_sendable(e), traceback.format_exc())))
    except ConnectionError:
        return False
    return True


def _serve(mine: Any, theirs: Sequence[Any]) -> None:
    """A worker's whole life: say hello, and answer what comes down its pipe until there
    is no more."""
    for end in theirs:                       # a forked one holds the pipes of the rest
        end.close()
    answer: Any = _HELLO
    while _say(mine, answer):
        try:
            fn, items = mine.recv()
            answer = True, [fn(item) for item in items]
        except (EOFError, ConnectionError):
            return
        except Exception as e:                                       # noqa: BLE001
            answer = False, (_sendable(e), traceback.format_exc())


def chunk_of(n: int, workers: int) -> int:
    """How many of *n* items to send a worker at a time: each gets four goes."""
    return max(1, -(-n // (4 * workers)))


class _Worker:
    """A process, its pipe, and which part of the work it has, if any."""

    def __init__(self, process: Any, pipe: Any):
        self.process, self.pipe = process, pipe
        self.greeted, self.part = False, None


class Workers:
    """*n* processes, to be used in a ``with``; :attr:`n` is how many there are, which
    :func:`most` may make fewer.

    :param who: what starts the message of an error, ``map_waves()``.
    :param error: what is raised when a process dies.
    :param method: how the processes are started; Python's own way if not given.
    """

    def __init__(self, n: int, who: str, error: Type[Exception] = KudioError,
                 method: Optional[str] = None):
        self.n, self.who, self.error = most(n), who, error
        self._context = get_context(method)
        self._team: List[_Worker] = []

    def __enter__(self) -> "Workers":
        try:
            forked = self._context.get_start_method() == 'fork'
            for _ in range(self.n):
                ours, theirs = self._context.Pipe()
                ends = [worker.pipe for worker in self._team] + [ours] if forked else []
                process = self._context.Process(target=_serve, args=(theirs, ends), daemon=True)
                process.start()
                theirs.close()
                self._team.append(_Worker(process, ours))
        except BaseException:
            self._stop(now=True)
            raise
        return self

    def __exit__(self, kind, value, trace) -> None:
        self._stop(now=kind is not None)

    def _stop(self, now: bool) -> None:
        for worker in self._team:
            worker.pipe.close()                                # which is the end of it
        for worker in self._team:
            if now:
                worker.process.terminate()
            worker.process.join(10)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
        self._team = []

    def answer(self) -> None:
        """Wait until every process has said hello, or raise: a process that may not
        have children gets no further, nor does a script without its
        ``if __name__ == '__main__':``."""
        while not all(worker.greeted for worker in self._team):
            self._listen([worker for worker in self._team if not worker.greeted], 1, 0)

    def imap(self, fn: Callable[[Any], Any], items: Sequence[Any],
             chunk: int = 1) -> Iterator[Any]:
        """``fn(item)`` of each item, in the order of the items, as they are ready.
        What one of them raises is raised here, when its turn comes."""
        items = list(items)
        parts = [items[k:k + chunk] for k in range(0, len(items), chunk)]
        ready: Dict[int, Tuple[bool, Any]] = {}
        sent = 0
        for turn in range(len(parts)):
            while turn not in ready:
                for worker in self._team:
                    # only one that has said hello is sure to take in what it is sent
                    if worker.greeted and worker.part is None and sent < len(parts):
                        worker.part, sent = sent, sent + 1
                        try:
                            worker.pipe.send((fn, parts[worker.part]))
                        except ConnectionError:
                            # it was dead before it was sent anything
                            self._died(worker, chunk, len(items), given=False)
                        except BaseException:              # what will not pickle is not sent
                            worker.part = None
                            raise
                ready.update(self._listen(
                    [worker for worker in self._team
                     if worker.part is not None or not worker.greeted], chunk, len(items)))
            worked, answer = ready.pop(turn)
            if not worked:
                error, trace = answer
                raise error from _Remote(trace)
            yield from answer

    def _listen(self, workers: Sequence[_Worker], chunk: int,
                n: int) -> Dict[int, Tuple[bool, Any]]:
        """What those of *workers* have to say that have something to, by part."""
        # the process too: until it has read what it was started with, this one holds both
        # ends of its pipe on Windows, and the pipe of one that dies before has no end
        ends = {end: worker for worker in workers
                for end in (worker.pipe, worker.process.sentinel)}
        said: Dict[int, Tuple[bool, Any]] = {}
        for worker in {id(ends[end]): ends[end] for end in connection.wait(list(ends))}.values():
            try:
                answer = worker.pipe.recv() if worker.pipe.poll() else None
            except (EOFError, ConnectionError):
                answer = None
            if answer is None:
                self._died(worker, chunk, n)
            if worker.greeted:
                said[worker.part], worker.part = answer, None
            worker.greeted = True
        return said

    def _died(self, worker: _Worker, chunk: int, n: int, given: bool = True) -> None:
        worker.process.join(5)
        k, at = worker.part, "over" if given else "before"
        over = ("as it started" if k is None else f"{at} item {k + 1} of {n}" if chunk == 1 else
                f"{at} items {k * chunk + 1} to {min(n, (k + 1) * chunk)} of {n}")
        raise self.error(
            f"{self.who}: a worker process died (exit code {worker.process.exitcode}) {over}. One "
            f"that was killed for want of memory dies like that; so does every one of a script "
            f"whose top level is not behind if __name__ == '__main__':")
