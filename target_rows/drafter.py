"""Local and separate-process interfaces for the same draft-pool builder."""
import multiprocessing as mp
import time

import torch

from .decoding import draft_pool_fast, synchronize
from .torch_backend import FastLM


class LocalDrafter:
    """Run drafting synchronously, including when both models share a GPU."""

    def __init__(self, model, n_vocab):
        if model.embed.shape[0] < n_vocab or n_vocab < 10:
            raise ValueError("Drafter vocabulary does not cover the shared tokenizer.")
        self.model, self.n_vocab = model, n_vocab
        self.n_done = 0
        self.pending = None

    def reset(self, prompt):
        self.model.prefill(prompt)
        self.n_done = len(prompt)
        self.pending = None

    def submit(self, tokens):
        if self.pending is not None:
            raise RuntimeError("Read the previous draft before submitting another.")
        started = time.perf_counter()
        tok, score, par, self.n_done = draft_pool_fast(self.model, tokens, self.n_done, self.n_vocab)
        self.pending = tok, score, par, time.perf_counter() - started

    def result(self):
        if self.pending is None:
            raise RuntimeError("No draft is pending.")
        result, self.pending = self.pending, None
        return result

    def memory_gib(self):
        synchronize(self.model.dev)
        if self.model.dev.type == "cuda":
            return torch.cuda.max_memory_allocated(self.model.dev) / 2**30
        return 0.0

    def close(self):
        pass


def _worker(conn, model_kwargs, n_vocab):
    try:
        torch.set_num_threads(2)
        torch.cuda.set_device(model_kwargs["device"])
        drafter = LocalDrafter(FastLM(**model_kwargs), n_vocab)
        conn.send(("ready", None))
        while True:
            command, value = conn.recv()
            if command == "stop":
                break
            if command == "reset":
                drafter.reset(value)
            elif command == "draft":
                drafter.submit(value)
                conn.send(("pool", drafter.result()))
            elif command == "memory":
                conn.send(("memory", drafter.memory_gib()))
            else:
                raise ValueError("Unknown drafter command.")
    except Exception as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()


class ProcessDrafter:
    """Overlap drafting on a second GPU with the target's root pass."""

    def __init__(self, model_kwargs, n_vocab, timeout=900):
        ctx = mp.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.timeout = timeout
        self.process = ctx.Process(target=_worker, args=(child_conn, model_kwargs, n_vocab))
        self.process.start()
        child_conn.close()
        try:
            self._receive("ready", timeout=max(timeout, 1800))
        except BaseException:
            self.close()
            raise

    def _receive(self, kind, timeout=None):
        if not self.conn.poll(self.timeout if timeout is None else timeout):
            raise TimeoutError("The drafter did not respond; check its GPU memory and model access.")
        try:
            status, payload = self.conn.recv()
        except EOFError as exc:
            raise RuntimeError("The drafter process exited before returning a result.") from exc
        if status == "error":
            raise RuntimeError(f"Drafter failed: {payload}")
        if status != kind:
            raise RuntimeError(f"Expected {kind}, received {status}.")
        return payload

    def reset(self, prompt):
        self.conn.send(("reset", prompt))

    def submit(self, tokens):
        self.conn.send(("draft", tokens))

    def result(self):
        return self._receive("pool")

    def memory_gib(self):
        self.conn.send(("memory", None))
        return self._receive("memory")

    def close(self):
        if self.process.is_alive():
            try:
                self.conn.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.conn.close()
