import os
import sys
import json
import types
import pathlib
import subprocess
import pytest

# Import the module under test
from openhands.llm.providers.claude_code import (
    ClaudeCodeOptions,
    run_claude_code,
    ClaudeCodeError,
)


def fake_claude_process(monkeypatch, stdout_lines: list[str], returncode: int = 0):
    """Monkeypatch subprocess.Popen so run_claude_code reads our provided JSONL and exits.
    The fake process supports .stdin.write/.stdin.close and yields lines from stdout_lines.
    """
    class FakeProc:
        def __init__(self):
            self.returncode = returncode
            self.stdin = types.SimpleNamespace(write=lambda b: None, close=lambda: None)
            # Store as text lines; Popen was created with text=True
            self._lines = list(stdout_lines)
            self._idx = 0
            # Prepare stderr text when process fails
            self._stderr_text = "simulated error" if returncode != 0 else ""

        def stdout_readline(self):
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        class _Stdout:
            def __init__(self, outer):
                self._outer = outer
            def readline(self):
                return self._outer.stdout_readline()
            def __iter__(self):
                return self
            def __next__(self):
                line = self._outer.stdout_readline()
                if line == "":
                    raise StopIteration
                return line

        class _Stderr:
            def __init__(self, text: str):
                self._text = text
            def read(self):
                return self._text

        @property
        def stdout(self):
            return FakeProc._Stdout(self)

        @property
        def stderr(self):
            return FakeProc._Stderr(self._stderr_text)

        def wait(self):
            return self.returncode

    def _fake_popen(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)


def test_run_claude_code_happy_path(monkeypatch):
    # A minimal stream-json happy path with a final assistant message
    events = [
        json.dumps({"type": "start"}) + "\n",
        json.dumps({"type": "message.delta", "delta": "Hello"}) + "\n",
        json.dumps({"type": "message.delta", "delta": ", world"}) + "\n",
        json.dumps({"type": "message.complete", "message": {"role": "assistant", "content": "Hello, world"}}) + "\n",
    ]
    fake_claude_process(monkeypatch, events, returncode=0)

    opts = ClaudeCodeOptions(
        path="claude",
        system_prompt="Say hi",
        messages=[{"role": "user", "content": "Say hi"}],
        model_id=None,
        max_output_tokens=256,
    )

    out = list(run_claude_code(opts))
    assert any(ev.get("type") == "message.complete" for ev in out)


def test_run_claude_code_bad_json(monkeypatch):
    # Emits an invalid JSON line; should be ignored, not crash
    events = ["{not-json}\n", json.dumps({"type": "message.complete"}) + "\n"]
    fake_claude_process(monkeypatch, events, returncode=0)

    opts = ClaudeCodeOptions(
        path="claude",
        system_prompt="test",
        messages=[{"role": "user", "content": "test"}],
        model_id=None,
        max_output_tokens=None,
    )

    out = list(run_claude_code(opts))
    assert any(ev.get("type") == "message.complete" for ev in out)


def test_run_claude_code_nonzero_exit(monkeypatch):
    # No output, non-zero exit should raise ClaudeCodeError
    fake_claude_process(monkeypatch, [], returncode=1)

    opts = ClaudeCodeOptions(
        path="claude",
        system_prompt="oops",
        messages=[{"role": "user", "content": "oops"}],
        model_id=None,
        max_output_tokens=None,
    )

    with pytest.raises(ClaudeCodeError):
        list(run_claude_code(opts))
