import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass
class ClaudeCodeOptions:
    system_prompt: str
    messages: list[dict]
    path: str = "claude"
    model_id: Optional[str] = None
    max_output_tokens: Optional[int] = None


class ClaudeCodeError(Exception):
    pass


def run_claude_code(options: ClaudeCodeOptions) -> Iterator[dict]:
    """Run the Claude Code CLI and yield parsed JSON events per line.

    This mirrors Roo Code's integration: we pass messages via stdin and
    ask for stream-json output, then parse each line as a JSON object.
    """
    is_windows = sys.platform.startswith("win32")

    args = ["-p", "--verbose", "--output-format", "stream-json", "--disallowedTools",
            ",".join([
                "Task","Bash","Glob","Grep","LS","exit_plan_mode","Read","Edit",
                "MultiEdit","Write","NotebookRead","NotebookEdit","WebFetch","TodoRead",
                "TodoWrite","WebSearch"
            ]), "--max-turns", "1"]

    if options.model_id:
        args.extend(["--model", options.model_id])

    env = os.environ.copy()
    mot = options.max_output_tokens
    if mot is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(mot)

    # Windows: include system prompt in stdin to avoid cmd length limits
    # Non-Windows: pass as flag like Roo does
    if not is_windows:
        args.extend(["--system-prompt", options.system_prompt])

    try:
        proc = subprocess.Popen(
            [options.path] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise ClaudeCodeError(
            f"Claude Code executable '{options.path}' not found. Original error: {e}"
        )

    # Prepare stdin payload
    if is_windows:
        stdin_payload = json.dumps({
            "systemPrompt": options.system_prompt,
            "messages": options.messages,
        })
    else:
        stdin_payload = json.dumps(options.messages)

    assert proc.stdin and proc.stdout
    try:
        proc.stdin.write(stdin_payload)
        proc.stdin.close()
    except Exception as e:
        proc.kill()
        raise ClaudeCodeError(f"Failed writing to Claude Code stdin: {e}")

    # Stream and parse line-delimited JSON
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # keep partial lines until process ends; skip noisy lines
            continue

    # Wait for process exit and raise on error
    exit_code = proc.wait()
    if exit_code != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise ClaudeCodeError(
            f"Claude Code process exited with code {exit_code}. Error output: {stderr.strip()}"
        )


def to_litellm_like_response(stream_events: list[dict]) -> dict:
    """Convert Claude Code stream events to a LiteLLM-like response dict.

    We search for the last assistant message and return it in the familiar
    { choices: [ { message: { role, content } } ], usage: {...} } shape.
    """
    assistant_text: str = ""
    # Prefer assistant events
    for ev in stream_events:
        if ev.get("type") == "assistant":
            msg = ev.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        t = b.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                assistant_text = "".join(parts)
            elif isinstance(content, str):
                assistant_text = content
    # Fallback for stream-json "message.complete" final events
    if not assistant_text:
        for ev in reversed(stream_events):
            if ev.get("type") == "message.complete":
                msg = ev.get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            t = b.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    assistant_text = "".join(parts)
                elif isinstance(content, str):
                    assistant_text = content
                break

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": assistant_text,
                }
            }
        ],
        # Usage is not exposed by CLI in a stable way; leave unset
    }
