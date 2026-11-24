#!/usr/bin/env python3
"""A terminal agent for Linux powered by a Mistral model.

Uses the Mistral chat-completions API with tool calling so the model can
run shell commands, read files, write files, and list directories on your
machine until your task is done.

Requirements:
    - Python 3.9+
    - MISTRAL_API_KEY environment variable (get one at https://console.mistral.ai)

Usage:
    export MISTRAL_API_KEY="..."
    python3 agent.py "list all python files in this repo and count their lines"
    python3 agent.py            # interactive REPL
    python3 agent.py --model mistral-large-latest "fix the failing test"
    python3 agent.py --yes --max-steps 30 "..."
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-small-latest"

SYSTEM_PROMPT = """You are a Linux terminal agent. You help the user complete tasks
on their machine by using the provided tools.

Rules:
- Work step by step. Call one tool at a time and inspect results before continuing.
- Prefer read-only commands first to understand the situation before changing anything.
- Be careful with destructive commands (rm, mv over existing files, package installs).
- When the task is complete, reply concisely to the user in plain text with a short
  summary of what was done. Do not call tools anymore.
- If the task is impossible or needs information from the user, explain clearly."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command on the host (bash) and return stdout, stderr and exit code. Use this for executing things, not for reading/writing single files where dedicated tools exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 60, max 600)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its content. Fails if the file does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                    "content": {"type": "string", "description": "Full content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a path (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to current directory."},
                },
            },
        },
    },
]

DESTRUCTIVE_HINTS = ("rm ", "rm -", "mkfs", "dd ", "shutdown", "reboot", "> /dev/", "git push --force")


def tool_run_command(args: dict) -> str:
    command = args.get("command", "")
    timeout = min(int(args.get("timeout", 60)), 600)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        result = f"exit code: {proc.returncode}"
        if out:
            result += f"\nstdout:\n{out[:20000]}"
        if err:
            result += f"\nstderr:\n{err[:20000]}"
        return result
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def tool_read_file(args: dict) -> str:
    path = os.path.expanduser(args.get("path", ""))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(200000)
        return content if content else "(empty file)"
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def tool_write_file(args: dict) -> str:
    path = os.path.expanduser(args.get("path", ""))
    content = args.get("content", "")
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"ok: wrote {len(content)} bytes to {path}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def tool_list_dir(args: dict) -> str:
    path = os.path.expanduser(args.get("path") or ".")
    try:
        entries = sorted(os.listdir(path))
        visible = [e for e in entries if not e.startswith(".")]
        return "\n".join(visible) if visible else "(empty directory)"
    except FileNotFoundError:
        return f"error: directory not found: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


TOOL_IMPL = {
    "run_command": tool_run_command,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
}


def confirm_run(command: str, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    risky = any(h in command for h in DESTRUCTIVE_HINTS)
    if not risky:
        return True
    try:
        answer = input(f"Approve potentially destructive command '{command}'? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def call_mistral(api_key: str, model: str, messages: list) -> dict:
    payload = json.dumps(
        {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.2}
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Mistral API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Mistral API: {exc.reason}") from exc


def chat_once(api_key: str, model: str, messages: list, auto_yes: bool) -> list:
    """One request/execution round. Returns message objects to append."""
    data = call_mistral(api_key, model, messages)
    choice = data["choices"][0]
    assistant_msg = choice["message"]
    tool_calls = assistant_msg.get("tool_calls") or []

    new_messages = [{"role": "assistant", "content": assistant_msg.get("content") or ""}]
    if tool_calls:
        new_messages[0]["tool_calls"] = tool_calls

    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}

        print(f"  [tool] {name} {json.dumps(args, ensure_ascii=False)[:200]}")

        if name == "run_command" and not confirm_run(args.get("command", ""), auto_yes):
            result = "error: user denied this command"
        else:
            impl = TOOL_IMPL.get(name)
            result = impl(args) if impl else f"error: unknown tool '{name}'"

        new_messages.append(
            {"role": "tool", "name": name, "content": result, "tool_call_id": tc["id"]}
        )

    return new_messages


def run_task(api_key: str, model: str, task: str, auto_yes: bool, max_steps: int) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for step in range(max_steps):
        new_messages = chat_once(api_key, model, messages, auto_yes)
        messages.extend(new_messages)
        had_tool_calls = any("tool_calls" in m for m in new_messages)
        if not had_tool_calls:
            text = new_messages[0].get("content", "").strip()
            if text:
                print(f"\n{text}\n")
            return
    print(f"\n[agent] stopped: reached max steps ({max_steps})")


def repl(api_key: str, model: str, auto_yes: bool, max_steps: int) -> None:
    print("Mistral terminal agent. Type a task, or 'exit' to quit.")
    while True:
        try:
            task = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task or task.lower() in ("exit", "quit"):
            break
        run_task(api_key, model, task, auto_yes, max_steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal agent powered by Mistral.")
    parser.add_argument("task", nargs="*", help="Task to run in one shot. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Mistral model (default {DEFAULT_MODEL}).")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve destructive commands.")
    parser.add_argument("--max-steps", type=int, default=20, help="Max tool-calling steps per task.")
    parser.add_argument(
        "--check", action="store_true", help="Run a tiny connectivity/self test against the API and exit."
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("error: set the MISTRAL_API_KEY environment variable (https://console.mistral.ai).", file=sys.stderr)
        return 1

    if not shutil.which("bash"):
        print("error: bash is required.", file=sys.stderr)
        return 1

    try:
        if args.check:
            data = call_mistral(api_key, args.model, [{"role": "user", "content": "say ok"}])
            print("API OK ->", data["choices"][0]["message"]["content"].strip())
            return 0
        if args.task:
            run_task(api_key, args.model, " ".join(args.task), args.yes, args.max_steps)
        else:
            repl(api_key, args.model, args.yes, args.max_steps)
    except KeyboardInterrupt:
        print("\n[agent] interrupted.")
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
