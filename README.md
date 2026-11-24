# mistral-agent

A lightweight terminal agent for Linux powered by a Mistral model.

The agent uses Mistral's **tool-calling** API to complete tasks on your machine by
running shell commands, reading/writing files, and listing directories — looping
until the task is done (a mini coding/ops assistant). Zero third-party Python
dependencies; only the Python standard library.

## Setup

1. Get an API key at [console.mistral.ai](https://console.mistral.ai) (free tier available).
2. Export it:

```bash
export MISTRAL_API_KEY="your-key-here"
```

3. Verify connectivity:

```bash
python3 agent.py --check
```

## Usage

One-shot task (agent works until done):

```bash
python3 agent.py "list all python files in this directory and count their total lines"
python3 agent.py --model mistral-large-latest "find and fix the failing test in this repo"
```

Interactive mode:

```bash
python3 agent.py
```

Options:

| Flag | Description |
|---|---|
| `--model` | Mistral model id (default: `mistral-small-latest`). Try `mistral-large-latest`, `open-mistral-nemo`, etc. |
| `--yes`, `-y` | Auto-approve destructive-looking commands instead of asking. |
| `--max-steps` | Max tool-calling steps per task (default 20). |
| `--check` | Connectivity test, then exit. |

## Tools available to the model

- `run_command(command, timeout)` — run a bash command, returns exit code/stdout/stderr
- `read_file(path)` — read a text file
- `write_file(path, content)` — create or overwrite a file
- `list_dir(path)` — list a directory

## Safety

- Commands that look destructive (`rm`, `mkfs`, `dd`, forced pushes…) prompt for
  confirmation unless `--yes` is passed.
- Everything happens locally; only task text and tool results are sent to the Mistral API.
- Do not point it at sensitive directories you don't want modified.
