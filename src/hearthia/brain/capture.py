"""Brain capture: quick note capture with AI-driven title/tag/folder filing.

Capture never fails: if the local model is down, the raw text lands in 00 Inbox.
"""

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

FOLDERS = ["00 Inbox", "03 Resources/Code Snippets", "03 Resources/Tools & Configs"]

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 80},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "folder": {"type": "string", "enum": FOLDERS},
        "language": {"type": "string"},
    },
    "required": ["title", "tags", "folder"],
}

PROMPT = """You file notes into a PARA second brain for a software engineer.
Given a captured note, return JSON with:
- title: short descriptive title (no date, no quotes, filesystem-safe)
- tags: 1-5 lowercase kebab-case tags
- folder: "03 Resources/Code Snippets" ONLY if it's primarily a code snippet/command,
  "03 Resources/Tools & Configs" ONLY if it's tool configuration/setup instructions,
  otherwise "00 Inbox"
- language: programming language if it contains code, else ""

Note:
---
{text}
---"""


def get_text(args: list[str] | None = None, stdin: str | None = None) -> str:
    """Get text from args, stdin, or $EDITOR."""
    if args is None:
        args = sys.argv[1:]
    if args:
        return " ".join(args)
    if stdin is not None and stdin.strip():
        return stdin
    if stdin is not None:
        return stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w+", delete=False) as f:
        path = f.name
    subprocess.call([editor, path])
    text = open(path).read()
    os.unlink(path)
    return text


async def classify(
    client: httpx.AsyncClient,
    text: str,
    gateway_url: str,
    model: str = "fast",
) -> dict | None:
    """Classify a note using the local model. Returns None if model is down."""
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 300,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": PROMPT.format(text=text[:4000])}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "note_meta", "schema": SCHEMA},
            },
        }
    ).encode()
    try:
        r = await client.post(
            f"{gateway_url}/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(120.0),
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return json.loads(data["choices"][0]["message"]["content"])
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, TimeoutError):
        return None


def safe_name(title: str) -> str:
    title = re.sub(r'[/\\:*?"<>|#^\[\]]', "", title).strip()
    return title[:80] or "Untitled"


def write_note(
    vault: Path,
    text: str,
    meta: dict | None = None,
) -> str:
    """Write a note to the vault. Returns the path written."""
    now = datetime.datetime.now()
    if meta:
        title = safe_name(meta.get("title", ""))
        tags = meta.get("tags", []) or ["capture"]
        folder = meta.get("folder") if meta.get("folder") in FOLDERS else "00 Inbox"
        lang = meta.get("language", "")
    else:
        title = f"Capture {now:%Y-%m-%d %H%M%S}"
        tags, folder, lang = ["capture", "unfiled"], "00 Inbox", ""

    dest_dir = vault / str(folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{title}.md"
    n = 1
    while path.exists():
        n += 1
        path = dest_dir / f"{title} {n}.md"

    front = ["---", f"tags: [{', '.join(tags)}]", f"captured: {now:%Y-%m-%d %H:%M}"]
    if lang:
        front.append(f"language: {lang}")
    front.append("---")
    path.write_text("\n".join(front) + "\n\n" + text + "\n")
    return str(path)
