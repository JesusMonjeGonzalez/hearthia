"""Narrow subprocess bridge to TreePact's independently installed CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hearthia.settings import TreePactSettings

REVIEW_SCHEMA_VERSION = 1
_REVIEW_TIMEOUT_SECONDS = 5.0
_REVIEW_MAX_OUTPUT_BYTES = 1024 * 1024
_RUN_ID_RE = re.compile(r"^run_[a-f0-9]{32}$")

# Minimal allowlisted environment for the review subprocess: no inherited
# secrets, credentials or agent sockets.
_REVIEW_ENV_ALLOWLIST = ("HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "PATH")


class TreePactBridgeError(RuntimeError):
    """TreePact is unavailable or incompatible with this Hearthia release."""


class TreePactReviewNotFoundError(TreePactBridgeError):
    """The requested TreePact run does not exist."""


@dataclass(frozen=True)
class TreePactBridge:
    executable: Path
    expected_version: str

    @classmethod
    def from_settings(cls, settings: TreePactSettings) -> TreePactBridge:
        candidate = settings.executable or shutil.which("treepact")
        if candidate is None:
            raise TreePactBridgeError(
                "treepact is not installed; install TreePact separately or configure "
                "[treepact].executable"
            )

        executable = Path(candidate).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise TreePactBridgeError(f"treepact executable is not runnable: {executable}")
        return cls(executable=executable, expected_version=settings.expected_version)

    def verify_version(self) -> None:
        try:
            result = subprocess.run(
                [str(self.executable), "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise TreePactBridgeError(f"cannot run treepact: {exc}") from exc

        output = result.stdout.strip()
        match = re.match(r"^treepact\s+([^\s|]+)", output)
        if result.returncode != 0 or match is None:
            raise TreePactBridgeError("treepact did not return a valid version")
        if match.group(1) != self.expected_version:
            raise TreePactBridgeError(
                f"unsupported treepact version {match.group(1)}; expected {self.expected_version}"
            )

    def invoke(self, args: list[str]) -> int:
        """Run a fixed TreePact command with inherited stdio and no shell."""
        self.verify_version()
        try:
            result = subprocess.run([str(self.executable), *args], check=False)
        except OSError as exc:
            raise TreePactBridgeError(f"cannot run treepact: {exc}") from exc
        return result.returncode

    def doctor(self, repo: Path | None = None, *, deep: bool = False) -> int:
        args = ["doctor"]
        if repo is not None:
            args.extend(["--repo", str(repo.expanduser().resolve())])
        if deep:
            args.append("--deep")
        return self.invoke(args)

    def validate(self, repo: Path) -> int:
        return self.invoke(["validate", "--repo", str(repo.expanduser().resolve())])

    def status(self, run_id: str) -> int:
        return self.invoke(["status", run_id])

    def diff(self, run_id: str, *, stat: bool = False, name_only: bool = False) -> int:
        if stat and name_only:
            raise TreePactBridgeError("stat and name_only cannot be used together")
        args = ["diff", run_id]
        if stat:
            args.append("--stat")
        if name_only:
            args.append("--name-only")
        return self.invoke(args)

    def evidence(
        self, run_id: str, *, output_format: str = "summary", verify_hashes: bool = False
    ) -> int:
        if output_format not in {"summary", "json", "markdown"}:
            raise TreePactBridgeError("evidence format must be summary, json or markdown")
        args = ["evidence", run_id, "--format", output_format]
        if verify_hashes:
            args.append("--verify-hashes")
        return self.invoke(args)

    def verify(
        self, run_id: str, *, check_artifacts: bool = True, check_events: bool = True
    ) -> int:
        args = ["verify", run_id]
        if check_artifacts:
            args.append("--check-artifacts")
        if check_events:
            args.append("--check-events")
        return self.invoke(args)

    def run(
        self,
        repo: Path,
        task: str,
        *,
        mode: str,
        model_profile: str | None = None,
        max_attempts: int | None = None,
        max_minutes: int | None = None,
    ) -> int:
        if mode not in {"observe", "repair"}:
            raise TreePactBridgeError("mode must be observe or repair")

        args = [
            "run",
            task,
            "--repo",
            str(repo.expanduser().resolve()),
            "--mode",
            mode,
            "--runtime",
            "native",
        ]
        if model_profile is not None:
            args.extend(["--model-profile", model_profile])
        if max_attempts is not None:
            if not 1 <= max_attempts <= 3:
                raise TreePactBridgeError("max_attempts must be between 1 and 3")
            args.extend(["--max-attempts", str(max_attempts)])
        if max_minutes is not None:
            if max_minutes < 1:
                raise TreePactBridgeError("max_minutes must be at least 1")
            args.extend(["--max-minutes", str(max_minutes)])
        return self.invoke(args)

    # ---- read-only review contract ---------------------------------------
    #
    # These methods never call self.invoke(): they capture stdout instead of
    # inheriting it, run with a minimized environment, enforce a timeout and
    # an output-size ceiling, and validate the parsed document's schema
    # before returning it. They never expose task text, repository paths,
    # worktree paths, artifact content, prompts, provider payloads, logs or
    # diffs — TreePact's `review` command already excludes them.

    def _capture_review(self, args: list[str]) -> dict[str, Any]:
        self.verify_version()
        env = {key: os.environ[key] for key in _REVIEW_ENV_ALLOWLIST if key in os.environ}
        try:
            result = subprocess.run(
                [str(self.executable), "review", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=_REVIEW_TIMEOUT_SECONDS,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise TreePactBridgeError("treepact review timed out") from exc
        except OSError as exc:
            raise TreePactBridgeError(f"cannot run treepact: {exc}") from exc

        if result.returncode != 0:
            match = re.search(r"\[([a-z0-9_]+)\]", result.stderr)
            code = match.group(1) if match else "unknown"
            if code == "run_not_found":
                raise TreePactReviewNotFoundError("TreePact run was not found")
            raise TreePactBridgeError(f"treepact review failed ({code})")

        stdout = result.stdout
        if len(stdout.encode("utf-8")) > _REVIEW_MAX_OUTPUT_BYTES:
            raise TreePactBridgeError("treepact review output exceeded the size limit")
        try:
            document = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise TreePactBridgeError("treepact review returned invalid JSON") from exc
        if not isinstance(document, dict):
            raise TreePactBridgeError("treepact review returned an unexpected document")
        if (
            document.get("schema") != "treepact.review"
            or document.get("schema_version") != REVIEW_SCHEMA_VERSION
        ):
            raise TreePactBridgeError("treepact review returned an incompatible contract version")
        return document

    def review_runs(self, limit: int = 20) -> dict[str, Any]:
        """List runs via the strict read-only `treepact review` contract."""
        if not 1 <= limit <= 100:
            raise TreePactBridgeError("limit must be between 1 and 100")
        document = self._capture_review(
            ["--schema-version", str(REVIEW_SCHEMA_VERSION), "--limit", str(limit)]
        )
        if document.get("kind") != "run_list":
            raise TreePactBridgeError("treepact review returned an unexpected document kind")
        return document

    def review_run(self, run_id: str) -> dict[str, Any]:
        """Fetch one run's read-only detail via `treepact review --run-id`."""
        if _RUN_ID_RE.match(run_id) is None:
            raise TreePactBridgeError("run_id must match run_<32 lowercase hex>")
        document = self._capture_review(
            ["--schema-version", str(REVIEW_SCHEMA_VERSION), "--run-id", run_id]
        )
        if document.get("kind") != "run_detail":
            raise TreePactBridgeError("treepact review returned an unexpected document kind")
        return document
