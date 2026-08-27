#!/usr/bin/env python3
"""Measure what Hearthia's memory planning buys you on a live stack.

Walks every running llama-server, compares measured RSS against the GGUF
header estimate, and prints a markdown summary: per-model residency, the
co-resident total against the GPU-wired ceiling, and the KV-cache cost per
1K context tokens — the number that decides whether a second model fits.

Usage:  uv run scripts/benchmark.py [--json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psutil  # noqa: E402

from hearthia.budget import estimate_model_ram, profile_for  # noqa: E402
from hearthia.registry import Model  # noqa: E402
from hearthia.telemetry import llama_server_procs, wired_limit_bytes  # noqa: E402


def _model_from_gguf(gguf_path: str) -> Model:
    path = Path(gguf_path) if gguf_path else Path("/dev/null")
    return Model(
        id=path.stem or "unknown",
        name=path.stem or "unknown",
        description="",
        ttl=None,
        aliases=(),
        roles=(),
        ctx=None,
        temp=None,
        embedding=False,
        file=path if gguf_path else None,
        cmd="",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    args = parser.parse_args()

    vm = psutil.virtual_memory()
    wired = wired_limit_bytes(vm.total)
    procs = llama_server_procs()

    rows = []
    resident_total = 0
    for p in procs:
        m = _model_from_gguf(p["gguf"])
        profile = profile_for(m)
        est = estimate_model_ram(m, profile)
        rss = p["rss"]
        resident_total += rss
        rows.append(
            {
                "model": m.id,
                "pid": p["pid"],
                "rss_bytes": rss,
                "estimated_resident_bytes": est.resident_bytes,
                "estimate_known": est.known,
                "detail": est.detail,
                "kv_per_1k_tokens_bytes": None,
            }
        )
        if profile:
            per_1k = int(
                profile.n_layer
                * (profile.k_len + profile.v_len)
                * profile.n_kv_heads
                * 1.0625
                * 1024
            )
            rows[-1]["kv_per_1k_tokens_bytes"] = per_1k
        else:
            rows[-1]["kv_per_1k_tokens_bytes"] = None

    if args.json:
        print(
            json.dumps(
                {
                    "ram_total": vm.total,
                    "ram_available": vm.available,
                    "wired_limit": wired,
                    "resident_total": resident_total,
                    "models": rows,
                },
                indent=2,
            )
        )
        return

    gib = 2**30
    print("## Hearthia memory report")
    print()
    print(
        f"- Machine: {vm.total / gib:.1f} GiB unified memory, "
        f"{vm.available / gib:.1f} GiB available"
    )
    print(f"- GPU-wired ceiling: {wired / gib:.1f} GiB")
    print(
        f"- Resident model servers: {len(procs)}, holding {resident_total / gib:.1f} GiB measured"
    )
    print()
    if rows:
        print("| model | measured RSS | estimate | KV cost / 1K tok |")
        print("|---|---|---|---|")
        for r in rows:
            kv = r["kv_per_1k_tokens_bytes"]
            kv_txt = f"{kv / 2**20:.1f} MiB" if kv else "–"
            print(
                f"| {r['model']} | {r['rss_bytes'] / gib:.2f} GiB | "
                f"{r['estimated_resident_bytes'] / gib:.1f} GiB"
                f"{'' if r['estimate_known'] else ' (guess)'} | {kv_txt} |"
            )
    else:
        print("No running llama-server processes — warm a model and re-run.")
    print()
    print("The KV column is why two models of similar size differ tenfold in")
    print("real cost: context windows scale memory independently of weights.")


if __name__ == "__main__":
    main()
