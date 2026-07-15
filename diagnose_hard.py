#!/usr/bin/env python3
"""Diagnosi failure-mode di un run bench_hard: per ogni task fallito, mostra
tool trace + finale, e classifica il fallimento (MAX_TURNS / no-tool / valore
sbagliato / formato). Uso: python3 diagnose_hard.py results/<run>.jsonl [task_id]"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def classify(t: dict) -> str:
    if t["final"] == "[MAX_TURNS]":
        return "MAX_TURNS"
    if not t["tool_names"]:
        return "NO_TOOL"
    if "RISPOSTA" not in (t["final"] or "").upper():
        return "NO_FORMAT"
    return "WRONG_VALUE"


def main() -> None:
    path = sys.argv[1]
    only = sys.argv[2] if len(sys.argv) > 2 else None
    rows = [json.loads(line) for line in open(path)]
    trials = [r for r in rows if "task" in r and "error" not in r]
    fails = [t for t in trials if not t["strict"] and (not only or t["task"] == only)]
    by_mode: dict[str, list] = defaultdict(list)
    for t in fails:
        by_mode[classify(t)].append(t)
    n_by_task: dict[str, int] = defaultdict(int)
    s_by_task: dict[str, int] = defaultdict(int)
    for t in trials:
        n_by_task[t["task"]] += 1
        s_by_task[t["task"]] += t["strict"]
    print(f"trials {len(trials)} | fail {len(fails)}")
    for task in n_by_task:
        marker = " ←" if s_by_task[task] < n_by_task[task] else ""
        print(f"  {task:18} {s_by_task[task]}/{n_by_task[task]}{marker}")
    print("\nfailure modes:")
    for mode, ts in sorted(by_mode.items(), key=lambda kv: -len(kv[1])):
        print(f"\n── {mode} ({len(ts)}) ──")
        for t in ts:
            print(
                f"  [{t['task']}] turns={t['turns']} calls={len(t['tool_names'])} "
                f"tools={t['tool_names']} tok_out={t['tokens_out']}"
            )
            final = (t["final"] or "").strip().replace("\n", " ⏎ ")
            print(f"    final: {final[:180]}")


if __name__ == "__main__":
    main()
