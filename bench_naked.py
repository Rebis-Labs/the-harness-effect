#!/usr/bin/env python3
"""bench_naked — CAPACITÀ PURA: ledger senza tool, il modello risolve IN TESTA.

Complementare a bench_hard: là il tool `calc` fa l'aritmetica e l'harness misura la
DISCIPLINA del loop; qui NIENTE tool, NEUTRAL_SYSTEM, 1 turno → si misura il ragionamento
sequenziale + working-memory del MODELLO nudo. Task = ledger (numeri inline, self-contained,
oracolo esatto). È il rung h0 della ladder, esteso alle colonne frontiera via OpenRouter.

Confound dichiarato: locale e frontier girano entrambi in modalità THINKING NATIVA
(qui il ragionamento È la capacità misurata → giusto lasciarlo, a differenza di bench_hard
dove il think-off evitava la truncation da pianificazione-tool). Seed fissi = stesso set per
tutti i modelli.

Uso:
  python3 bench_naked.py --local -L 24 -n 5                          # colonna locale
  python3 bench_naked.py --openrouter anthropic/claude-sonnet-5 -L 24 -n 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from datetime import datetime

import tasks_ledger as task
from harness import NEUTRAL_SYSTEM, OpenAIProvider, run_agent

HERE = pathlib.Path(__file__).parent
_RISP = re.compile(r"RISPOSTA:\s*(-?\d+(?:\.\d+)?)", re.I)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    import math

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - hw) / d), min(1.0, (c + hw) / d))


def _parse(final: str) -> float | None:
    ms = _RISP.findall(final or "")
    return float(ms[-1]) if ms else None


def run(name: str, provider, L: int, n: int, logf) -> None:
    print(f"\n===== {name}  (naked, L={L}, N={n}) =====")
    ok = err = 0
    lat = 0.0
    tok = 0
    for i in range(n):
        prompt, gold, meta = task.generate(seed=1000 + i, L=L)
        try:
            r = run_agent(
                provider,
                prompt,
                max_turns=1,
                system=NEUTRAL_SYSTEM,
                tools_enabled=False,
            )
        except Exception as e:
            err += 1
            logf.write(json.dumps({"i": i, "L": L, "error": str(e)}) + "\n")
            continue
        ans = _parse(r["final"])
        good = task.verify(ans, gold)
        ok += good
        lat += r["latency_s"]
        tok += r["tokens_out"]
        logf.write(
            json.dumps(
                {
                    "i": i,
                    "L": L,
                    "gold": gold,
                    "answer": ans,
                    "pass": bool(good),
                    "final": (r["final"] or "")[-80:],
                    "tok": r["tokens_out"],
                    "lat": round(r["latency_s"], 1),
                    "finish": r["finish_reason"],
                }
            )
            + "\n"
        )
        print(f"  task {i}: {'✓' if good else '✗'} (gold={gold}, ans={ans})")
    done = max(1, n - err)
    lo, hi = _wilson(ok, done)
    print(
        f"  ── {name}: PURE-ACC {ok}/{done} = {ok / done:.0%} "
        f"[CI95 {lo:.0%}-{hi:.0%}] | err {err} | ~{tok // done} tok | {lat / done:.1f}s/task"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--local", default="qwen/qwen3-14b", help="modello locale LM Studio"
    )
    ap.add_argument("--openrouter", default=None, help="ID OR al posto del locale")
    ap.add_argument("-L", type=int, default=24, help="lunghezza ledger")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if args.openrouter:
        from bench_frontier import OPENROUTER_URL, ORProvider, _load_key

        provider = ORProvider(
            model=args.openrouter,
            base_url=OPENROUTER_URL,
            api_key=_load_key(),
            max_tokens=args.max_tokens,
            think=True,
            timeout=args.timeout,
            omit_sampling=args.openrouter.startswith("anthropic/"),
        )
        name = f"OR: {args.openrouter}"
    else:
        provider = OpenAIProvider(
            model=args.local,
            max_tokens=args.max_tokens,
            think=True,
            timeout=args.timeout,
        )
        name = f"LOCAL: {args.local}"

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id,
        "bench": "naked-ledger",
        "harness_md5": hashlib.md5((HERE / "harness.py").read_bytes()).hexdigest()[:8],
        "model": args.openrouter or args.local,
        "L": args.L,
        "n": args.n,
        "thinking": "native",
        "note": "rung h0 naked, no tools, ledger self-contained",
    }
    with open(HERE / "results" / f"{run_id}_naked.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        run(name, provider, args.L, args.n, logf)
    print(f"\nlog: results/{run_id}_naked.jsonl")
