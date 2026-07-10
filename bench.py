#!/usr/bin/env python3
"""Benchmark: STESSO harness, modello diverso → misura il modello isolando l'harness.

Rigore aggiunto (review 6 lug): N ripetizioni + Wilson CI, scorer STRICT sulla riga
RISPOSTA (no falsi positivi da numeri intermedi), ERROR separato da FAIL, retry/backoff
nel provider, logging JSONL completo (transcript+usage+latency+finish_reason) + manifest.

Uso:
  python3 bench.py                                     # locale, N=3
  python3 bench.py -n 5                                # 5 ripetizioni/task
  python3 bench.py --anthropic claude-fable-5          # + Fable (serve ANTHROPIC_API_KEY)
  python3 bench.py --max-tokens 1024                   # per replicare il vecchio confound di truncation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
from datetime import datetime

from harness import AnthropicProvider, OpenAIProvider, run_agent

HERE = pathlib.Path(__file__).parent

# Task che richiedono il loop. expected_tools = insieme minimo atteso (trajectory-check leggero).
TASKS = [
    {
        "id": "sum_two",
        "answer": 59,
        "expected_tools": {"kv_get", "calc"},
        "prompt": "Usa kv_get per leggere alpha e beta, poi calc per sommarli. Dimmi il risultato.",
    },
    {
        "id": "chain_mul",
        "answer": 273,
        "expected_tools": {"kv_get", "calc"},
        "prompt": "Leggi gamma e delta con kv_get, poi calcola (gamma - delta) * 3. Dammi il numero.",
    },
    {
        "id": "three_sum",
        "answer": 53,
        "expected_tools": {"kv_get", "calc"},
        "prompt": "Leggi alpha, beta, gamma con kv_get. Sommali tutti e tre, poi dividi per 3. Numero finale?",
    },
    {
        "id": "word_count",
        "answer": 6,
        "expected_tools": {"word_count"},
        "prompt": "Quante parole ci sono in: 'the quick brown fox jumps over'? Usa word_count.",
    },
    {
        "id": "no_tool",
        "answer": 51,
        "expected_tools": None,  # trajectory non valutata (vedi README limiti)
        "prompt": "Quanto fa 17 per 3? Rispondi solo col numero.",
    },
]

_RISPOSTA = re.compile(r"RISPOSTA:\s*(-?\d+(?:\.\d+)?)", re.I)


def score_strict(final: str, answer: float) -> bool:
    """L'ULTIMA riga 'RISPOSTA: <n>' con confronto NUMERICO (no match su numeri intermedi).
    findall()[-1], non search(): un rung verify/repair può emettere più righe RISPOSTA
    (bozza + corretta) → va scorata la FINALE, non la prima. (Panel design 7 lug.) # VERIFIED"""
    ms = _RISPOSTA.findall(final or "")
    if not ms:
        return False
    try:
        return abs(float(ms[-1]) - float(answer)) < 1e-6
    except ValueError:
        return False


def score_lenient(final: str, answer: float) -> bool:
    nums = re.findall(r"-?\d+\.?\d*", (final or "").replace(",", ""))
    return any(
        abs(float(n) - float(answer)) < 1e-6 for n in nums if n not in (".", "-")
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - hw) / d), min(1.0, (c + hw) / d))


def run_suite(name: str, provider, n: int, logf, repair: int = 0) -> dict:
    print(f"\n===== {name}  (N={n}{f', repair≤{repair}' if repair else ''}) =====")
    agg = {
        "strict": 0,
        "strict_clean": 0,  # pass SENZA repair — il numero comparabile coi run storici
        "strict_repaired": 0,  # pass CON repair — capacità del LOOP, mai sommarli ai clean
        "lenient": 0,
        "runs_ok": 0,
        "errors": 0,
        "trunc": 0,
        "recov": 0,
        "traj_ok": 0,
        "traj_n": 0,
        "tool_calls": 0,
        "tok_out": 0,
        "lat": 0.0,
        "refusal": 0,
    }
    per_task = {}
    for t in TASKS:
        s = ln = err = ref = rep = 0
        for _ in range(n):
            try:
                r = run_agent(
                    provider,
                    t["prompt"],
                    required_tools=t["expected_tools"],
                    max_repairs=repair,
                )
            except Exception as e:  # rete/API già ritentata → ERROR, non FAIL
                err += 1
                agg["errors"] += 1
                logf.write(json.dumps({"task": t["id"], "error": str(e)}) + "\n")
                continue
            agg["runs_ok"] += 1
            strict = score_strict(r["final"], t["answer"])
            lenient = score_lenient(r["final"], t["answer"])
            s += strict
            ln += lenient
            rep += r["repaired"]
            agg["strict"] += strict
            agg["strict_repaired"] += strict and r["repaired"]
            agg["strict_clean"] += strict and not r["repaired"]
            agg["lenient"] += lenient
            agg["trunc"] += r["truncated"]
            agg["recov"] += r["recovered_from_reasoning"]
            agg["tool_calls"] += r["tool_calls"]
            agg["tok_out"] += r["tokens_out"]
            agg["lat"] += r["latency_s"]
            agg["refusal"] += r["finish_reason"] == "refusal"
            ref += r["finish_reason"] == "refusal"
            if t["expected_tools"] is not None:
                agg["traj_n"] += 1
                agg["traj_ok"] += t["expected_tools"].issubset(set(r["tool_names"]))
            logf.write(
                json.dumps(
                    {
                        "task": t["id"],
                        "answer": t["answer"],
                        "strict": strict,
                        "lenient": lenient,
                        "repaired": r["repaired"],
                        "repair_rounds": r["repair_rounds"],
                        "final": r["final"][:200],
                        "tool_names": r["tool_names"],
                        "turns": r["turns"],
                        "truncated": r["truncated"],
                        "recovered_from_reasoning": r["recovered_from_reasoning"],
                        "finish_reason": r["finish_reason"],
                        "tokens_out": r["tokens_out"],
                        "latency_s": r["latency_s"],
                    }
                )
                + "\n"
            )
        per_task[t["id"]] = (s, ln, err, ref)
        rf = f" refusal={ref}" if ref else ""
        rp = f" repaired={rep}" if rep else ""
        print(f"  {t['id']:10} strict={s}/{n} lenient={ln}/{n}{rf}{rp} err={err}")
    ro = max(1, agg["runs_ok"])
    # Un refusal (safety-classifier decline) NON è una risposta sbagliata: è un TERZO esito.
    # La capacità si misura sulle chiamate TENTATE (answered), il refusal si riporta a parte.
    # Contarlo come strict=0 = misurare "il guardrail è scattato?" spacciandolo per "sa fare il conto?".
    # (Verdetto workflow fable-refusal-diagnosis, 6 lug — Fable 15/15 refusal su aritmetica banale.) # VERIFIED
    answered = ro - agg["refusal"]
    if answered > 0:
        lo, hi = wilson(agg["strict"], answered)
        acc = f"ANSWER-ACC {agg['strict']}/{answered} = {agg['strict'] / answered:.0%} [CI95 {lo:.0%}-{hi:.0%}] | lenient {agg['lenient'] / answered:.0%}"
    else:
        acc = "ANSWER-ACC n/a (tutte rifiutate: 0 chiamate tentate)"
    # clean vs repaired SEMPRE separati: 'clean' è il numero comparabile coi run storici,
    # 'repaired' misura il valore del repair loop. La loro somma NON è una metrica. # VERIFIED
    rep = (
        f" | strict clean {agg['strict_clean']} + repaired {agg['strict_repaired']} (MAI sommare)"
        if agg["strict_repaired"]
        else ""
    )
    print(
        f"  ── {name}: {acc}{rep} | "
        f"REFUSAL {agg['refusal']}/{ro} | err {agg['errors']} | trunc {agg['trunc']}/{ro} | recov {agg['recov']}/{ro} | "
        f"traj {agg['traj_ok']}/{max(1, agg['traj_n'])} | "
        f"~{agg['tok_out'] // ro} tok_out | {agg['lat'] / ro:.1f}s/task"
    )
    return {
        "name": name,
        "per_task": per_task,
        **agg,
        "runs_ok": ro,
        "answered": answered,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3, help="ripetizioni per task")
    ap.add_argument(
        "--local", default="qwen/qwen3-14b"
    )  # 32B usabile con --local qwen3-32b-abliterated
    ap.add_argument(
        "--anthropic", default=None, help="es. claude-fable-5 (serve ANTHROPIC_API_KEY)"
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--no-reasoning-fallback", action="store_true")
    ap.add_argument(
        "--no-local",
        action="store_true",
        help="salta la suite LOCAL (lenta) → solo Anthropic",
    )
    ap.add_argument(
        "--repair",
        type=int,
        default=0,
        help="max round di repair loop (0=off). I pass riparati sono contati A PARTE.",
    )
    ap.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="on",
        help="off appende /no_think (SOLO locale; con --anthropic resta on, asimmetria dichiarata)",
    )
    ap.add_argument(
        "--timeout", type=int, default=600, help="timeout per chiamata (s), no retry"
    )
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    hmd5 = hashlib.md5((HERE / "harness.py").read_bytes()).hexdigest()[:8]
    manifest = {
        "run_id": run_id,
        "harness_md5": hmd5,
        "n": args.n,
        "tasks": len(TASKS),
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
        "reasoning_fallback": not args.no_reasoning_fallback,
        "repair": args.repair,
        "thinking": args.thinking,
        "timeout_s": args.timeout,
        "note": "locale = Q4 quantizzato; thinking/repair = assi harness dichiarati; "
        "confronto tra DEPLOYMENT, non modelli ideali",
    }

    # buffering=1: i trial completati sopravvivono a un kill a metà run (lezione 10 lug)
    with open(results_dir / f"{run_id}.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        fb = not args.no_reasoning_fallback
        if not args.no_local:
            run_suite(
                f"LOCAL: {args.local}",
                OpenAIProvider(
                    model=args.local,
                    temperature=args.temp,
                    max_tokens=args.max_tokens,
                    use_reasoning_fallback=fb,
                    think=args.thinking == "on",
                    timeout=args.timeout,
                ),
                args.n,
                logf,
                repair=args.repair,
            )
        if args.anthropic:
            # think NON passato: Anthropic è sempre-on (il provider rifiuta think=False);
            # regime misto locale-off/frontiera-on è VIETATO in ladder, qui è dichiarato nel manifest.
            run_suite(
                f"ANTHROPIC: {args.anthropic}",
                AnthropicProvider(
                    model=args.anthropic,
                    temperature=args.temp,
                    max_tokens=args.max_tokens,
                    use_reasoning_fallback=fb,
                    timeout=args.timeout,
                ),
                args.n,
                logf,
                repair=args.repair,
            )

    print(f"\nlog completo: results/{run_id}.jsonl  |  manifest: {manifest}")
    if not args.anthropic:
        print(
            "(passa --anthropic claude-fable-5 con ANTHROPIC_API_KEY per la colonna Fable)"
        )
