#!/usr/bin/env python3
"""bench_hard — benchmark DURO per qwen3-14b: prova che capability = model × harness.

Protocollo (pre-registrato, vedi goal 15 lug):
  1. Baseline con harness CORRENTE a N>=5 → qualifica solo se ANSWER-ACC <= 60%.
  2. Dopo la baseline: TASKS_HARD e score_strict CONGELATI — mai più editati.
  3. Si migliora SOLO harness.py (loop, assi thinking/repair/fallback, wording tool) →
     target: ANSWER-ACC >= baseline + 25pp a N>=5, con Wilson CI95 lower bound
     sopra il point estimate della baseline. Modello fisso: qwen/qwen3-14b.

Import da harness.py/bench.py INVARIATI: provider, run_agent, score_strict.
Design dei task (ogni task >= 3 tool call sequenziali):
  - chiavi kv distrattrici (sigma/omega citate ma inutili; zeta/epsilon INESISTENTI)
  - aritmetica annidata/composta con trappole di precedenza
  - >= 2 task dove un tool RITORNA UN ERRORE da cui l'agente deve recuperare:
    err_fallback (NOT_FOUND obbligato), double_not_found (2× NOT_FOUND),
    div_zero (calc → ERROR su divisione per zero)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from datetime import datetime

from bench import score_lenient, score_strict, wilson
from harness import OpenAIProvider, run_agent

HERE = pathlib.Path(__file__).parent

# ── TASKS_HARD v4 — CONGELATI dopo la baseline qualificata (<=60%) ──
# Storia indurimento (i canary sono nei results/ del 15 lug):
# v1 (run 20260715T020617, N=5): 45/45 = 100% → NON qualificata.
# v2: il modello BATCHA decine di call per turno predicendo in-testa gli
#   argomenti dipendenti (83 call in 3 turni) → tetto turni aggirato.
# v3: mult 6x3 cifre in stadi dipendenti → il 14B le macina IN-TESTA nel
#   thinking (temp=0) e risponde giusto anche con 5 calc andate in ERROR
#   (argomenti simbolici 'result * 641'). LCG 12 passi mod 97: pure in-testa.
# v4, la morsa a doppio muro:
#   (a) JOINT di estrazione-cifre ("ultime 4 cifre del risultato", "somma
#       delle cifre") tra gli stadi: l'argomento del calc successivo richiede
#       il VALORE REALE del tool → niente composizione simbolica, niente
#       predizione; ogni joint = 1 round-trip obbligato.
#   (b) >=5 joint per task → servono >=7 turni assistant, ma max_turns=6:
#       la via-tool corretta è AFFAMATA dall'harness corrente.
#   (c) mult 7x7 e 7x4 cifre negli stadi → la scorciatoia in-testa scivola.
# KV reali: alpha=42, beta=17, gamma=100, delta=9. zeta/epsilon/omega/sigma NON esistono.
TASKS_HARD = [
    {
        # recovery #1: NOT_FOUND obbligato + 7 joint di estrazione-cifre
        "id": "err_fallback",
        "answer": 7590,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi epsilon con kv_get. Se ottieni NOT_FOUND, usa 8377 come valore di epsilon. "
            "Poi leggi alpha, beta e delta con kv_get. Procedi con calc uno stadio alla volta, "
            "leggendo ogni volta il risultato prima di proseguire "
            "(le 'ultime N cifre' vanno prese come numero intero, es. le ultime 4 cifre di 1234567 sono 4567): "
            "Stadio 1: (alpha + beta) * epsilon. Stadio 2: le ultime 4 cifre dello stadio 1, moltiplicate per 96179. "
            "Stadio 3: la somma delle cifre dello stadio 2, moltiplicata per 7145203. "
            "Stadio 4: le ultime 4 cifre dello stadio 3, moltiplicate per 6373. "
            "Stadio 5: le ultime 3 cifre dello stadio 4, moltiplicate per 8362917. "
            "Stadio 6: le ultime 4 cifre dello stadio 5, moltiplicate per 417. "
            "Stadio 7: la somma delle cifre dello stadio 6, moltiplicata per 9241. "
            "Stadio 8: le ultime 3 cifre dello stadio 7, moltiplicate per (alpha - delta). "
            "Concludi con RISPOSTA: <numero dello stadio 8>."
        ),
    },
    {
        # recovery #2: divisione per zero → ERROR → percorso alternativo con 5 joint
        "id": "div_zero",
        "answer": 88812,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi gamma e delta con kv_get. Prova a calcolare con calc: (gamma * delta) / (delta - 9). "
            "Se calc restituisce un errore o un risultato non valido, procedi con questo percorso alternativo, "
            "uno stadio alla volta con calc, leggendo ogni risultato "
            "(le 'ultime N cifre' sono un numero intero): "
            "Stadio A: (gamma + delta) * 48731. Stadio B: le ultime 4 cifre dello stadio A, moltiplicate per 7919. "
            "Stadio C: la somma delle cifre dello stadio B, moltiplicata per 62171. "
            "Stadio D: le ultime 3 cifre dello stadio C, moltiplicate per 907. "
            "Stadio E: le ultime 4 cifre dello stadio D, moltiplicate per 3141. "
            "Stadio F: la somma delle cifre dello stadio E, moltiplicata per 2467. "
            "Concludi con RISPOSTA: <numero dello stadio F>."
        ),
    },
    {
        # recovery #3: DUE chiavi inesistenti (3187 ciascuna) + 6 joint
        "id": "double_not_found",
        "answer": 12525,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi zeta e omega con kv_get. Regola: ogni valore NOT_FOUND vale 3187. "
            "Poi leggi alpha e gamma con kv_get. Procedi con calc uno stadio alla volta, "
            "leggendo ogni risultato (le 'ultime N cifre' sono un numero intero): "
            "Stadio 1: zeta * omega. Stadio 2: le ultime 4 cifre dello stadio 1, moltiplicate per 8383. "
            "Stadio 3: la somma delle cifre dello stadio 2, moltiplicata per 95917. "
            "Stadio 4: le ultime 4 cifre dello stadio 3, moltiplicate per 773. "
            "Stadio 5: le ultime 3 cifre dello stadio 4, moltiplicate per 6841. "
            "Stadio 6: le ultime 4 cifre dello stadio 5, moltiplicate per (alpha + gamma). "
            "Stadio 7: la somma delle cifre dello stadio 6, moltiplicata per 501. "
            "Concludi con RISPOSTA: <numero dello stadio 7>."
        ),
    },
    {
        # distrattori + trappola di segno + 5 joint (valore assoluto esplicito)
        "id": "distractor_nest",
        "answer": 2921,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Nel sistema esistono anche i nomi sigma e omega ma NON ti servono: ignorali. "
            "Leggi alpha, beta, gamma e delta con kv_get. Procedi con calc uno stadio alla volta, "
            "leggendo ogni risultato. Quando estrai cifre da un numero NEGATIVO usa il suo valore assoluto "
            "(le 'ultime N cifre' sono un numero intero): "
            "Stadio 1: alpha * beta - delta * gamma. Stadio 2: risultato * 6373. "
            "Stadio 3: le ultime 4 cifre del valore assoluto dello stadio 2, moltiplicate per 947. "
            "Stadio 4: la somma delle cifre dello stadio 3, moltiplicata per 81703. "
            "Stadio 5: le ultime 4 cifre dello stadio 4, moltiplicate per 33. "
            "Stadio 6: le ultime 3 cifre dello stadio 5, moltiplicate per 7145. "
            "Stadio 7: le ultime 4 cifre dello stadio 6, meno il prodotto alpha * beta. "
            "Concludi con RISPOSTA: <numero dello stadio 7>."
        ),
    },
    {
        # catena pura lunga: 9 joint → la via-tool corretta chiede ~11 turni
        "id": "staged_chain",
        "answer": 1749,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi gamma e beta con kv_get (ti servono solo come controllo: se gamma non vale 100 rispondi 0). "
            "Procedi con calc uno stadio alla volta, leggendo ogni risultato "
            "(le 'ultime N cifre' sono un numero intero, gli zeri iniziali cadono): "
            "Stadio 1: 62171 * 8443. Stadio 2: le ultime 4 cifre dello stadio 1, moltiplicate per 96179. "
            "Stadio 3: la somma delle cifre dello stadio 2, moltiplicata per 8362917. "
            "Stadio 4: le ultime 4 cifre dello stadio 3, moltiplicate per 61. "
            "Stadio 5: le ultime 3 cifre dello stadio 4, moltiplicate per 7919. "
            "Stadio 6: le ultime 4 cifre dello stadio 5, moltiplicate per 773. "
            "Stadio 7: la somma delle cifre dello stadio 6, moltiplicata per 90107. "
            "Stadio 8: le ultime 4 cifre dello stadio 7, moltiplicate per 33. "
            "Stadio 9: le ultime 3 cifre dello stadio 8, moltiplicate per 6217. "
            "Stadio 10: le ultime 4 cifre dello stadio 9, moltiplicate per 3. "
            "Concludi con RISPOSTA: <numero dello stadio 10>."
        ),
    },
    {
        # iterazione con stato: 8 passi di (x*313+197) mod 9973 via divisione+resto
        "id": "lcg_iter",
        "answer": 3516,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi alpha e beta con kv_get. Il valore iniziale è x = alpha * 100 + beta (calcolalo con calc). "
            "Ripeti per ESATTAMENTE 8 passi: (1) con calc calcola v = x * 313 + 197; "
            "(2) con calc dividi v per 9973 e leggi il risultato: la sua parte intera è k; "
            "(3) con calc calcola v - 9973 * k: questo è il nuovo x. "
            "Usa SEMPRE calc con numeri espliciti, mai nomi. Dopo il passo 8, "
            "concludi con RISPOSTA: <valore finale di x>."
        ),
    },
    {
        # word_count su stringhe-esca lunghe + 4 joint + controllo kv
        "id": "wc_trap",
        "answer": 963,
        "expected_tools": {"word_count", "kv_get", "calc"},
        "prompt": (
            "Usa word_count sul testo: 'zeta uno omega due tre sigma quattro epsilon cinque zeta sei "
            "omega sette sigma otto'. Poi usa word_count sul testo: 'alpha nove beta dieci gamma undici "
            "delta dodici alpha tredici beta' (conta le PAROLE, non leggere i valori). "
            "Poi leggi delta con kv_get; se non vale 9 fermati e rispondi 0. "
            "Procedi con calc uno stadio alla volta, leggendo ogni risultato "
            "(le 'ultime N cifre' sono un numero intero): "
            "Stadio 1: primo conteggio * secondo conteggio. Stadio 2: risultato * 96179. "
            "Stadio 3: le ultime 4 cifre dello stadio 2, moltiplicate per 4217. "
            "Stadio 4: la somma delle cifre dello stadio 3, moltiplicata per 7145203. "
            "Stadio 5: le ultime 4 cifre dello stadio 4, moltiplicate per 61. "
            "Stadio 6: le ultime 3 cifre dello stadio 5, moltiplicate per delta. "
            "Concludi con RISPOSTA: <numero dello stadio 6>."
        ),
    },
    {
        # condizionale su parità DI UNA JOINT (somma cifre di un prodotto 7x4) + 5 joint
        "id": "parity_branch",
        "answer": 438165,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi gamma e delta con kv_get. Procedi con calc uno stadio alla volta, leggendo ogni "
            "risultato (le 'ultime N cifre' sono un numero intero): "
            "Stadio 1: 8362917 * 4873. Stadio 2: calcola la somma delle cifre dello stadio 1. "
            "Se quella somma è PARI moltiplicala per 12334 con calc; se è DISPARI moltiplicala per 24671 con calc. "
            "Stadio 3: le ultime 4 cifre dello stadio 2, moltiplicate per 7919. "
            "Stadio 4: la somma delle cifre dello stadio 3, moltiplicata per 90107. "
            "Stadio 5: le ultime 3 cifre dello stadio 4, moltiplicate per 7145. "
            "Stadio 6: le ultime 4 cifre dello stadio 5, moltiplicate per (gamma - delta). "
            "Concludi con RISPOSTA: <numero dello stadio 6>."
        ),
    },
    {
        # quadrati di 4 cifre (muro in-testa) + quadrato della somma-cifre + 4 joint
        "id": "square_gap",
        "answer": 17759,
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Il nome sigma non ti serve: non leggerlo. Leggi gamma, delta, alpha e beta con kv_get. "
            "Procedi con calc uno stadio alla volta, leggendo ogni risultato "
            "(le 'ultime N cifre' sono un numero intero): "
            "Stadio 1: gamma * delta * 10 + beta. Stadio 2: il quadrato dello stadio 1 (stadio1 * stadio1). "
            "Stadio 3: le ultime 4 cifre dello stadio 2, moltiplicate per 8779. "
            "Stadio 4: la somma delle cifre dello stadio 3. Stadio 5: il quadrato dello stadio 4, "
            "moltiplicato per 6373. Stadio 6: le ultime 4 cifre dello stadio 5, moltiplicate per 417. "
            "Stadio 7: le ultime 3 cifre dello stadio 6, moltiplicate per (alpha + beta). "
            "Concludi con RISPOSTA: <numero dello stadio 7>."
        ),
    },
]

_FROZEN_MSG = (
    "TASKS_HARD e score_strict sono CONGELATI dopo la baseline registrata. "
    "Si migliora solo harness.py."
)


def _git_rev() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=HERE,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "n/a"
        )
    except Exception:
        return "n/a"


def run_hard_suite(name: str, provider, n: int, logf, repair: int = 0) -> dict:
    print(f"\n===== {name}  (N={n}{f', repair≤{repair}' if repair else ''}) =====")
    agg = {
        "strict": 0,
        "strict_clean": 0,
        "strict_repaired": 0,
        "lenient": 0,
        "runs_ok": 0,
        "errors": 0,
        "trunc": 0,
        "recov": 0,
        "max_turns_hit": 0,
        "traj_ok": 0,
        "traj_n": 0,
        "tool_calls": 0,
        "tok_out": 0,
        "lat": 0.0,
        "refusal": 0,
    }
    per_task = {}
    for t in TASKS_HARD:
        s = ln = err = ref = rep = mt = 0
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
            mt += r["final"] == "[MAX_TURNS]"
            agg["strict"] += strict
            agg["strict_repaired"] += strict and r["repaired"]
            agg["strict_clean"] += strict and not r["repaired"]
            agg["lenient"] += lenient
            agg["trunc"] += r["truncated"]
            agg["recov"] += r["recovered_from_reasoning"]
            agg["max_turns_hit"] += r["final"] == "[MAX_TURNS]"
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
        per_task[t["id"]] = {"strict": s, "lenient": ln, "err": err, "refusal": ref}
        flags = (f" refusal={ref}" if ref else "") + (f" repaired={rep}" if rep else "")
        flags += f" max_turns={mt}" if mt else ""
        print(f"  {t['id']:17} strict={s}/{n} lenient={ln}/{n}{flags} err={err}")
    ro = max(1, agg["runs_ok"])
    answered = ro - agg["refusal"]
    if answered > 0:
        lo, hi = wilson(agg["strict"], answered)
        p = agg["strict"] / answered
        acc = (
            f"ANSWER-ACC {agg['strict']}/{answered} = {p:.1%} "
            f"[CI95 {lo:.1%}-{hi:.1%}] | lenient {agg['lenient'] / answered:.0%}"
        )
        agg["wilson_lo"], agg["wilson_hi"], agg["answer_acc"] = lo, hi, p
    else:
        acc = "ANSWER-ACC n/a (tutte rifiutate)"
    rep_note = (
        f" | strict clean {agg['strict_clean']} + repaired {agg['strict_repaired']} (MAI sommare)"
        if agg["strict_repaired"]
        else ""
    )
    print(
        f"  ── {name}: {acc}{rep_note} | REFUSAL {agg['refusal']}/{ro} | err {agg['errors']} | "
        f"trunc {agg['trunc']}/{ro} | max_turns {agg['max_turns_hit']}/{ro} | "
        f"traj {agg['traj_ok']}/{max(1, agg['traj_n'])} | "
        f"~{agg['tok_out'] // ro} tok_out | {agg['lat'] / ro:.1f}s/task"
    )
    return {"name": name, "per_task": per_task, **agg, "runs_ok": ro, "answered": answered}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=_FROZEN_MSG)
    ap.add_argument("-n", type=int, default=5, help="ripetizioni per task (goal: N>=5)")
    ap.add_argument("--local", default="qwen/qwen3-14b")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--no-reasoning-fallback", action="store_true")
    ap.add_argument("--repair", type=int, default=0, help="asse harness: max round repair loop")
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", default=None, help="task id singolo (canary/debug)")
    args = ap.parse_args()

    tasks = TASKS_HARD
    if args.only:
        tasks = [t for t in TASKS_HARD if t["id"] == args.only]
        if not tasks:
            raise SystemExit(f"task '{args.only}' non trovato")
        TASKS_HARD = tasks  # canary: riduce la lista SOLO per questo processo

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    hmd5 = hashlib.md5((HERE / "harness.py").read_bytes()).hexdigest()[:8]
    bmd5 = hashlib.md5((HERE / "bench_hard.py").read_bytes()).hexdigest()[:8]
    manifest = {
        "run_id": run_id,
        "bench": "hard",
        "harness_md5": hmd5,
        "bench_hard_md5": bmd5,
        "git_rev": _git_rev(),
        "model": args.local,
        "n": args.n,
        "tasks": len(tasks),
        "only": args.only,
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
        "reasoning_fallback": not args.no_reasoning_fallback,
        "repair": args.repair,
        "thinking": args.thinking,
        "timeout_s": args.timeout,
        "note": "bench_hard: task congelati post-baseline; si migliora solo harness.py",
    }

    with open(results_dir / f"{run_id}.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        summary = run_hard_suite(
            f"LOCAL: {args.local}",
            OpenAIProvider(
                model=args.local,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                use_reasoning_fallback=not args.no_reasoning_fallback,
                think=args.thinking == "on",
                timeout=args.timeout,
            ),
            args.n,
            logf,
            repair=args.repair,
        )
        logf.write(
            json.dumps(
                {"SUMMARY": {k: v for k, v in summary.items() if k != "per_task"},
                 "per_task": summary["per_task"]}
            )
            + "\n"
        )

    print(f"\nlog completo: results/{run_id}.jsonl")
