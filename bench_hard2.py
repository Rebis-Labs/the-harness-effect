#!/usr/bin/env python3
"""bench_hard2 — de-satura la fascia alta (piano: bench_hard2_plan.md, 16 lug).

bench_hard è saturo su entrambi i lati (locale v8 100%, frontier 94-100%).
Qui: catene 12-18 stadi, due catene interlacciate, recovery multipli in mezzo
alla catena, branch di parità ripetuti. Stessi tool, stesso harness v8: cambia
solo la LUNGHEZZA/INTRECCIO dell'orizzonte (≤ ~35 call dipendenti, dentro
max_turns=48).

⚠️ NON CONGELATO. Congelamento solo dopo qualificazione (piano):
   un task vale se il frontier di riferimento scende <80% a N>=5.

Le risposte sono CALCOLATE da _sim() alla import (zero aritmetica a mano):
il prompt e la risposta derivano dagli stessi parametri → coerenti per costruzione.

Uso:
  python3 bench_hard2.py --only mega_chain_18 -n 2 --thinking off   # canary locale
  python3 bench_hard2.py -n 5 --thinking off                        # colonna locale
  python3 bench_hard2.py --openrouter anthropic/claude-sonnet-5 -n 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime

import bench_hard
from bench_frontier import OPENROUTER_URL, ORProvider, _load_key
from bench_hard import _git_rev, run_hard_suite
from harness import KV, OpenAIProvider

HERE = pathlib.Path(__file__).parent

# ── DSL: stadio = (extract, extract_param, moltiplicatore) ──────────────────
# extract: "id" (nessuna estrazione) | "last" (ultime N cifre) |
#          "digsum" (somma cifre) | "intdiv" (parte intera della divisione per D)


def _apply(extract: str, p: int, x: int) -> int:
    if extract == "id":
        return x
    if extract == "last":
        return int(str(x)[-p:])
    if extract == "digsum":
        return sum(int(c) for c in str(x))
    if extract == "intdiv":
        return x // p
    raise ValueError(extract)


def _sim(start: int, stages: list[tuple[str, int, int]]) -> int:
    x = start
    for extract, p, k in stages:
        x = _apply(extract, p, x) * k
    return x


def _stage_txt(i: int, extract: str, p: int, k: int, prev: str) -> str:
    if extract == "id":
        return f"Stadio {i}: {prev} moltiplicato per {k}."
    if extract == "last":
        return f"Stadio {i}: le ultime {p} cifre {prev}, moltiplicate per {k}."
    if extract == "digsum":
        return f"Stadio {i}: la somma delle cifre {prev}, moltiplicata per {k}."
    if extract == "intdiv":
        return (
            f"Stadio {i}: la parte intera della divisione {prev} per {p}, "
            f"moltiplicata per {k}."
        )
    raise ValueError(extract)


def _chain_prompt(intro: str, stages: list[tuple[str, int, int]], outro: str) -> str:
    lines = []
    for n, (extract, p, k) in enumerate(stages, start=1):
        prev = "dello stadio precedente" if n > 1 else "del valore iniziale"
        lines.append(_stage_txt(n, extract, p, k, prev))
    return (
        intro + " Procedi con calc uno stadio alla volta, leggendo ogni risultato "
        "(le 'ultime N cifre' sono un numero intero, gli zeri iniziali cadono; "
        "la 'parte intera' sono le cifre prima della virgola): "
        + " ".join(lines)
        + " "
        + outro
    )


A, B, G, D = KV["alpha"], KV["beta"], KV["gamma"], KV["delta"]  # 42 17 100 9

# 1) mega_chain_18 — 18 stadi misti
_MEGA = [
    ("id", 0, 8443),
    ("last", 4, 9377),
    ("digsum", 0, 662917),
    ("last", 4, 61),
    ("last", 3, 7919),
    ("digsum", 0, 33203),
    ("last", 4, 773),
    ("intdiv", 97, 41),
    ("digsum", 0, 90107),
    ("last", 4, 33),
    ("last", 3, 6217),
    ("digsum", 0, 5081),
    ("last", 4, 7),
    ("intdiv", 53, 829),
    ("last", 4, 3163),
    ("digsum", 0, 7145203),
    ("last", 4, 61),
    ("last", 3, 13),
]
_MEGA_START = 62171


# 2) lcg_12 — 12 iterazioni LCG via divisione/parte-intera/resto (36 call dipendenti)
def _lcg(x0: int, n: int) -> int:
    x = x0
    for _ in range(n):
        x = (x * 313 + 197) % 9973
    return x


# 3) interleave_ab — due catene da 6 stadi alternate, finale = A6 - B6
_CA = [
    ("id", 0, 941),
    ("last", 3, 6673),
    ("digsum", 0, 8362),
    ("last", 4, 47),
    ("last", 3, 7919),
    ("digsum", 0, 613),
]
_CB = [
    ("id", 0, 733),
    ("last", 3, 9377),
    ("digsum", 0, 4217),
    ("last", 4, 61),
    ("last", 3, 3163),
    ("digsum", 0, 829),
]

# 4) deep_recovery_12 — 12 stadi, kv mancanti a stadio 4 (epsilon) e 9 (theta)
_DR_PRE = [("id", 0, 6673), ("last", 4, 9377), ("digsum", 0, 90107)]  # stadi 1-3
_DR_MID = [
    ("last", 4, 773),
    ("digsum", 0, 5081),
    ("last", 3, 7145),
    ("last", 4, 41),
]  # 5-8
_DR_POST = [("last", 4, 61), ("digsum", 0, 6217), ("last", 3, 33)]  # 10-12


def _deep_recovery_answer() -> int:
    x = _sim(A * G + B, _DR_PRE)  # stadi 1-3, start = alpha*gamma+beta
    x = _apply("last", 4, x) * (A + B)  # stadio 4: epsilon NOT_FOUND -> alpha+beta
    x = _sim(x, _DR_MID)  # stadi 5-8
    x = _apply("digsum", 0, x) * (D * D)  # stadio 9: theta NOT_FOUND -> delta*delta
    return _sim(x, _DR_POST)  # stadi 10-12


# 5) branch_ladder_12 — 12 stadi, ogni 3° il moltiplicatore dipende dalla parità
def _branch_ladder_answer() -> int:
    x = G * 313 + D  # start
    for i in range(1, 13):
        if i % 3 == 0:
            k = 7 if x % 2 == 0 else 11
            x = _apply("last", 4, x) * k
        elif i % 3 == 1:
            x = _apply("digsum", 0, x) * 947
        else:
            x = _apply("last", 3, x) * 6673
    return x


TASKS_HARD2 = [
    {
        "id": "mega_chain_18",
        "answer": _sim(_MEGA_START, _MEGA),
        "expected_tools": {"kv_get", "calc"},
        "prompt": _chain_prompt(
            "Leggi gamma con kv_get: se non vale il numero che ottieni dalla lettura, "
            f"rispondi 0. Il valore iniziale è {_MEGA_START}.",
            _MEGA,
            "Concludi con RISPOSTA: <numero dello stadio 18>.",
        ),
    },
    {
        "id": "lcg_12",
        "answer": _lcg(A * 100 + B, 12),
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi alpha e beta con kv_get. Il valore iniziale è x = alpha * 100 + beta "
            "(calcolalo con calc). Ripeti per ESATTAMENTE 12 passi: (1) con calc calcola "
            "v = x * 313 + 197; (2) con calc dividi v per 9973 e leggi il risultato: la sua "
            "parte intera è k; (3) con calc calcola v - 9973 * k: questo è il nuovo x. "
            "Usa SEMPRE calc con numeri espliciti, mai nomi. Dopo il passo 12, "
            "concludi con RISPOSTA: <valore finale di x>."
        ),
    },
    {
        "id": "interleave_ab",
        "answer": int(str(_sim(G + D, _CB) - _sim(A * B, _CA))[-3:]) * 41,  # diff mid-chain, finale=estrazione+mult (v8 copia l'echo sui finali aritmetici)
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Due catene INDIPENDENTI, da portare avanti ALTERNANDO: prima lo stadio i di A, "
            "poi lo stadio i di B, poi i+1 di A, e così via. Leggi alpha, beta, gamma, delta "
            "con kv_get. Valore iniziale di A = alpha * beta. Valore iniziale di B = gamma + delta. "
            "Procedi con calc uno stadio alla volta (le 'ultime N cifre' sono un numero intero, "
            "gli zeri iniziali cadono). "
            + " ".join(
                f"Catena A, {_stage_txt(i, e, p, k, 'dello stadio A precedente' if i > 1 else 'iniziale di A')}"
                for i, (e, p, k) in enumerate(_CA, 1)
            )
            + " "
            + " ".join(
                f"Catena B, {_stage_txt(i, e, p, k, 'dello stadio B precedente' if i > 1 else 'iniziale di B')}"
                for i, (e, p, k) in enumerate(_CB, 1)
            )
            + " Poi calcola con calc: risultato di B6 meno risultato di A6. "
            "Stadio finale: le ultime 3 cifre di quella differenza, moltiplicate per 41. "
            "Concludi con RISPOSTA: <numero dello stadio finale>."
        ),
    },
    {
        "id": "deep_recovery_12",
        "answer": _deep_recovery_answer(),
        "expected_tools": {"kv_get", "calc"},
        "prompt": _chain_prompt(
            "Leggi alpha, beta, gamma, delta con kv_get. Valore iniziale = alpha * gamma + beta "
            "(calcolalo con calc).",
            _DR_PRE,
            "",
        ).rstrip()
        + (
            " Stadio 4: leggi epsilon con kv_get; se ottieni NOT_FOUND usa al suo posto "
            "alpha + beta come moltiplicatore: le ultime 4 cifre dello stadio 3, moltiplicate "
            "per quel valore. "
            + " ".join(
                _stage_txt(i, e, p, k, "dello stadio precedente")
                for i, (e, p, k) in enumerate(_DR_MID, 5)
            )
            + " Stadio 9: leggi theta con kv_get; se ottieni NOT_FOUND usa al suo posto "
            "delta * delta come moltiplicatore: la somma delle cifre dello stadio 8, "
            "moltiplicata per quel valore. "
            + " ".join(
                _stage_txt(i, e, p, k, "dello stadio precedente")
                for i, (e, p, k) in enumerate(_DR_POST, 10)
            )
            + " Concludi con RISPOSTA: <numero dello stadio 12>."
        ),
    },
    {
        "id": "branch_ladder_12",
        "answer": _branch_ladder_answer(),
        "expected_tools": {"kv_get", "calc"},
        "prompt": (
            "Leggi gamma e delta con kv_get. Valore iniziale = gamma * 313 + delta "
            "(calcolalo con calc). Esegui 12 stadi, uno alla volta con calc. "
            "Per gli stadi 1, 4, 7, 10 (cioè quando il numero dello stadio diviso 3 dà resto 1): "
            "la somma delle cifre del valore corrente, moltiplicata per 947. "
            "Per gli stadi 2, 5, 8, 11 (resto 2): le ultime 3 cifre del valore corrente, "
            "moltiplicate per 6673. "
            "Per gli stadi 3, 6, 9, 12 (resto 0): guarda l'ultima cifra del valore corrente; "
            "se è pari usa 7, se è dispari usa 11 come moltiplicatore, applicato alle "
            "ultime 4 cifre del valore corrente. "
            "(Le 'ultime N cifre' sono un numero intero, gli zeri iniziali cadono.) "
            "Concludi con RISPOSTA: <numero dello stadio 12>."
        ),
    },
]

for _t in TASKS_HARD2:  # sanity: risposte intere, prompt non vuoti
    assert isinstance(_t["answer"], int), _t["id"]
    assert len(_t["prompt"]) > 100, _t["id"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--local", default="qwen/qwen3-14b")
    ap.add_argument(
        "--openrouter", default=None, help="ID modello OR al posto del locale"
    )
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    tasks = TASKS_HARD2
    if args.only:
        tasks = [t for t in tasks if t["id"] == args.only]
        if not tasks:
            raise SystemExit(f"task sconosciuto: {args.only}")
    bench_hard.TASKS_HARD = tasks  # run_hard_suite legge il global di bench_hard

    if args.openrouter:
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
            think=args.thinking == "on",
            timeout=args.timeout,
        )
        name = f"LOCAL: {args.local}"

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id,
        "bench": "hard2-NOT-FROZEN",
        "harness_md5": hashlib.md5((HERE / "harness.py").read_bytes()).hexdigest()[:8],
        "bench_hard2_md5": hashlib.md5(
            (HERE / "bench_hard2.py").read_bytes()
        ).hexdigest()[:8],
        "git_rev": _git_rev(),
        "model": args.openrouter or args.local,
        "n": args.n,
        "tasks": len(tasks),
        "only": args.only,
        "thinking": args.thinking if not args.openrouter else "native",
        "note": "NON congelato: qualifica = frontier <80% a N>=5 (bench_hard2_plan.md)",
    }
    with open(HERE / "results" / f"{run_id}_hard2.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        run_hard_suite(name, provider, args.n, logf)
    print(f"\nlog: results/{run_id}_hard2.jsonl")
