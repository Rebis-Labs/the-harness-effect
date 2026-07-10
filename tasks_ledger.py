#!/usr/bin/env python3
"""Categoria-task LEDGER: saldo + L operazioni sequenziali, ≥45% condizionali data-dependent.

Perché questa famiglia (design panel 7 lug): `calc` rende esatta l'ARITMETICA, ma le DECISIONI
DI BRANCH (il saldo è pari? è > T?) dipendono dal valore vivo e restano ragionamento puro che la
calcolatrice non tocca → resta errore residuo SOPRA il tool-floor → i rung alti (verify, voto)
hanno lavoro reale (evita il 'calc ceiling'). Procedurale = non memorizzabile. Numeri inline =
niente lookup, niente collisione col KV globale dell'harness (solo calc è load-bearing). Oracolo
= il generatore stesso esegue la sequenza in int esatti → ground truth gratis ed esatta.

Interfaccia categoria (contratto per ladder.py):
  generate(seed, L) -> (prompt:str, gold:int, meta:dict)
  verify(answer:float|None, gold:int) -> bool
"""

from __future__ import annotations

import random

X_RANGE = (1, 40)
T_RANGE = (50, 150)
START_RANGE = (50, 150)
COND_MIN_FRAC = (
    0.45  # almeno il 45% delle operazioni sono condizionali (data-dependent)
)


def _build_ops(rng: random.Random, L: int) -> list[dict]:
    n_cond = max(1, round(COND_MIN_FRAC * L))
    kinds = ["cond_parity"] * (n_cond // 2 + n_cond % 2) + ["cond_thresh"] * (
        n_cond // 2
    )
    kinds += [rng.choice(["add", "sub"]) for _ in range(L - len(kinds))]
    rng.shuffle(kinds)
    ops = []
    for k in kinds:
        if k == "add":
            ops.append({"k": "add", "x": rng.randint(*X_RANGE)})
        elif k == "sub":
            ops.append({"k": "sub", "x": rng.randint(*X_RANGE)})
        elif k == "cond_parity":
            ops.append(
                {
                    "k": "cond_parity",
                    "x": rng.randint(*X_RANGE),
                    "y": rng.randint(*X_RANGE),
                }
            )
        else:  # cond_thresh
            ops.append(
                {
                    "k": "cond_thresh",
                    "t": rng.randint(*T_RANGE),
                    "x": rng.randint(*X_RANGE),
                    "y": rng.randint(*X_RANGE),
                }
            )
    return ops


def _apply(b: int, op: dict) -> int:
    if op["k"] == "add":
        return b + op["x"]
    if op["k"] == "sub":
        return b - op["x"]
    if op["k"] == "cond_parity":
        return b + op["x"] if b % 2 == 0 else b - op["y"]
    if op["k"] == "cond_thresh":
        return b - op["x"] if b > op["t"] else b + op["y"]
    raise ValueError(op["k"])


def _oracle(start: int, ops: list[dict]) -> int:
    b = start
    for op in ops:
        b = _apply(b, op)
    return b


def _render(start: int, ops: list[dict]) -> str:
    lines = [
        f"Parti da un saldo di {start}. Applica in ordine, una alla volta, queste operazioni:"
    ]
    for i, op in enumerate(ops, 1):
        if op["k"] == "add":
            s = f"aggiungi {op['x']}"
        elif op["k"] == "sub":
            s = f"sottrai {op['x']}"
        elif op["k"] == "cond_parity":
            s = f"se il saldo attuale è pari aggiungi {op['x']}, altrimenti sottrai {op['y']}"
        else:
            s = f"se il saldo attuale è maggiore di {op['t']} sottrai {op['x']}, altrimenti aggiungi {op['y']}"
        lines.append(f"{i}) {s}")
    lines.append("Qual è il saldo finale dopo tutte le operazioni?")
    lines.append("Concludi con una riga: RISPOSTA: <numero>.")
    return "\n".join(lines)


def generate(seed: int, L: int = 12) -> tuple[str, int, dict]:
    rng = random.Random(seed)
    start = rng.randint(*START_RANGE)
    ops = _build_ops(rng, L)
    gold = _oracle(start, ops)
    n_cond = sum(1 for o in ops if o["k"].startswith("cond"))
    return (
        _render(start, ops),
        gold,
        {"seed": seed, "L": L, "start": start, "n_cond": n_cond},
    )


def verify(answer: float | None, gold: int) -> bool:
    return answer is not None and abs(float(answer) - float(gold)) < 1e-6


if __name__ == "__main__":
    # Sanity: mostra un esempio + verifica che l'oracolo sia stabile e i condizionali presenti.
    prompt, gold, meta = generate(seed=7, L=12)
    print(prompt)
    print(f"\nGOLD = {gold}   (meta: {meta})")
    assert meta["n_cond"] >= round(COND_MIN_FRAC * meta["L"]), (
        "troppi pochi condizionali"
    )
    # rigenerare con lo stesso seed dà lo stesso gold (determinismo)
    assert generate(7, 12)[1] == gold, "non deterministico!"
    print(f"\n✅ deterministico, {meta['n_cond']}/{meta['L']} condizionali")
