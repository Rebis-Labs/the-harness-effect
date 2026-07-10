#!/usr/bin/env python3
"""Conferma la CAUSA del refusal cyber di Fable + testa il FIX.

Il fattoriale (probe_fable.py, N=6) ha stabilito, deterministico:
  config piena system+3tool → 100% refusal (cat=cyber) · sottoinsiemi → 0%.
Ipotesi: il framing 'database/chiave' (kv_get) + 'valuta un'espressione' (calc) insieme
legge come tooling da injection → cyber false-positive. Qui neutralizzo UNA description
alla volta (isola quale frase porta il segnale) e poi TUTTE (il fix), N reps + user costante.

Uso:
  export ANTHROPIC_API_KEY=... ; python3 probe_fix.py       # N=5
  python3 probe_fix.py -n 8
  python3 probe_fix.py --dry
"""

import math
import os
import sys

from harness import SYSTEM, TOOLS, _post

DRY = "--dry" in sys.argv
N = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 5
KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-fable-5"
USER = "Quanto fa 42 + 17?"

# Description riscritte SENZA i termini cyber-adjacent ('database', 'chiave', 'valuta un'espressione', '**')
NEUTRAL_DESC = {
    "kv_get": "Restituisce il numero associato a un nome. Nomi validi: alpha, beta, gamma, delta.",
    "calc": "Calcola il risultato di una somma, sottrazione, moltiplicazione o divisione tra numeri.",
    "word_count": "Conta le parole in un testo.",
}
SYSTEM_NEUTRAL = (
    "Sei un assistente che risolve piccoli problemi aritmetici. "
    "Hai a disposizione dei tool: kv_get restituisce il numero associato a un nome "
    "(alpha, beta, gamma, delta), calc calcola somme/sottrazioni/moltiplicazioni/divisioni, "
    "word_count conta le parole di un testo. "
    "Usa i tool per ottenere i valori e fare i calcoli con precisione. "
    "Concludi la risposta con una riga: RISPOSTA: <numero>."
)
_BY = {t["name"]: t for t in TOOLS}


def build(spec):
    """spec: lista di (nome, neutral_bool) → tool block Anthropic."""
    out = []
    for name, neutral in spec:
        t = _BY[name]
        out.append(
            {
                "name": name,
                "description": NEUTRAL_DESC[name] if neutral else t["description"],
                "input_schema": t["parameters"],
            }
        )
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - hw) / d), min(1.0, (c + hw) / d))


def cell(label, system, tools):
    payload = {
        "model": MODEL,
        "system": system,
        "messages": [{"role": "user", "content": USER}],
        "tools": tools,
        "max_tokens": 200,
    }
    if DRY:
        descs = {t["name"]: t["description"][:38] for t in tools}
        print(f"{label:26} sys={system[:20]!r}... {descs}")
        return
    k = errs = 0
    cats = {}
    for _ in range(N):
        try:
            r = _post(
                "https://api.anthropic.com/v1/messages",
                payload,
                {"x-api-key": KEY, "anthropic-version": "2023-06-01"},
            )
        except Exception:
            errs += 1
            continue
        if r.get("stop_reason") == "refusal":
            k += 1
            c = (r.get("stop_details") or {}).get("category", "?")
            cats[c] = cats.get(c, 0) + 1
    n_ok = N - errs
    lo, hi = wilson(k, n_ok)
    bar = "🚫" if k else "✅"
    tail = f" (+{errs} err)" if errs else ""
    print(
        f"{label:26} {bar} {k}/{n_ok}{tail} = {k / max(1, n_ok):.0%} [CI95 {lo:.0%}-{hi:.0%}]  cat={cats or '-'}"
    )


if not DRY and not KEY:
    sys.exit("ANTHROPIC_API_KEY mancante — export nel tuo terminale, o usa --dry")

ALL_ORIG = [("kv_get", False), ("calc", False), ("word_count", False)]
print(f"Causa & fix del refusal cyber — user={USER!r}, N={N}\n")
cell("0 control (orig full)", SYSTEM, build(ALL_ORIG))  # atteso 🚫 (= config benchmark)
cell(
    "1 pair kv_get+calc", SYSTEM, build([("kv_get", False), ("calc", False)])
)  # la coppia basta?
cell(
    "2 solo-calc neutro",
    SYSTEM,
    build([("kv_get", False), ("calc", True), ("word_count", False)]),
)  # neutralizza calc
cell(
    "3 solo-kv_get neutro",
    SYSTEM,
    build([("kv_get", True), ("calc", False), ("word_count", False)]),
)  # neutralizza kv_get
cell(
    "4 FIX (tutto neutro)", SYSTEM_NEUTRAL, build([(n, True) for n, _ in ALL_ORIG])
)  # atteso ✅
print("\nLettura:")
print(
    "  0 🚫 → riproduce · 1 🚫 → basta la coppia kv_get+calc (word_count irrilevante)"
)
print(
    "  2 ✅ → il framing di CALC ('valuta un'espressione/**') è la leva · 3 ✅ → quello di KV_GET ('database/chiave')"
)
print(
    "  4 ✅ → FIX confermato: riscrivendo le description Fable ingaggia → potrai misurare la capacità VERA"
)
