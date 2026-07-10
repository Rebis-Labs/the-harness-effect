#!/usr/bin/env python3
"""LADDER — misura quanto l'HARNESS aggiunge all'intelligenza osservata, isolato dal compute.

Modello FISSO, harness VARIABILE lungo una scala; + asse-modello (locale / Fable / Opus) per la
heatmap gap-to-frontier. Metodologia robusta-ai-confound (design panel 7 lug, vedi README):

RUNG (harness crescente, temp=0):
  h0  naked      — NEUTRAL_SYSTEM, NIENTE tool, 1 turno. Il modello risolve in-testa. FLOOR.
  h1  tool-1     — 1 giro di tool poi risposta forzata (tool tolti al turno finale).
  h2  loop       — l'agentic loop pieno (run_agent). Il valore del LOOP = memoria-scratch esterna.
  h4  verify     — h2, poi un passo indipendente 'verifica e correggi' a contesto fresco.
  p   prompt-twin— h2 col wording 'controlla il tuo lavoro' già nel prompt (0 struttura extra).

CONTROLLI COMPUTE (voto self-consistency a budget-token pari, temp>0, NESSUNA struttura):
  nullA  voto su h0  — gemello flat-compute per h1/h2.  h2 = cleverness solo se sta SOPRA nullA.
  nullB  voto su h2  — gemello flat-compute per h4.      h4 = architettura solo se batte nullB E p.

LETTURA: ogni rung è un PUNTO su (token totali, accuratezza). Un rung 'aggiunge intelligenza'
solo se sta SOPRA la sua curva-null a token pari con CI Wilson non sovrapposte. p neutralizza
'il guadagno è solo un prompt migliore'.

Uso:
  python3 ladder.py --canary                       # 1 istanza, h0/h1/h2, locale → smoke test
  python3 ladder.py --pilot --n 20                 # h0-only, calibra L (mira H0 ~30-40%)
  python3 ladder.py --n 8 --rungs h0,h1,h2,nullA   # curva meno-confusa, locale
  python3 ladder.py --n 8 --models local,fable     # + colonna frontiera (serve ANTHROPIC_API_KEY)
  python3 ladder.py --n 8 --rungs h0,h1,h2,h4,p,nullA,nullB --K 5   # scala piena
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import urllib.request
from datetime import datetime

from bench import (
    wilson,
    _RISPOSTA as _ANS,
)  # regex UNICA bench/ladder (stessa semantica di score)
from harness import (
    NEUTRAL_SYSTEM,
    SYSTEM,
    AnthropicProvider,
    OpenAIProvider,
    _TOOL_FN,
    run_agent,
)
import tasks_ledger as task

HERE = pathlib.Path(__file__).parent


def extract(final: str) -> float | None:
    ms = _ANS.findall(final or "")
    if not ms:
        return None
    try:
        return float(ms[-1])  # l'ULTIMA riga RISPOSTA (un verify può riscriverla)
    except ValueError:
        return None


def _exec(tc: dict) -> str:
    fn = _TOOL_FN.get(tc["name"])
    try:
        return fn(tc["args"]) if fn else f"ERROR: tool {tc['name']} sconosciuto"
    except Exception as e:
        return f"ERROR: {e}"


# ───────────────────────── Rung (ognuno: (ctx, prompt) -> result) ─────────────
# result = {"final", "tok", "calls", "refused", "trunc", "tok_in", "ref_partial"}.
# ctx = {"cold", "hot", "K"}. NOTA: il compute-matching dei null resta su tok (=output);
# tok_in è tracciato per trasparenza (il context cresce col loop) ma NON entra nel matching.
def _res(final, tok, calls, refused=False, trunc=False, tok_in=0, ref_partial=0):
    return {
        "final": final,
        "tok": tok,
        "calls": calls,
        "refused": refused,
        "trunc": trunc,
        "tok_in": tok_in,
        "ref_partial": ref_partial,  # rifiuti PARZIALI dentro un voto (0 nei rung singoli)
    }


def rung_h0(ctx, prompt):
    r = run_agent(
        ctx["cold"], prompt, max_turns=1, system=NEUTRAL_SYSTEM, tools_enabled=False
    )
    return _res(
        r["final"],
        r["tokens_out"],
        r["turns"],
        r["finish_reason"] == "refusal",
        r["truncated"],
        r["tokens_in"],
    )


def rung_h1(ctx, prompt):
    prov = ctx["cold"]
    transcript = [{"role": "user", "text": prompt}]
    tok = calls = tok_in = 0
    trunc = False
    out = prov.call(SYSTEM, transcript, True)  # turno 1: con tool
    calls += 1
    tok += out["usage"]["out"]
    tok_in += out["usage"]["in"]
    trunc = trunc or out["truncated"]
    if out["finish_reason"] == "refusal":
        return _res("", tok, calls, True, trunc, tok_in)
    transcript.append(
        {"role": "assistant", "text": out["text"], "tool_calls": out["tool_calls"]}
    )
    if out["tool_calls"]:
        for tc in out["tool_calls"]:
            transcript.append(
                {"role": "tool", "id": tc["id"], "output": str(_exec(tc))}
            )
        out = prov.call(
            SYSTEM, transcript, False
        )  # turno 2: SENZA tool → risposta forzata
        calls += 1
        tok += out["usage"]["out"]
        tok_in += out["usage"]["in"]
        trunc = trunc or out["truncated"]
        if out["finish_reason"] == "refusal":
            return _res("", tok, calls, True, trunc, tok_in)
    return _res(out["text"], tok, calls, False, trunc, tok_in)


def rung_h2(ctx, prompt):
    r = run_agent(ctx["cold"], prompt, max_turns=6, system=SYSTEM, tools_enabled=True)
    return _res(
        r["final"],
        r["tokens_out"],
        r["turns"],
        r["finish_reason"] == "refusal",
        r["truncated"],
        r["tokens_in"],
    )


def rung_h4(ctx, prompt):
    base = rung_h2(ctx, prompt)
    a = extract(base["final"])
    hint = (
        f"Una prima soluzione ha dato come saldo finale {a}. " if a is not None else ""
    )
    vprompt = (
        f"{prompt}\n\n{hint}Verifica ricalcolando da capo, passo per passo. "
        "Se il risultato precedente è sbagliato, correggilo. Concludi con: RISPOSTA: <numero>."
    )
    v = run_agent(ctx["cold"], vprompt, max_turns=6, system=SYSTEM, tools_enabled=True)
    # SCELTA DICHIARATA (conservativa): se la base ha rifiutato, il trial conta refused anche
    # se il passo verify risponde — un h4 che "ripesca" un refusal misurerebbe il guardrail,
    # non la verifica. Confronto onesto con nullB: usa --K 2 (h4 ≈ 2×h2 di compute). # VERIFIED
    return _res(
        v["final"],
        base["tok"] + v["tokens_out"],
        base["calls"] + v["turns"],
        base["refused"] or v["finish_reason"] == "refusal",
        base["trunc"] or v["truncated"],
        base["tok_in"] + v["tokens_in"],
    )


def rung_p(ctx, prompt):
    aug = (
        prompt
        + "\n(Risolvi, poi ricontrolla il tuo calcolo passo per passo, poi dai la risposta.)"
    )
    r = run_agent(ctx["cold"], aug, max_turns=6, system=SYSTEM, tools_enabled=True)
    return _res(
        r["final"],
        r["tokens_out"],
        r["turns"],
        r["finish_reason"] == "refusal",
        r["truncated"],
        r["tokens_in"],
    )


def _vote(base_fn, ctx, prompt):
    """Voto self-consistency a maggioranza su base_fn, K campioni col provider HOT (temp>0).
    Ties = SBAGLIATO (conservativo). Somma token/chiamate → confronto compute onesto.
    ref_partial: rifiuti DENTRO il voto quando non sono K/K — prima erano invisibili (il null
    votava su 4/5 campioni senza traccia; con Fable ~20% refusal stocastico è sistematico)."""
    hot_ctx = {"cold": ctx["hot"], "hot": ctx["hot"], "K": ctx["K"]}
    votes, tok, calls, refused, trunc, tok_in = [], 0, 0, 0, False, 0
    for _ in range(ctx["K"]):
        r = base_fn(hot_ctx, prompt)
        tok += r["tok"]
        calls += r["calls"]
        tok_in += r["tok_in"]
        trunc = trunc or r["trunc"]
        if r["refused"]:
            refused += 1
            continue
        a = extract(r["final"])
        if a is not None:
            votes.append(a)
    if not votes:
        return _res(
            "", tok, calls, refused == ctx["K"], trunc, tok_in, ref_partial=refused
        )
    cnt = collections.Counter(votes)
    top, n = cnt.most_common(1)[0]
    tie = sum(1 for _, c in cnt.items() if c == n) > 1
    final = (
        "" if tie else f"RISPOSTA: {top:g}"
    )  # tie → nessuna risposta → conta sbagliato
    return _res(final, tok, calls, False, trunc, tok_in, ref_partial=refused)


def rung_nullA(ctx, prompt):
    return _vote(rung_h0, ctx, prompt)


def rung_nullB(ctx, prompt):
    return _vote(rung_h2, ctx, prompt)


RUNGS = {
    "h0": rung_h0,
    "h1": rung_h1,
    "h2": rung_h2,
    "h4": rung_h4,
    "p": rung_p,
    "nullA": rung_nullA,
    "nullB": rung_nullB,
}


# ───────────────────────── Providers per modello ─────────────────────────
def make_providers(
    model: str,
    max_tokens: int = 8192,
    local_model: str = "qwen/qwen3-14b",
    think: bool = True,
    timeout: int = 600,
):
    """Ritorna (cold temp=0, hot temp>0). max_tokens ALTO (8192): il 32B pensa molto sul ledger
    duro e a 4096 troncava PRIMA di chiamare un tool o emettere RISPOSTA (canary 7 lug) → confound
    di truncation. Frontiera: temperature RIMOSSA su Fable/Opus → hot≈cold (diversità dal thinking
    non-deterministico, non da temp). think/timeout: assi dell'harness, dichiarati nel manifest;
    think=False vale SOLO per il locale (AnthropicProvider lo rifiuta, asimmetria dichiarata). # VERIFIED"""
    if model == "local":
        return (
            OpenAIProvider(
                model=local_model,
                temperature=0.0,
                max_tokens=max_tokens,
                think=think,
                timeout=timeout,
            ),
            OpenAIProvider(
                model=local_model,
                temperature=0.7,
                max_tokens=max_tokens,
                think=think,
                timeout=timeout,
            ),
        )
    if model == "fable":
        return (
            AnthropicProvider(
                model="claude-fable-5", max_tokens=max_tokens, timeout=timeout
            ),
            AnthropicProvider(
                model="claude-fable-5", max_tokens=max_tokens, timeout=timeout
            ),
        )
    if model == "opus":
        return (
            AnthropicProvider(
                model="claude-opus-4-8", max_tokens=max_tokens, timeout=timeout
            ),
            AnthropicProvider(
                model="claude-opus-4-8", max_tokens=max_tokens, timeout=timeout
            ),
        )
    raise ValueError(f"modello sconosciuto: {model} (local|fable|opus)")


def preflight(models, local_model="qwen/qwen3-14b", think=True):
    """Fallisce SUBITO (non dopo 30 min di timeout×retry) se un modello non è raggiungibile.
    Il canary 7 lug è caduto per server locale su-ma-modello-scarico: questo lo becca in 5s."""
    problems = []
    if "local" in models:
        url = "http://localhost:1234"
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
                ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
        except Exception as e:
            return [
                f"server locale localhost:1234 non risponde ({type(e).__name__}). "
                "Avvia il server HTTP: `lms server start`."
            ]
        if local_model not in ids:
            problems.append(
                f"modello '{local_model}' non elencato (trovati: {ids or 'nessuno'}). "
                f"Caricalo: `lms load {local_model}`."
            )
        else:
            # readiness REALE: /v1/models elenca i modelli scaricabili, NON garantisce che sia
            # caricato in memoria e servito. Una completion minuscola deve tornare entro 60s.
            try:
                body = json.dumps(
                    {
                        "model": local_model,
                        "messages": [{"role": "user", "content": "2+2?"}],
                        "max_tokens": 16,
                    }
                ).encode()
                req = urllib.request.Request(
                    f"{url}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    json.loads(r.read())["choices"][0]["message"]
            except Exception as e:
                problems.append(
                    f"server risponde ma la completion di prova fallisce/hang ({type(e).__name__}). "
                    f"Modello caricato in memoria? `lms load {local_model}` + server avviato? `lms server start`."
                )
    if ("fable" in models or "opus" in models) and not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        problems.append(
            "ANTHROPIC_API_KEY mancante nell'ambiente ma richiesta per fable/opus."
        )
    if not think and ("fable" in models or "opus" in models):
        problems.append(
            "thinking=off richiesto con fable/opus: non supportato (thinking sempre-on "
            "lato Anthropic). Regola anti-confound: mai mescolare regimi in un run — "
            "lancia il locale think=off e la frontiera think=on come DUE run/manifest."
        )
    return problems


def serving_info() -> dict:
    """Fotografa il deployment di serving locale per il manifest (best-effort).
    Endpoint REST LM Studio /api/v0/models: probato 10 lug, ritorna quantization/arch/
    context del modello caricato. Se irraggiungibile → {} (mai bloccare il run). # VERIFIED"""
    try:
        with urllib.request.urlopen(
            "http://localhost:1234/api/v0/models", timeout=5
        ) as r:
            data = json.loads(r.read()).get("data", [])
        return {
            m["id"]: {
                "quant": m.get("quantization"),
                "arch": m.get("arch"),
                "format": m.get("compatibility_type"),
                "ctx_loaded": m.get("loaded_context_length"),
            }
            for m in data
            if m.get("state") == "loaded"
        }
    except Exception:
        return {}


# ───────────────────────── Driver ─────────────────────────
def run_experiment(
    models,
    rung_names,
    seeds,
    L,
    K,
    logf,
    max_tokens=8192,
    local_model="qwen/qwen3-14b",
    think=True,
    timeout=600,
):
    results = {}  # (model, rung) -> list of trial dicts
    for model in models:
        cold, hot = make_providers(
            model,
            max_tokens=max_tokens,
            local_model=local_model,
            think=think,
            timeout=timeout,
        )
        ctx = {"cold": cold, "hot": hot, "K": K}
        print(f"\n════════ MODELLO: {model} ════════")
        for rname in rung_names:
            fn = RUNGS[rname]
            trials = []
            for seed in seeds:
                prompt, gold, meta = task.generate(seed, L)
                try:
                    r = fn(ctx, prompt)
                except Exception as e:  # rete/API già ritentata → ERROR (non FAIL)
                    print(f"  {rname:6} seed={seed} ERROR {e}")
                    # l'errore ENTRA nei trial (marcato) → _agg lo conta e lo stampa.
                    # Prima veniva solo loggato e sparire dal riepilogo = 'acc 0/0' muto
                    # (run 20260707T150854: 20/20 timeout invisibili). Empty = failure. # VERIFIED
                    trials.append({"error": str(e)})
                    logf.write(
                        json.dumps(
                            {
                                "model": model,
                                "rung": rname,
                                "seed": seed,
                                "error": str(e),
                            }
                        )
                        + "\n"
                    )
                    continue
                ans = extract(r["final"])
                correct = (not r["refused"]) and task.verify(ans, gold)
                trials.append({**r, "correct": correct, "gold": gold, "ans": ans})
                logf.write(
                    json.dumps(
                        {
                            "model": model,
                            "rung": rname,
                            "seed": seed,
                            "gold": gold,
                            "ans": ans,
                            "correct": correct,
                            "refused": r["refused"],
                            "trunc": r["trunc"],
                            "tok": r["tok"],
                            "tok_in": r["tok_in"],
                            "ref_partial": r["ref_partial"],
                            "calls": r["calls"],
                        }
                    )
                    + "\n"
                )
            results[(model, rname)] = trials
            _print_rung(model, rname, trials)
    return results


def _agg(trials):
    errs = [t for t in trials if t.get("error")]
    done = [t for t in trials if not t.get("error")]
    ok = [t for t in done if not t["refused"]]
    n = len(ok)
    k = sum(t["correct"] for t in ok)
    ref = sum(t["refused"] for t in done)
    ref_p = sum(t.get("ref_partial", 0) for t in done)
    trunc = sum(t.get("trunc") for t in done)
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    tok = sum(t["tok"] for t in done) / max(1, len(done))
    calls = sum(t["calls"] for t in done) / max(1, len(done))
    return {
        "n": n,
        "k": k,
        "acc": k / n if n else 0.0,
        "lo": lo,
        "hi": hi,
        "ref": ref,
        "ref_partial": ref_p,
        "trunc": trunc,
        "err": len(errs),
        "N": len(trials),
        "tok": tok,
        "calls": calls,
    }


def _print_rung(model, rname, trials):
    a = _agg(trials)
    ref = f" refused={a['ref']}" if a["ref"] else ""
    refp = f" ref_partial={a['ref_partial']}" if a["ref_partial"] else ""
    # ⚠️ error = trial MAI completato (timeout/rete): se >0 il rung è sotto-campionato, MAI 'acc 0/0' muto
    er = f"  ⚠️ERR {a['err']}/{a['N']}" if a["err"] else ""
    # ⚠️ truncation = confound: se >0, il numero NON è affidabile (alza --max-tokens)
    tr = f"  ⚠️TRUNC {a['trunc']}/{a['N']}" if a["trunc"] else ""
    print(
        f"  {rname:6} acc {a['k']}/{a['n']} = {a['acc']:.0%} [CI {a['lo']:.0%}-{a['hi']:.0%}]"
        f"  ~{a['tok']:.0f} tok  {a['calls']:.1f} calls{ref}{refp}{er}{tr}"
    )


def _verdict(name_hi, hi, name_lo, lo):
    """hi 'batte' lo se acc più alta E CI non sovrapposte (Wilson)."""
    if not hi or not lo:
        return
    sep = hi["lo"] > lo["hi"]
    rel = "SOPRA ✓ (cleverness)" if sep else "dentro le CI ✗ (non risolto / = compute)"
    print(
        f"    {name_hi} vs {name_lo}: {hi['acc']:.0%} @~{hi['tok']:.0f}tok  vs  "
        f"{lo['acc']:.0%} @~{lo['tok']:.0f}tok  →  {rel}"
    )
    # onestà sul compute-matching: se un lato spende >1.5× i token dell'altro il verdetto
    # '=compute' non è affidabile (es. nullB con K=5 ≈ 2.5× h4 → tara con --K 2). # VERIFIED
    big, small = max(hi["tok"], lo["tok"]), max(1.0, min(hi["tok"], lo["tok"]))
    if big / small > 1.5:
        print(
            f"    ⚠️ compute-mismatch: {big / small:.1f}× di divario token tra i due lati — "
            "verdetto compute-matched non affidabile (tara K/il budget prima di leggerlo)"
        )


def print_plane(models, rung_names, results):
    print("\n════════ PIANO (token, accuratezza) + verdetti pre-registrati ════════")
    for model in models:
        ag = {r: _agg(results[(model, r)]) for r in rung_names if (model, r) in results}
        print(f"\n── {model} ──")
        for r in rung_names:
            if r in ag:
                x = ag[r]
                tr = f"  ⚠️TRUNC {x['trunc']}/{x['N']}" if x["trunc"] else ""
                print(
                    f"  {r:6} {x['acc']:.0%} [CI {x['lo']:.0%}-{x['hi']:.0%}] @ ~{x['tok']:.0f} tok, {x['calls']:.1f} calls{tr}"
                )
        # giunzioni pre-registrate (solo se i rung ci sono)
        if "h1" in ag and "h2" in ag:
            _verdict("h2 (loop)", ag["h2"], "h1 (tool-1)", ag["h1"])
        if "h2" in ag and "nullA" in ag:
            _verdict("h2 (loop)", ag["h2"], "nullA (voto-h0)", ag["nullA"])
        if "h4" in ag and "nullB" in ag:
            _verdict("h4 (verify)", ag["h4"], "nullB (voto-h2)", ag["nullB"])
        if "h4" in ag and "p" in ag:
            _verdict("h4 (verify)", ag["h4"], "p (prompt-twin)", ag["p"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="local", help="local,fable,opus")
    ap.add_argument("--rungs", default="h0,h1,h2,nullA")
    ap.add_argument("--n", type=int, default=8, help="numero istanze (seed)")
    ap.add_argument("--L", type=int, default=12, help="difficoltà: operazioni per task")
    ap.add_argument("--K", type=int, default=5, help="campioni per voto null")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="budget output (32B pensa molto → 4096 tronca)",
    )
    ap.add_argument(
        "--seed0", type=int, default=1000, help="primo seed (istanze = seed0..seed0+n)"
    )
    ap.add_argument(
        "--local-model",
        default="qwen/qwen3-14b",
        help="id del modello locale (prima hardcoded)",
    )
    ap.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="on",
        help="asse harness: off appende /no_think (solo locale). MAI mescolare regimi in un "
        "run; on-vs-off = due run/manifest. Default on = comparabile coi run 6-8 lug.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="timeout per singola chiamata (s); i timeout NON vengono ritentati",
    )
    ap.add_argument(
        "--canary", action="store_true", help="1 istanza, h0/h1/h2, solo locale"
    )
    ap.add_argument(
        "--pilot", action="store_true", help="solo h0, per calibrare L (mira ~30-40%)"
    )
    args = ap.parse_args()

    think = args.thinking == "on"
    if args.canary:
        if args.models != "local" or args.rungs != "h0,h1,h2,nullA":
            print(
                f"⚠️ --canary SOVRASCRIVE --models/--rungs → local, h0/h1/h2 (ignorati: "
                f"--models {args.models} --rungs {args.rungs})"
            )
        models, rung_names, n = ["local"], ["h0", "h1", "h2"], 1
    elif args.pilot:
        models, rung_names, n = args.models.split(","), ["h0"], args.n
    else:
        models = args.models.split(",")
        rung_names = args.rungs.split(",")
        n = args.n
    seeds = list(range(args.seed0, args.seed0 + n))

    probs = preflight(models, local_model=args.local_model, think=think)
    if probs:
        print("❌ PRE-FLIGHT fallito — niente run:")
        for p in probs:
            print(f"   • {p}")
        raise SystemExit(1)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id,
        "models": models,
        "rungs": rung_names,
        "n": n,
        "L": args.L,
        "K": args.K,
        "seed0": args.seed0,
        "max_tokens": args.max_tokens,
        "local_model": args.local_model,
        "thinking": args.thinking,
        "timeout_s": args.timeout,
        "serving": serving_info(),  # quant/arch/ctx del deployment locale (prima: invisibile)
        "task": "ledger",
        "note": "ladder: harness fisso-per-rung, modello per-colonna; null = voto compute-matched",
    }
    print(f"LADDER · {manifest}")
    # buffering=1 (line-buffered): un run killed a metà NON perde i trial completati —
    # il run 20260710T163022 è morto con 24 trial fatti e 0 righe su disco. # VERIFIED
    with open(HERE / "results" / f"{run_id}_ladder.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        results = run_experiment(
            models,
            rung_names,
            seeds,
            args.L,
            args.K,
            logf,
            max_tokens=args.max_tokens,
            local_model=args.local_model,
            think=think,
            timeout=args.timeout,
        )
        print_plane(models, rung_names, results)
    print(f"\nlog: results/{run_id}_ladder.jsonl")
