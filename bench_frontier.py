#!/usr/bin/env python3
"""Colonne frontiera/cloud su bench_hard CONGELATO, via OpenRouter.

Domanda (15 lug): il 14B locale + harness v7 (40/45) regge il confronto con
(a) i frontier e (b) openai/gpt-4o-mini — il generator di ATANOR? Se il locale
è ≈ gpt-4o-mini sul tool-use loop, un generator proprietario è plausibile.

TASKS_HARD e score_strict restano congelati: questo file è solo un runner.
Stesso harness (SYSTEM v7, loop, tool): cambia SOLO il modello sotto.

Confound dichiarati (README caveat #5 vale anche qui):
- locale girato think-OFF; i cloud girano in modalità nativa (adaptive/none).
- per anthropic/* il sampling è OMESSO (temperature/top_p rifiutati dai
  modelli 4.7+/Sonnet 5 sull'API nativa; OR può droppare, non garantito) →
  solo max_tokens. gpt-* tiene temperature=0 come il locale.

Uso:
  python3 bench_frontier.py --model openai/gpt-4o-mini --only err_fallback -n 1   # canary
  python3 bench_frontier.py --model openai/gpt-4o-mini -n 5
  python3 bench_frontier.py --model anthropic/claude-sonnet-5 -n 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from datetime import datetime

import bench_hard
from bench_hard import _git_rev, run_hard_suite
from harness import OpenAIProvider

HERE = pathlib.Path(__file__).parent
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:  # fallback: la chiave vive in atanor/.env (convenzione workspace)
        env = HERE.parent / "atanor" / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
                break
    if not key:
        raise SystemExit("OPENROUTER_API_KEY mancante (env o atanor/.env)")
    return key


class ORProvider(OpenAIProvider):
    """OpenAIProvider puntato a OpenRouter; sampling omettibile per anthropic/*."""

    def __init__(self, *args, omit_sampling: bool = False, **kw):
        super().__init__(*args, **kw)
        self.omit_sampling = omit_sampling

    def sampling(self) -> dict:
        if self.omit_sampling:
            return {"max_tokens": self.max_tokens}
        return super().sampling()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", required=True, help="ID OpenRouter, es. openai/gpt-4o-mini"
    )
    ap.add_argument("-n", type=int, default=5, help="ripetizioni per task")
    ap.add_argument("--only", default=None, help="filtra un singolo task id (canary)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    omit = args.model.startswith("anthropic/")
    provider = ORProvider(
        model=args.model,
        base_url=OPENROUTER_URL,
        api_key=_load_key(),
        max_tokens=args.max_tokens,
        think=True,  # niente /no_think: modalità nativa del cloud (confound dichiarato)
        timeout=args.timeout,
        omit_sampling=omit,
    )

    if args.only:  # filtro su copia locale — TASKS_HARD nel modulo resta intatto
        bench_hard.TASKS_HARD = [
            t for t in bench_hard.TASKS_HARD if t["id"] == args.only
        ]
        if not bench_hard.TASKS_HARD:
            raise SystemExit(f"task id sconosciuto: {args.only}")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    manifest = {
        "run_id": run_id,
        "bench": "hard-frontier",
        "harness_md5": hashlib.md5((HERE / "harness.py").read_bytes()).hexdigest()[:8],
        "bench_hard_md5": hashlib.md5(
            (HERE / "bench_hard.py").read_bytes()
        ).hexdigest()[:8],
        "git_rev": _git_rev(),
        "provider": "openrouter",
        "model": args.model,
        "n": args.n,
        "tasks": len(bench_hard.TASKS_HARD),
        "only": args.only,
        "max_tokens": args.max_tokens,
        "sampling_omitted": omit,
        "thinking": "native",
        "timeout_s": args.timeout,
        "note": "stesso harness v7 del run locale 40/45; cambia solo il modello (via OR)",
    }

    with open(results_dir / f"{run_id}_frontier.jsonl", "w", buffering=1) as logf:
        logf.write(json.dumps({"MANIFEST": manifest}) + "\n")
        run_hard_suite(f"OR: {args.model}", provider, args.n, logf)

    print(f"\nlog: results/{run_id}_frontier.jsonl  |  manifest: {manifest}")
