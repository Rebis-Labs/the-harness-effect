# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cos'è

Esperimento: **capacità = modello × harness**. Due direzioni di misura: (a) harness FISSO, modello variabile → benchmark del modello; (b) modello FISSO, harness variabile → contributo dell'harness (`ladder.py`, `bench_hard.py`). Python stdlib puro (urllib, ast, argparse) — **zero dipendenze, zero venv, zero test suite**. È ~250 righe fatte in casa apposta: l'harness È l'oggetto di studio. NON farlo crescere in un framework (anti-mostro, vedi README "Null baseline").

Il **README.it.md** è il log sperimentale completo (risultati, confound, lezioni) — leggerlo prima di toccare qualsiasi cosa; `README.md` è la versione pubblica EN (repo pubblicato come `Rebis-Labs/the-harness-effect`). Ogni run vincente citato ha run_id + md5 verificabili in `results/`.

## Comandi

Prerequisiti: modello locale servito da LM Studio su `localhost:1234` (default `qwen/qwen3-14b`); `ANTHROPIC_API_KEY` per Fable/Opus diretti; `OPENROUTER_API_KEY` (env o `.env` nel repo, gitignored) per `bench_frontier.py`.

```bash
python3 bench.py                                  # bench facile (5 task), locale, N=3
python3 bench.py --anthropic claude-fable-5 --no-local -n 8

python3 bench_hard.py -n 5 --thinking off         # bench duro CONGELATO (9 task), locale
python3 bench_hard.py --only lcg_iter -n 1        # canary/debug singolo task

python3 bench_hard2.py --only mega_chain_18 -n 2 --thinking off   # canary
python3 bench_hard2.py -n 5 --thinking off        # colonna locale (NON congelato)

python3 bench_frontier.py --model anthropic/claude-sonnet-5 --only err_fallback -n 1  # canary
python3 bench_frontier.py --model openai/gpt-4o-mini -n 5         # colonna via OpenRouter

python3 ladder.py --canary                        # smoke test (1 istanza, mock/locale)
python3 ladder.py --n 8 --rungs h0,h1,h2,nullA

python3 diagnose_hard.py results/<run>.jsonl [task_id]   # classifica failure-mode di un run
python3 probe_fable.py --dry                      # probe refusal Fable (payload senza inviare)

ruff check . && ruff format .                     # lint (convenzione workspace)
```

Assi CLI comuni: `-n` (ripetizioni/task), `--thinking on|off`, `--repair R`, `--timeout`, `--max-tokens`, `--only <task_id>`.

## Architettura

- **`harness.py`** — il cuore, tutto il resto lo importa. Contiene: 3 tool deterministici (`kv_get`, `calc`, `word_count`), 2 provider (`OpenAIProvider` per OpenAI-compat/LM Studio, `AnthropicProvider`) che normalizzano su un **transcript neutro** identico (`{"role","text","tool_calls"}`), il loop `run_agent()` (identico per ogni provider — è il punto scientifico), e i prompt (`SYSTEM`, `NEUTRAL_SYSTEM`, `REPAIR_MSG`, `CONTINUE_MSG`). Versione corrente: **v8** (echo contract su calc: ritorna `'expr = risultato'`, non il numero nudo).
- **`bench.py`** — bench facile + le utility condivise: `score_strict` (match numerico sulla riga `RISPOSTA: <n>`), `score_lenient`, `wilson` (CI95). Gli altri bench le importano da qui.
- **`bench_hard.py`** — 9 task duri, importa scorer da bench.py e provider/loop da harness.py; `bench_frontier.py` riusa la sua `run_hard_suite` via OpenRouter; `bench_hard2.py` (risposte auto-calcolate da `_sim()` alla import) importa da bench_hard.
- **`ladder.py` + `tasks_ledger.py`** — scala di rung h0→h4 + controlli compute nullA/nullB; il ledger genera task procedurali con oracolo esatto (`generate(seed,L)` / `verify`).
- **`results/`** — JSONL per run (`<run_id>.jsonl`: manifest con md5 harness/bench + transcript completi). **È TRACCIATO in git** (record sperimentale, vedi .gitignore) — mai cancellare o ignorare.

## Regole non negoziabili (violarle invalida gli esperimenti)

1. **bench_hard è CONGELATO**: `TASKS_HARD` + `score_strict` fissati dal commit `e06b358`, verificati via md5 nel manifest. Non editarli MAI. Si migliora SOLO `harness.py`. `bench_hard2` NON è ancora congelato (cancello: qualifica un task solo se il frontier scende <80% a N≥5).
2. **Le description dei tool sono load-bearing**: il wording "database/chiave" + "valuta espressione" causava 100% refusal cyber su Fable (bisezione in `probe_fix.py`). Non riformulare description o SYSTEM senza ri-validare — il tuning wording sul 14B think-off è a somma ~zero (prompt-budget saturo: aggiungere una regola ne diluisce un'altra). Quando il wording sella, la leva è nella **meccanica del loop** (es. echo contract, errori istruttivi).
3. **I messaggi d'errore dei tool SONO harness**: un errore che dice *come* rimediare ha trasformato 0%→100% (chain_mul). Mantenerli istruttivi.
4. **Refusal = terzo esito**, mai FAIL: la capacità si misura su `ANSWER-ACC` (answered = runs_ok − refusal), refusal-rate a parte. Un refusal non si ripara MAI (coercizione ≠ correzione). Pass `repaired` e `strict_clean` mai sommati.
5. **ERROR ≠ FAIL**: timeout/429 non contano contro il modello. I timeout NON si ritentano (temp=0 → ri-chiamata identica, solo costo).
6. **Mai mescolare regimi thinking on/off in un run** — due manifest separati. `AnthropicProvider` FALLISCE il preflight se chiedi think=off (sempre-on, asimmetria dichiarata). Anthropic: niente temperature/top_p (400 su Fable/Opus 4.7+/Sonnet 5) — solo `max_tokens`.
7. **Canary-first sui run a pagamento**: 1 task × 1 modello → costo REALE misurato → moltiplica → decide Ale col numero in mano. MAI fan-out a stima (lezione 15 lug: $11.89 reali vs $2.3 stimati).
8. **Ogni claim di risultato cita run_id + N + Wilson CI**; con CI sovrapposte la differenza è rumore, non classifica. Modifiche all'harness (v9 candidati ecc.) richiedono ri-validazione di bench_hard 45/45 prima di ogni confronto.

## Convenzioni locali

- Commenti e output in italiano; flag `# VERIFIED` sui fatti provati empiricamente (convenzione workspace, qui usata fitta — ogni riga strana di harness.py ha la sua storia in commento: leggerla prima di "ripulirla").
- Stdlib only anche per l'HTTP (`_post` in harness.py con retry/backoff): niente requests/httpx/sdk.
