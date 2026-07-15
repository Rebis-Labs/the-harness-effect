# harness-lab — isolare l'harness per benchmarkare i modelli

> Esperimento: **capacità = modello × harness.** Per misurare il *modello*, tieni l'harness IDENTICO e scambia solo chi risponde. Nato dalla domanda "l'1,6% è vero?" (sì come righe-di-codice, no come 'il modello è banale' — vedi memory `reference_local_ai_stack` + chat 6 lug).

## Cosa c'è
- `harness.py` — l'harness model-agnostico: 3 tool deterministici (`kv_get`, `calc`, `word_count`), il loop agentico (call → esegui tool → rifeeda → ripeti), e 2 provider (`OpenAIProvider` per Qwen locale/OpenAI-compat, `AnthropicProvider` per Fable/Opus). **Il loop è identico**; cambia solo l'adapter di formato.
- `bench.py` — 5 task che *richiedono* il loop (tool multipli in sequenza) + 1 anti-over-tool. Scorer deterministico (match numerico).

## Come si gira
```bash
cd harness-lab
python3 bench.py                                 # solo Qwen locale (localhost:1234)
python3 bench.py --anthropic claude-fable-5      # + Fable (serve ANTHROPIC_API_KEY)
python3 bench.py --anthropic claude-opus-4-8     # o Opus
```

## Il punto scientifico (perché è un confronto onesto)
Ciò che tengo COSTANTE (l'harness): stesso system prompt, stessi tool + schema, stesso loop, stesso `max_turns`, stesso `temperature=0`, stessi task, stesso scorer. Cambia SOLO il modello sotto → la differenza di score è attribuibile al **modello**, non all'impalcatura.

## Confound onesti (dove il confronto NON è pulito — leggili)
1. **Formato tool-call nativo diverso.** Anthropic usa blocchi `tool_use`/`tool_result`; OpenAI usa `tool_calls`/`function`. L'harness normalizza a un'interfaccia comune ma ogni modello usa il formato su cui è stato addestrato — questo è *equo* (ognuno "a casa sua"), ma non identico byte-per-byte.
2. **Thinking mode.** Qwen3 e Fable ragionano entrambi prima di rispondere → i token/latency non sono comparabili; lo score sì. Non forzo `/no_think` per non svantaggiare il locale.
3. **Quantizzazione.** Il locale è Q4 (18GB) — non il modello full-precision. Stai benchmarkando *quella build*, non "Qwen3-32B ideale".
4. **Task minimali.** 5 task tool-use ≠ SWE-Bench. Servono a *dimostrare il metodo*, non a produrre un ranking pubblicabile. Per numeri seri: aumenta i task, usa un dataset esterno (es. una fetta di GAIA/SWE-bench-lite) e ripeti N volte.
5. **Sampling non appaiabile con Fable.** Il locale gira a `temperature=0, top_p=1` pinnati; **Fable 5 rifiuta temperature/top_p** (400 — sono rimossi su Opus 4.7+/Fable/Sonnet 5) e gira sempre con **adaptive thinking** a `effort` default. Quindi la determinizzazione non è simmetrica: il locale è ~deterministico, Fable no. Confound inerente al modello, non fixabile dall'harness. (Per questo l'AnthropicProvider manda solo `max_tokens`.)

## Null baseline (anti-over-engineering, per onestà)
Esistono già framework di eval agentici (inspect-ai, OpenAI evals, ecc.). Questo è ~250 righe fatto in casa **apposta**, perché qui *l'harness È l'oggetto di studio*: lo vuoi trasparente e tuo, non una scatola nera. Se un giorno serve un benchmark serio (non pedagogico), usa un framework esistente + un dataset vero — NON far crescere questo in un mostro (cluster #7).

## Risultati — primo giro (6 lug 2026, Qwen3-32B-abliterated Q4, temp=0)
| Harness | Score | Delta |
|---|---|---|
| **v1** (base: legge solo `content`) | **3/5** | — |
| **v2** (+ fallback su `reasoning_content` se content vuoto) | **4/5** | +1 |
- **Modello identico**, cambiata UNA riga di harness → +20 punti. Dimostrazione empirica dell'1,6%: parte della "capacità" osservata È qualità dell'harness.
- `chain_mul` (273): recuperato in v2 — il modello aveva calcolato la risposta (3 tool-call) ma non l'aveva committata nel content (thinking mode). Recupero legittimo.
- `three_sum` (53): fallito anche in v2 — **0 tool-call, nessuna risposta nel reasoning** = limite reale del modello (il fallback NON l'ha gonfiato → test onesto).
- Caveat: leggere il reasoning può in teoria matchare un numero intermedio spurio; qui l'unico recupero è su un target esatto dopo tool-call → quasi certamente reale.

### Curva harness (stesso modello, 6 lug)
| Harness | Score | Nota |
|---|---|---|
| v1 base (solo `content`) | 3/5 | — |
| v2 patch (fallback `reasoning`) | 4/5 | recupera `chain_mul` raschiando il reasoning (fragile) |
| v3 source-fix (prompt forza-tool + contratto `RISPOSTA:`) | 4/5 | stesso score ma pass ROBUSTI (risposta committata, non raschiata) |
- **Lezione**: i guadagni harness non sono monotoni su 5 task; "score uguale" ≠ "harness uguale" (v3 > v2 per robustezza).
- **`three_sum` fallisce in v1/v2/v3**: 0 tool-call anche con prompt che lo impone → limite che NON si fixa via prompt. Prossimo lever = **repair loop** (rileva "risposta senza tool richiesti" → re-inietta e riprova).

## Cross-model: Fable vs Qwen-local (6 lug) — IL risultato che conta
Primo confronto a parità di harness: **Qwen-local (abliterated Q4) 15/15 = 100% · Fable 0/15**. Ma il "0" NON è capacità:
- **Fable ha rifiutato tutte 15 le chiamate** (`REFUSAL 15/15`): HTTP 200, `stop_reason=refusal`, content vuoto, ~7 token, ~3.2s — rifiuto di sicurezza **pre-output**, anche sul primissimo `sum_two` (payload pulito, nessun contesto). Fable non ha *tentato* zero aritmetica.
- **Leggi lo scoreboard come "il guardrail è scattato: no vs sì", NON "aritmetica: passa vs fallisce".** Il locale fa 100% *perché è abliterated* — l'abliteration rimuove chirurgicamente proprio l'asse (il rifiuto) che il benchmark stava di fatto misurando. Così com'era, **il benchmark premiava l'abliteration** e spacciava "ha un guardrail" per "non sa fare il conto".
- **Causa CONFERMATA** (`probe_fable.py` + `probe_fix.py`, N-reps + Wilson): categoria del rifiuto = **`cyber`** ("violative cyber content"). Bisezione fattoriale, user costante:
  - bare `2+2` (no system, no tools) → **end_turn OK** → il modello/key/version-header è sano, non è account né protocollo.
  - Il trigger è l'**AND di due framing** nelle description dei tool: kv_get="legge da un **database** per **chiave**" + calc="**valuta un'espressione** … `**`". La **coppia** kv_get+calc originale → **100% deterministico** (`word_count` innocente). Neutralizzare *una qualsiasi* delle due → **crolla a ~20%**. Il co-occorrere "lookup in DB per chiave" + "eval di espressioni" legge come tooling da injection.
  - **NON** era il tono del prompt (il v3 imperativo era un red herring: addolcirlo alzò 80→100). Reword applicato in `harness.py` v5.
  - **Residuo**: anche tutto-neutro resta **~20-40%** (N=5, CI larghe/sovrapposte) → oltre al framing c'è un **residuo stocastico** (thinking sempre-on). Perciò il refusal-handling è **necessario, non opzionale**.
- **Fix impalcatura applicato** (`bench.py`): il refusal è un **terzo esito**. La capacità si misura su `answered = runs_ok − refusal` (`ANSWER-ACC`), col refusal-rate a parte. Un refusal non è più `strict=0`.
- **Distingui i fix**: il *server-side fallback* Anthropic (`betas:[server-side-fallback-2026-06-01]`+`fallbacks:[opus-4-8]`) è giusto in **produzione** ma **sbagliato per un benchmark** — ri-serve via Opus → misureresti Opus, non Fable. Retry client-side → taggalo "recovered" e riportalo separato.
- **Numero finale misurato** (wording v5, `bench.py --anthropic claude-fable-5 --no-local -n 8`, run `20260707T020741`, md5 `9d9f0912`): **ANSWER-ACC 33/33 = 100%** [CI95 90-100%] · **REFUSAL 7/40 = 17.5%**. Fable, quando ingaggia (82.5% delle volte), azzecca ogni operazione. `chain_mul` è l'hotspot residuo (4/8).
- **La dimostrazione più pulita della tesi**: stesso modello, stesso task, cambiata SOLO la *descrizione testuale* dei tool (puro harness, zero codice-modello) → Fable passa da **0% usabile (100% refusal) a 100% corretto sull'82.5% che ingaggia**. L'harness ha *creato* il gap di capacità. `capacità = modello × harness`, non uno slogan: due numeri misurati (0% → 100%) a modello identico.

## Upgrade di rigore (6 lug, post-review avversaria)
Il benchmark ora fa sul serio (le 3 mosse a più alto ROI della review):
- **N ripetizioni per task** + **Wilson CI95** → un delta dentro i CI non è reale (con 5 task, 3/3 → CI [44%,100%]: N piccolo non dimostra nulla).
- **Scorer STRICT** sulla riga `RISPOSTA: <n>` con confronto numerico → niente falsi positivi da numeri intermedi (530≠53). `lenient` (substring) tenuto come metrica separata.
- **ERROR ≠ FAIL** → un 429/timeout non conta come −1 del modello; retry+backoff esponenziale in `_post`.
- **finish_reason letto** + `max_tokens` alzato 1024→4096 → misura la **truncation-rate**. ✅ **VERDETTO (run 20260706T140929): il "+20 di v1→v2" ERA un artefatto di truncation.** A 4096: `trunc 0/12`, `recov 0/12` (il reasoning-fallback NON scatta mai), e `chain_mul` che a 1024 usciva vuoto ora fa 3/3 strict pulito. Cioè: a 1024 il thinking veniva **troncato dall'harness**, non "il modello non consegnava". La review avversaria aveva ragione, il mio claim iniziale era sbagliato sul meccanismo. (Resta vero "l'harness conta" — anzi di più: `max_tokens` È un parametro dell'harness che penalizza selettivamente i modelli che pensano.)
- **timeout 240→600s**: `three_sum` (il task più duro) a 4096 genera un thinking lunghissimo sul 32B lento → superava i 240s → veniva classificato ERROR. Alzato a 600s. Insegna un 2° confound-di-harness: un timeout troppo corto misclassifica slow-ma-corretto.
- **Metriche**: tokens_out, latency/task, tool-call efficiency, trajectory-check (`expected_tools`), recovered_from_reasoning.
- **Logging**: `results/<run_id>.jsonl` (transcript+usage+timing) + manifest (model, endpoint, sampling, quant, harness md5) → run auditabili e riproducibili.

**Limiti noti ancora aperti** (deferred, non ROI-primari): task che contengono il piano ("usa kv_get poi calc") = instruction-following non pianificazione; valori KV fissi = memorizzabili → servono task **procedurali** (valori random, chiedono il risultato senza nominare i tool). E la tensione system-prompt ("DEVI usare calc") vs task `no_tool` (trajectory non valutata lì).

## Ladder — misurare quanto l'HARNESS aggiunge (apparato, 7 lug)
L'altra direzione dell'esperimento: **modello FISSO, harness VARIABILE** lungo una scala → quanto contribuisce l'architettura all'intelligenza osservata, isolato dal compute. + asse-modello (locale/Fable/Opus) = heatmap gap-to-frontier. File: `ladder.py` + `tasks_ledger.py` (+ giuntura `tools_enabled`/`system` in `harness.py`).

- **Rung** (harness crescente, temp=0): `h0` naked (no tool, in-testa) → `h1` tool-1-giro → `h2` loop pieno → `h4` verify-and-revise → `p` prompt-twin (wording verify senza struttura).
- **Controllo compute** (la mossa portante): `nullA`/`nullB` = voto self-consistency a **budget-token pari** (temp>0, zero struttura). Ogni rung è un PUNTO su (token, accuratezza); un rung "aggiunge intelligenza" solo se sta **SOPRA la sua curva-null a token pari** con CI Wilson non sovrapposte. `h4` deve battere `nullB` (compute) **E** `p` (prompt) per dirsi "architettura" → neutralizza "il guadagno è solo un prompt migliore".
- **Task-ledger** (procedurale, oracolo esatto gratis): saldo + L operazioni, ≥45% condizionali data-dependent → `calc` rende esatta l'aritmetica ma NON le decisioni di branch → resta headroom sopra il tool-floor (evita il 'calc ceiling'). Non memorizzabile, niente piano nel prompt.
- **Protocollo** (canary-first): `--canary` (1 istanza, mock/locale) → `--pilot --n 20` (calibra `L` per H0 ~30-40%) → `--n 8 --rungs h0,h1,h2,nullA` (curva meno-confusa, 1 notte) → scala piena + `--models local,fable,opus`.
- **Verificato** (7 lug): canary mock passato (contabilità chiamate/token, voto, scoring, piano, verdetti); scoring end-to-end OK. Run reale in attesa del server locale su.
- **Limiti**: misura punti-causati su QUESTO task/modello/quant, mai "N% intelligenza in generale"; il delta-tool è model-dependent BY DESIGN (è il reperto). Frontiera: temp rimossa → il voto-null varia per non-determinatezza intrinseca, non per temp (asimmetria dichiarata).
- **Prossima categoria** = coding-con-test (il punto dolce di Ralph: dove loop+verificatore chiudono il gap col frontier). Richiede una **sandbox di esecuzione codice** (infra + superficie di rischio: si esegue codice generato dal modello) → build separata, flaggata.

## Loop-engineering upgrade (10 lug 2026) — sblocco del locale + repair loop
Design: workflow 3-ricerche + critica avversaria (che ha bocciato KV-quant/speculative/parallel/MLX/update-engine come confound sotto ROI e lo swap modello come churn su fonti hype). Implementato solo il nucleo promosso:
- **Asse `thinking on|off` dichiarato** (`--thinking off` in ladder/bench): appende `/no_think` al system a livello provider (soft-switch nativo Qwen3). Smoke-test 14B: 15.8s/455tok/1124 reasoning-chars → **3.9s/109tok/0 reasoning**, risposta invariata. `chat_template_kwargs enable_thinking:false` NON funziona sull'endpoint OpenAI della versione installata (probato → non usato). Anthropic = sempre-on: preflight FALLISCE se chiedi off con fable/opus (asimmetria dichiarata). **Mai mescolare regimi in un run; on-vs-off = due manifest.** Era il collo di bottiglia: i run ladder 7 lug erano 20/20 `timed out` (thinking a ~12 tok/s > timeout 600s).
- **Timeout per-provider + NO retry sui timeout** (prima: 3×600s persi per trial) + **errori visibili**: gli ERROR entrano in `_agg`/`_print_rung` (`⚠️ERR k/N`), mai più 'acc 0/0' muto.
- **Onestà**: `ref_partial` nei voti null (prima un refusal su 4/5 campioni era invisibile); `tok_in` tracciato (matching resta su tok_out, dichiarato); warning `compute-mismatch` in `_verdict` se un lato spende >1.5× token; regex RISPOSTA unificata bench/ladder; `--local-model` (prima hardcoded); **manifest registra il serving** (quant/arch/ctx via `/api/v0/models`, probato) + thinking + timeout.
- **Repair loop (SOLO bench, `--repair R`)**: risposta senza i `required_tools` → re-inietta `REPAIR_MSG`, max R round, dentro `max_turns`. Refusal MAI riparato (coercizione ≠ correzione). Contatori `strict_clean` / `strict_repaired` **mai sommati**. Il rung h3 ladder resta differito (post prima curva locale).
- **Calibrazione L nel regime think=off** (pilot h0, n=12, 14B Q4): L=6→83% · L=10→100% · L=12→92% · L=16→83% · L=24→58% · L=32→50%. Nota: con `/no_think` il modello scrive comunque i passi nel content (CoT visibile) — h0 resta "in-testa" per definizione (nessun tool), ma non è "senza ragionamento". Il canary L=12 0/1 era un seed sfortunato (n=1 non dimostra nulla — di nuovo).
- **Reperto canary**: a temp=0, L=12 seed=1000, h0/h1/h2 danno TUTTI 168 vs gold 190 con **0 tool-call in h2** — il modello ignora i tool e sbaglia un branch in testa. Stesso failure-mode di `three_sum` → è il caso d'uso del repair loop.

### Prima curva ladder locale COMPLETA (10 lug, 14B Q4, think=off, L=48, n=8)
| rung | acc | CI95 | ~tok | nota |
|---|---|---|---|---|
| h0 naked | **50%** | 22-78% | 1531 | il floor vince |
| h1 tool-1 | 12% | 2-47% | 1407 | ⬇ |
| h2 loop | 12% | 2-47% | 1407 | **1.0 calls: tool MAI usati** |
| nullA voto×5 | 38% | 14-69% | 7597 | 5× compute < h0 greedy |
- **Lettura onesta**: CI larghe (n=8), nessuna separazione conclusiva — ma direzione coerente: **su questo task l'harness coi tool SOTTRAE** (h2 sotto nullA e sotto h0) perché il modello non ingaggia i tool e il system tool-oriented lo degrada vs NEUTRAL. L'harness toglie capacità oltre che aggiungerla: la tesi in negativo. Da blindare con n=20+ su h0/h2. Il rung `tool_choice` coercitivo (bocciato come variante bench) è ora una domanda sperimentale legittima per la ladder.
- **Voto self-consistency**: a temp 0.7 il per-sample degrada più di quanto la maggioranza recuperi (38% @ 5× vs 50% @ 1×).

### Conferma n=20 + rung h3 (10 lug sera) — il verdetto dell'arco
| rung | acc n=20 | CI95 | ~tok | calls | nota |
|---|---|---|---|---|---|
| h0 naked | **45%** | 26-66% | 1479 | 1.0 | floor stabile (era 50% a n=8) |
| h2 loop | 30% | 15-52% | 1751 | 1.1 | tool quasi mai ingaggiati (2/20) |
| h3 repair | 30% | 15-52% | 1779 | 3.0 | **repair scattato 19/20, ingaggio OTTENUTO, acc INVARIATA** |
- Il gap h0-vs-h2 a n=20 si restringe (45 vs 30, CI sovrapposte): la lettura n=8 (50 vs 12) era pessimista — direzione negativa mild, non separata.
- **h3 = h2 spaccato**: forzare l'uso di calc funziona meccanicamente ma non sposta l'accuratezza. Sul ledger gli errori NON sono aritmetici: sono nelle **decisioni di branch** (≥45% condizionali data-dependent BY DESIGN) che calc non tocca. Il task ha funzionato esattamente come progettato: headroom sopra il tool-floor che i tool non possono riempire.
- **La lezione dell'arco (con chain_mul come controesempio)**: il loop engineering paga quando il tool/verificatore COPRE il failure-mode reale (chain_mul: errore aritmetico → errore-istruttivo → 0%→100%); non può salvare ragionamento che il tool non raggiunge (ledger: branch logic → coercizione inutile). **L'harness va accoppiato al failure-mode, non aggiunto in generale.**
- Candidati per alzare il ledger (futuri, non ora): rung h4/p (verify sul branch reasoning), tool nuovo che verifichi le CONDIZIONI (non l'aritmetica), o modello più capace come colonna.

### Piano FINALE a 7 rung (10 lug sera, L=48, think=off, 14B Q4, n=20 salvo nota)
| rung | acc | CI95 | ~tok | calls | verdetto |
|---|---|---|---|---|---|
| **h0** naked | **45%** | 26-66% | 1479 | 1.0 | floor |
| h2 loop | 30% | 15-52% | 1751 | 1.1 | tool ignorati |
| h3 repair | 30% | 15-52% | 1779 | 3.0 | ingaggio forzato, acc invariata |
| h4 verify | 25% | 11-47% | 3346 | 2.1 | 2.3× compute, peggio di tutti |
| **p** prompt-twin | **45%** | 26-66% | **1385** | 1.0 | = h0 al costo MINIMO |
| nullA voto×5 (n=8) | 38% | 14-69% | 7597 | 5.0 | 5× compute < h0 |
| nullB voto×2 | 15% | 5-36% | 3158 | 2.1 | ⚠️ degenere: K=2 + tie=wrong |
- **Verdetti pre-registrati**: h4 vs nullB → sopra ma CI sovrapposte E nullB è handicappato (a K=2 due campioni hot raramente concordano → quasi tutto tie → il "compute-matched" è formale, non informativo; per un null onesto di h4 servirebbe K=2 con tie-break, o K=3). h4 vs p → **p vince nettamente in direzione** (45 vs 25 a 0.4× dei token; CI sovrapposte a n=20, ma h4 perde su ENTRAMBI i corni del test → NON è architettura).
- **La sintesi dell'intera giornata**: su questo (task procedurale branch-heavy × 14B Q4 × think-off), **la scala della struttura è INVERTITA** — una frase nel prompt ("risolvi, poi ricontrolla il tuo calcolo") vale +15pt su h2 e pareggia il naked al costo più basso del piano; ogni struttura aggiunta (tool, coercizione, verify a contesto fresco) rende uguale o peggio a costo maggiore. Insieme a chain_mul (errore-istruttivo 0→100%) il quadro è coerente: **l'harness paga solo dove tocca il failure-mode; altrove è tassa**. "Loop engineering" ≠ "più loop": è scegliere il rung giusto per il fallimento giusto.
- Caveat di validità: n=20, CI larghe, UN task, UN modello, UNA quant, regime think-off. Direzioni nette, separazioni statistiche no. Non generalizzare oltre.

### Bench 14B think=off + errore-istruttivo (10 lug) — chain_mul 0/4 → 4/4
- Run `20260710T175842`: 16/20, con **chain_mul 0/4 deterministico**. Transcript: il modello chiama kv_get×2 + calc **in parallelo nello stesso turno** passando a calc i SIMBOLI non risolti (`(gamma - delta) * 3`) → `ERROR: espressione non consentita` → pur vedendo 100 e 9, INVENTA "sottrazione non consentita" e consegna `RISPOSTA: 0`.
- **Fix = 1 stringa**: errore di calc reso ISTRUTTIVO ("usa solo NUMERI… riprova con i valori numerici"). Run `20260710T180147`: **ANSWER-ACC 20/20 = 100% [CI95 84-100%] · 3.3s/task** (vs 12-170s/task dei run col thinking). Il modello si auto-corregge al giro dopo (`calc('(100 - 9) * 3')` → 273).
- **Lezione (la più pulita finora)**: il messaggio d'errore di un tool È harness — un errore che dice *come* rimediare trasforma 0% in 100% a modello identico. Terza dimostrazione della tesi dopo le tool-description (Fable cyber) e max_tokens (truncation).
- Nota repair loop: su questi task non è MAI scattato (traj 16/16 — i prompt nominano i tool). Il trigger attuale (tool richiesti mancanti) non copre "tool errato + resa": estensione possibile ma differita (anti-mostro).

## Estensioni naturali (se e quando)
- Terzo provider = OpenRouter → confronti Qwen locale vs Qwen-cloud vs Fable a parità di harness (isola anche la quantizzazione).
- Variare l'HARNESS (system prompt, n tool, retry) tenendo il modello fisso → misura il *contributo dell'harness* (l'altra metà dell'1,6%).

## bench_hard — harness-engineering sul 14B (15 lug 2026): 33% → 89% a modello fisso
La prova più forte finora di `capacità = modello × harness`, stavolta nella direzione "harness povero → harness giusto" su task DURI. `bench_hard.py`: 9 task tool-use congelati (catene di 5-9 stadi dipendenti con **joint di estrazione-cifre** che obbligano un round-trip reale per stadio, mult 7×7 cifre, 3 task di recovery da tool-error, distrattori kv, condizionale su parità, iterazione con stato a 8 passi). Modello fisso `qwen/qwen3-14b`, temp=0, N=5, scorer strict invariato.

| Harness | ANSWER-ACC | Note |
|---|---|---|
| v5 (think-on, max_turns=6, max_tokens=4096) | **15/45 = 33.3%** [CI 21-48] | trunc 15/45 (thinking brucia 4096 tok PRIMA della 1ª call), MAX_TURNS 5/45, grind in-testa che scivola |
| v6 (think-OFF + 12 regole + max_turns=32 + continue-nudge) | **40/45 = 88.9%** [CI 76-95] | +55.6pp; Wilson lower 76.5% ≫ baseline point 33.3%; 5× più veloce (35s vs 182s/task) |

- **Il colpo di scena**: il thinking — che sui task facili è il superpotere del 14B (v1-v3 del bench: 100%, macinava in-testa mult 6×3 cifre e LCG a 12 passi) — sui task lunghi è il SABOTATORE: pianifica tutto nel 1° turno, esaurisce il budget token, e quando sopravvive predice in-testa gli argomenti invece di leggere i risultati. `/no_think` + disciplina esplicita nel system prompt ("una operazione dipendente per volta, leggi il risultato, ricopia lo stadio, somma-cifre per trascrizione, parità dall'ultima cifra") = il transcript DIVENTA la working memory. Externalized cognition, misurata.
- **Leve v6** (tutte harness, zero tocchi a task/scorer/modello): asse thinking off · max_turns 6→32 · SYSTEM 12 regole trascrittive · kv_get documenta NOT_FOUND · calc esplicita "solo numeri espliciti" · ERROR div-zero dedicato · CONTINUE-nudge loop (un messaggio senza call né RISPOSTA non chiude il task).
- **Residuo onesto**: `lcg_iter` 0/5 anche in v6 (contatore di 8 iterazioni senza thinking: si perde o sbaglia un passo) → il bench non è saturo, c'è headroom per la prossima leva (candidato: contatore di passi iniettato dal loop).
- Storia indurimento in `bench_hard.py` (v1 100% → v4 33%): il 14B con thinking batcha decine di call PREDICENDO gli argomenti e collassa catene simboliche — i task per essere duri devono forzare la lettura del valore reale (joint di cifre).
