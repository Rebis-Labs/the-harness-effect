# bench_hard2 — piano per de-saturare la fascia alta (16 lug 2026)

## Perché

bench_hard (9 task) è saturo in alto: Sonnet 5 = 100%, Opus 4.8 = 97%, locale v7 = 89%.
A N=5 le differenze osservate sono RUMORE, non classifica:
- double_not_found "locale 5/5 > Opus 4/5" = UN trial di differenza (Wilson 4/5 = CI [36-98%]).
- "Sonnet 100% > Opus 97%" = CI sovrapposti ([92-100] vs [85-99.5]) → indistinguibili.
- Più N sul bench attuale = pagare per confermare il soffitto. Inutile.

Il bench discrimina bene SOTTO (mini-model 0%, locale pre-harness 33%), non SOPRA.

## Design (stessa ricetta /goal dei bench precedenti)

- 15-25 stadi dipendenti per task, numeri più grandi, recovery multipli annidati,
  tool output rumorosi (campi extra da ignorare).
- **Cancello di qualificazione INVERTITO**: un task vale solo se il frontier di
  riferimento scende <80% (come il ≤60% locale de-saturò in basso). Poi CONGELA.
- Colonne: locale 14B · Sonnet 5 · Opus 4.8 (· gpt-5.x se si vuole terza famiglia).
- N=10 sui frontier SOLO qui (CI stretti dove il bench discrimina davvero).

## Budget (lezione run 15 lug: $11.89 reali vs $2.3 stimati)

Transcript a 20+ stadi = input ~quadratico + thinking token fatturati come output
sui modelli Anthropic → stima grezza $15-30 per colonna frontier a N=10.
**Regola vincolante**: canary 1 task × 1 modello → misura il costo REALE via
`/credits` delta → moltiplica → decide Ale col numero in mano. MAI fan-out a stima.

## Quando

Dopo il fix staged_chain sul locale (la leva meccanica tool-output entra
nell'harness PRIMA di congelare bench_hard2, così le colonne nascono comparabili).

## Stato qualificazione (19 lug, post colonna locale + canary)

- **Locale v8: 20/25 = 80%** — interleave_ab 0/5 con errore REALE (estrazione cifre sbagliata
  sotto carico di doppio stato, verificato da transcript), altri 4 task 5/5.
- **Canary frontier**: Sonnet 5 risolve interleave 2/2 ($0.20 misurati / 2 trial →
  colonna piena ≈ $2.5 Sonnet, ≈ $6 Opus).
- **Verdetto parziale**: bench_hard2 v1 misura il gap locale↔frontiera (via interleave) ma
  quasi certamente NON discrimina frontier↔frontier (proiezione: tutti ~100%). Prima di
  spendere su colonne frontier piene serve un'altra asse di difficoltà: 3 catene interlacciate
  e/o BACK-REFERENCE a stadi precedenti ("stadio 9: somma dello stadio 3 e dello stadio 6") —
  rompe la stampella dell'ultimo echo, forza memoria ad accesso casuale sul transcript.

## Finding harness v8 (candidato guard v9 — NON applicato)

L'echo contract fa copiare al modello l'echo dentro la riga RISPOSTA sui finali a
SOTTRAZIONE ("RISPOSTA: 21554 - 15325 = 6229", 10/10 deterministico) → strict fail con
sostanza giusta. Su bench_hard non emerge (nessun finale a sottrazione). Mitigazione
task-side adottata: mai sottrazione come ultimo stadio. Fix harness candidato: loop-guard
che su "RISPOSTA:.*=" chiede di riscrivere solo il numero (stessa classe di REPAIR/CONTINUE).
Applicarlo = v9 → ri-validare bench_hard 45/45 prima di ogni confronto.
