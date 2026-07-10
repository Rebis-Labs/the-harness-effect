#!/usr/bin/env python3
"""Model-agnostic agent harness — isola l'harness per scambiare il modello.

Stesso loop, stessi tool, stesso system prompt, stessi sampling params.
Cambia SOLO il provider (quale modello risponde) → un benchmark misura il
MODELLO, non l'harness (che resta costante). Vedi README.md per confound + metriche.

Provider:
  - OpenAIProvider    → endpoint OpenAI-compatibile (LM Studio localhost:1234, Qwen; o OpenAI/OpenRouter)
  - AnthropicProvider → Anthropic Messages API (Fable/Opus; richiede ANTHROPIC_API_KEY)

Ogni provider.call() ritorna:
  {"text", "tool_calls":[{id,name,args}], "finish_reason", "truncated":bool,
   "usage":{"in","out"}, "recovered_from_reasoning":bool}
"""

from __future__ import annotations

import ast
import json
import operator
import os
import time
import urllib.error
import urllib.request

# ───────────────────────── Tools (deterministici, verificabili) ─────────────
KV = {"alpha": 42, "beta": 17, "gamma": 100, "delta": 9}
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("espressione non consentita")


def tool_calc(expr: str) -> str:
    try:
        v = _safe_eval(ast.parse(str(expr), mode="eval").body)
        return str(int(v) if isinstance(v, float) and v.is_integer() else v)
    except Exception:
        # Errore ISTRUTTIVO: è un tool-output → parte dell'harness. Con 'espressione non
        # consentita' il 14B (chain_mul, 10 lug) vedeva i valori giusti ma si arrendeva
        # inventando 'sottrazione non consentita' → RISPOSTA: 0, 4/4 a temp=0. Il loop
        # può auto-correggersi solo se l'errore dice COME. # VERIFIED
        return (
            "ERROR: espressione non valida. Usa solo NUMERI e operatori aritmetici "
            "(es. '(100 - 9) * 3'), non nomi o simboli. Riprova con i valori numerici."
        )


def tool_kv_get(key: str) -> str:
    return str(KV.get(str(key).strip().lower(), "NOT_FOUND"))


def tool_word_count(text: str) -> str:
    return str(len(str(text).split()))


TOOLS = [
    {
        "name": "kv_get",
        # v5: "database/chiave" rimosso — con calc "valuta espressione" faceva scattare il
        # classificatore CYBER di Fable (100% refusal). Vedi probe_fix.py + README. # VERIFIED 6 lug
        "description": "Restituisce il numero associato a un nome. Nomi validi: alpha, beta, gamma, delta.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        "fn": lambda a: tool_kv_get(a["key"]),
    },
    {
        "name": "calc",
        # v5: "valuta un'espressione / **" rimosso (leggeva come code-eval → cyber). L'argomento
        # resta una stringa-espressione ('42 + 17'); il modello la formatta dal nome param `expr`.
        "description": "Calcola il risultato di una somma, sottrazione, moltiplicazione o divisione tra numeri.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
        "fn": lambda a: tool_calc(a["expr"]),
    },
    {
        "name": "word_count",
        "description": "Conta le parole in un testo.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "fn": lambda a: tool_word_count(a["text"]),
    },
]
_TOOL_FN = {t["name"]: t["fn"] for t in TOOLS}


# ───────────────────────── HTTP helper (retry + backoff) ─────────────────────────
def _is_timeout(e: Exception) -> bool:
    # urlopen può segnalare il timeout come TimeoutError diretto O come URLError(reason=timeout)
    return isinstance(e, TimeoutError) or (
        isinstance(e, urllib.error.URLError)
        and isinstance(getattr(e, "reason", None), TimeoutError)
    )


def _post(
    url: str, payload: dict, headers: dict, timeout: int = 600, retries: int = 3
) -> dict:
    # 600s default: un 32B locale con thinking lungo sul task più duro supera i 240s → altrimenti
    # slow-ma-corretto viene classificato ERROR (visto su three_sum, run 20260706T140929). # VERIFIED
    # I TIMEOUT NON SI RITENTANO: per il provider cold (temp=0) la ri-chiamata è ~identica →
    # 3×600s persi per run (i 20/20 'timed out' del run 20260707T150854 hanno pagato il triplo);
    # per il provider hot (temp>0) il retry potrebbe riuscire, ma il no-retry resta giusto per
    # COSTO (il timeout segnala thinking fuori budget, non un guasto transitorio). # VERIFIED
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json", **headers}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2**attempt)  # backoff esponenziale 1,2,4s
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if _is_timeout(e):
                raise  # timeout → subito visibile come ERROR, niente retry
            last = e
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
    raise last  # pragma: no cover


# ───────────────────────── Providers ─────────────────────────
# transcript = lista NEUTRA (identica per ogni provider):
#   {"role":"user","text":...} · {"role":"assistant","text":...,"tool_calls":[...]} · {"role":"tool","id":...,"output":...}


class OpenAIProvider:
    kind = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_reasoning_fallback: bool = True,
        think: bool = True,
        timeout: int = 600,
    ):
        # think=False → appende '/no_think' al system: soft-switch NATIVO di Qwen3, spegne il
        # thinking a livello modello. Smoke-test 10 lug su qwen3-14b: 15.8s/455tok/1124 reasoning
        # chars → 3.9s/109tok/0 reasoning, risposta invariata (190). L'alternativa
        # chat_template_kwargs={'enable_thinking':False} NON ha effetto sull'endpoint OpenAI
        # della versione LM Studio installata (probato: output identico al baseline) → non usata. # VERIFIED
        # È un asse dell'HARNESS: va dichiarato nel manifest, mai mescolato on/off in un run.
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.temperature, self.max_tokens = temperature, max_tokens
        self.use_reasoning_fallback = use_reasoning_fallback
        self.think = think
        self.timeout = timeout

    def sampling(self) -> dict:
        # pinnati per riproducibilità (repeat_penalty è estensione LM Studio, ignorata da OpenAI vero)
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
        }

    def _messages(self, system: str, transcript: list) -> list:
        if not self.think:
            system = (
                system + " /no_think"
            )  # a livello provider → uniforme su TUTTI i rung
        msgs = [{"role": "system", "content": system}]
        for m in transcript:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                a = {"role": "assistant", "content": m.get("text") or ""}
                if m.get("tool_calls"):
                    a["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                msgs.append(a)
            elif m["role"] == "tool":
                msgs.append(
                    {"role": "tool", "tool_call_id": m["id"], "content": m["output"]}
                )
        return msgs

    def call(self, system: str, transcript: list, include_tools: bool = True) -> dict:
        # include_tools=False → payload SENZA 'tools' (rung H0/naked della ladder): il modello
        # deve risolvere in-testa, nessun tool. Backward-compatible (default True). # VERIFIED
        payload = {
            "model": self.model,
            "messages": self._messages(system, transcript),
            **self.sampling(),
        }
        if include_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in TOOLS
            ]
            payload["tool_choice"] = "auto"
        resp = _post(
            f"{self.base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason")
        tcs = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            tcs.append(
                {
                    "id": tc.get("id") or tc["function"]["name"],
                    "name": tc["function"]["name"],
                    "args": args,
                }
            )
        text = msg.get("content") or ""
        recovered = False
        if self.use_reasoning_fallback and not text and not tcs:
            text = msg.get("reasoning_content") or ""
            recovered = bool(text)
        u = resp.get("usage") or {}
        return {
            "text": text,
            "tool_calls": tcs,
            "finish_reason": finish,
            "truncated": finish == "length",
            "usage": {
                "in": u.get("prompt_tokens", 0),
                "out": u.get("completion_tokens", 0),
            },
            "recovered_from_reasoning": recovered,
        }


class AnthropicProvider:
    kind = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_reasoning_fallback: bool = True,
        think: bool = True,
        timeout: int = 600,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.use_reasoning_fallback = (
            use_reasoning_fallback  # non usato lato Anthropic; simmetria d'interfaccia
        )
        # think=False NON è supportato lato Anthropic (Fable/Opus 4.7+: thinking sempre-on,
        # /no_think è un token Qwen senza effetto) → chi orchestra deve FALLIRE il preflight
        # se chiede thinking-off su fable/opus. Asimmetria dichiarata, come per temperature. # VERIFIED
        if not think:
            raise ValueError(
                "thinking-off non disponibile su Anthropic (sempre-on) — "
                "usa solo modelli locali nel regime think=off"
            )
        self.timeout = timeout

    def sampling(self) -> dict:
        # Fable 5 / Opus 4.7+ / Sonnet 5: temperature/top_p/top_k RIMOSSI → 400 se inviati
        # (fonte: skill claude-api). Thinking è sempre-on (si OMETTE il param); la profondità
        # si controlla con output_config.effort (default 'high'). # VERIFIED 6 lug
        return {"max_tokens": self.max_tokens}

    def _messages(self, transcript: list) -> list:
        msgs: list = []
        pending: list = []

        def flush() -> None:
            nonlocal pending
            if pending:
                msgs.append({"role": "user", "content": pending})
                pending = []

        for m in transcript:
            if m["role"] == "user":
                flush()
                msgs.append(
                    {"role": "user", "content": [{"type": "text", "text": m["text"]}]}
                )
            elif m["role"] == "assistant":
                flush()
                content = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls") or []:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["args"],
                        }
                    )
                if content:  # Anthropic rifiuta content vuoto (400); skip sicuro (solo turno terminale)
                    msgs.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                pending.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m["id"],
                        "content": m["output"],
                    }
                )
        flush()
        return msgs

    def call(self, system: str, transcript: list, include_tools: bool = True) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY mancante — Fable/Opus non testabili senza key"
            )
        payload = {
            "model": self.model,
            "system": system,
            "messages": self._messages(transcript),
            **self.sampling(),
        }
        if include_tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in TOOLS
            ]
        resp = _post(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            timeout=self.timeout,
        )
        text, tcs = "", []
        for block in resp.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                tcs.append(
                    {
                        "id": block["id"],
                        "name": block["name"],
                        "args": block.get("input", {}),
                    }
                )
        stop = resp.get("stop_reason")
        u = resp.get("usage") or {}
        return {
            "text": text,
            "tool_calls": tcs,
            "finish_reason": stop,
            "truncated": stop == "max_tokens",
            "usage": {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)},
            "recovered_from_reasoning": False,
        }


# ───────────────────────── Agent loop (IDENTICO per ogni provider) ─────────────
# v5 (de-cyber): la causa REALE del refusal Fable NON era il tono (il v3 imperativo era un red
# herring: addolcirlo alzò il refusal 80→100). Era il framing "database/chiave" (kv_get) + "valuta
# un'espressione/**" (calc) che co-occorrendo leggeva come tooling da injection → category=cyber,
# 100% deterministico. Isolato con probe_fix.py: neutralizzare le description → ~20-40% residuo
# stocastico (il thinking sempre-on). Perciò bench.py TRATTA il refusal come 3° esito. Vedi README.
SYSTEM = (
    "Sei un assistente che risolve piccoli problemi aritmetici. "
    "Hai a disposizione dei tool: kv_get restituisce il numero associato a un nome "
    "(alpha, beta, gamma, delta), calc calcola somme, sottrazioni, moltiplicazioni e divisioni, "
    "word_count conta le parole di un testo. "
    "Usa i tool per ottenere i valori e fare i calcoli con precisione. "
    "Concludi la risposta con una riga: RISPOSTA: <numero>."
)

# NEUTRAL_SYSTEM: nessun tool nominato, nessun piano — per il rung H0 (naked) della ladder,
# dove il modello deve risolvere in-testa. Il contratto RISPOSTA resta identico (format costante).
NEUTRAL_SYSTEM = (
    "Sei un assistente che risolve problemi passo per passo con precisione. "
    "Ragiona con attenzione e concludi la risposta con una riga: RISPOSTA: <numero>."
)

# REPAIR_MSG: iniettato quando il modello risponde SENZA i tool richiesti (repair loop, README:
# lever promesso per three_sum). Wording NEUTRO (lezione v5: niente framing eval/database → cyber).
REPAIR_MSG = (
    "Non hai usato i tool richiesti. Ottieni i valori con i tool disponibili, "
    "rifai il calcolo con precisione e concludi con una riga: RISPOSTA: <numero>."
)


def run_agent(
    provider,
    task_prompt: str,
    max_turns: int = 6,
    system: str = SYSTEM,
    tools_enabled: bool = True,
    required_tools: set | None = None,
    max_repairs: int = 0,
) -> dict:
    """Esegue un task. Ritorna metriche complete. Non solleva su errori-modello (li logga);
    solleva solo su errori di rete/API già ritentati (li conta come ERROR, non FAIL).
    system/tools_enabled: giunture per la ladder (H0 = NEUTRAL_SYSTEM + tools_enabled=False).

    REPAIR LOOP (default OFF, max_repairs=0 = comportamento identico a prima):
    se il modello risponde senza aver usato required_tools → re-inietta REPAIR_MSG e riprova,
    al massimo max_repairs volte, dentro il budget max_turns. Il refusal NON si ripara MAI
    (coercizione ≠ correzione; e ri-provocare un guardrail inquina la misura del refusal-rate).
    I pass riparati sono TAGGATI (repaired/repair_rounds) — mai sommarli ai pass lisci."""
    transcript = [{"role": "user", "text": task_prompt}]
    tool_names: list[str] = []
    tokens_in = tokens_out = 0
    truncated = recovered = False
    last_finish = None
    repair_rounds = 0
    t0 = time.perf_counter()
    for turn in range(max_turns):
        out = provider.call(system, transcript, tools_enabled)
        tokens_in += out["usage"]["in"]
        tokens_out += out["usage"]["out"]
        truncated = truncated or out["truncated"]
        recovered = recovered or out["recovered_from_reasoning"]
        last_finish = out["finish_reason"]
        transcript.append(
            {"role": "assistant", "text": out["text"], "tool_calls": out["tool_calls"]}
        )
        if not out["tool_calls"]:
            needs_repair = (
                max_repairs > 0
                and repair_rounds < max_repairs
                and required_tools is not None
                and not required_tools.issubset(set(tool_names))
                and out["finish_reason"] != "refusal"
                and turn + 1
                < max_turns  # serve almeno un turno per la risposta riparata
            )
            if needs_repair:
                repair_rounds += 1
                transcript.append({"role": "user", "text": REPAIR_MSG})
                continue
            return _result(
                out["text"],
                turn + 1,
                tool_names,
                tokens_in,
                tokens_out,
                truncated,
                recovered,
                t0,
                transcript,
                last_finish,
                repair_rounds,
            )
        for tc in out["tool_calls"]:
            tool_names.append(tc["name"])
            fn = _TOOL_FN.get(tc["name"])
            try:
                result = (
                    fn(tc["args"]) if fn else f"ERROR: tool {tc['name']} sconosciuto"
                )
            except Exception as e:  # arg malformato → il modello può auto-correggersi
                result = f"ERROR: argomenti non validi ({e})"
            transcript.append({"role": "tool", "id": tc["id"], "output": str(result)})
    return _result(
        "[MAX_TURNS]",
        max_turns,
        tool_names,
        tokens_in,
        tokens_out,
        truncated,
        recovered,
        t0,
        transcript,
        last_finish,
        repair_rounds,
    )


def _result(
    final,
    turns,
    tool_names,
    t_in,
    t_out,
    truncated,
    recovered,
    t0,
    transcript,
    finish=None,
    repair_rounds=0,
) -> dict:
    return {
        "final": final,
        "turns": turns,
        "tool_calls": len(tool_names),
        "tool_names": tool_names,
        "tokens_in": t_in,
        "tokens_out": t_out,
        "truncated": truncated,
        "recovered_from_reasoning": recovered,
        "finish_reason": finish,  # stop_reason raw: end_turn / tool_use / refusal / max_tokens
        "repaired": repair_rounds > 0,  # pass riparato ≠ pass liscio: mai sommarli
        "repair_rounds": repair_rounds,
        "latency_s": round(time.perf_counter() - t0, 2),
        "transcript": transcript,
    }
