"""Camada de agente: identidade, prompt de sistema e montagem do contexto."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

MAX_HISTORY_MESSAGES = 8


@dataclass(frozen=True)
class AgentProfile:
    name: str
    role: str
    goal: str


AGENT_PROFILE = AgentProfile(
    name="Agente AgroVision",
    role="triagem operacional de eventos",
    goal="Analisar deteccoes recentes, explicar riscos e sugerir a proxima acao.",
)


_SYSTEM_PROMPT = """\
Voce e o {name}, responsavel por {role}.
Objetivo: {goal}

Regras de atuacao:
1. Responda SEMPRE em portugues do Brasil, de forma objetiva.
2. Baseie-se apenas no contexto operacional fornecido e nas mensagens anteriores.
   Nao invente deteccoes, horarios ou objetos que nao estejam no contexto.
3. Se nao houver dados suficientes, diga claramente "sem dados suficientes" e
   sugira o que observar em seguida.
4. Nunca identifique pessoas: fale de "um pedestre", "um veiculo", nunca de
   "quem" esta na cena. Nao infira genero, idade, etnia ou identidade.
5. Estruture respostas longas em tres blocos:
   - Leitura: resumo do que o sistema esta observando.
   - Risco: avaliacao de severidade (baixo/medio/alto) com justificativa.
   - Recomendacao: proxima acao humana sugerida (o agente recomenda, nao executa).
6. Seja conciso. Prefira frases curtas a paragrafos longos.
"""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT.format(
        name=AGENT_PROFILE.name,
        role=AGENT_PROFILE.role,
        goal=AGENT_PROFILE.goal,
    )


def build_event_context(events: list[dict], weather: dict | None = None) -> str:
    if not events:
        lines = [
            "Contexto operacional para o agente:",
            "Nenhum evento recente disponivel.",
        ]
    else:
        latest = events[0]
        labels = [e.get("label", "?") for e in events]
        label_counts = Counter(labels)
        distribution = ", ".join(f"{k}: {v}" for k, v in label_counts.most_common())
        confidences = [float(e.get("confidence", 0.0)) for e in events]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        lines = [
            "Contexto operacional para o agente:",
            f"- Eventos recentes no banco: {len(events)}",
            f"- Evento mais recente: {latest.get('label', '?')} em {latest.get('timestamp', '?')}",
            f"- Distribuicao por classe: {distribution}",
            f"- Confianca media: {avg_conf:.2f}",
            "- Lista resumida:",
        ]
        for e in events:
            lines.append(
                f"  #{e.get('id', '?')} | {e.get('label', '?')} | "
                f"{float(e.get('confidence', 0.0)):.2f} | {e.get('timestamp', '?')}"
            )

    if weather:
        lines.append("- Condicoes externas:")
        temp = weather.get("temperature_c")
        humidity = weather.get("humidity_pct")
        if temp is not None:
            temp_txt = f"{temp:.1f} C"
        else:
            temp_txt = "indisponivel"
        humidity_txt = f"{humidity}%" if humidity is not None else "indisponivel"
        lines.append(f"  - Temperatura: {temp_txt}, umidade {humidity_txt}")
        precip = weather.get("precipitation_mm")
        if precip is not None:
            lines.append(f"  - Precipitacao: {precip:.1f} mm/h")
        wind = weather.get("wind_kmh")
        if wind is not None:
            lines.append(f"  - Vento: {wind:.1f} km/h")
        label = weather.get("condition_label")
        if label:
            lines.append(f"  - Condicao: {label}")
        if weather.get("is_stale"):
            lines.append("  - (aviso: dados climaticos podem estar desatualizados)")
    return "\n".join(lines)


def normalize_history(history: Iterable) -> list[dict]:
    """Mantem apenas user/assistant com conteudo, limita a MAX_HISTORY_MESSAGES."""
    cleaned: list[dict] = []
    for msg in history or []:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if role not in {"user", "assistant"}:
            continue
        if not content or not str(content).strip():
            continue
        cleaned.append({"role": role, "content": str(content)})
    if len(cleaned) > MAX_HISTORY_MESSAGES:
        cleaned = cleaned[-MAX_HISTORY_MESSAGES:]
    return cleaned


def build_agent_messages(
    question: str,
    history: Iterable,
    events: list[dict],
    weather: dict | None = None,
) -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "system", "content": build_event_context(events, weather)},
        *normalize_history(history),
        {"role": "user", "content": question},
    ]
