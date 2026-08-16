"""Optional LLM narrative layer.

The numbers are produced by code, not by a language model. The model is used
only for what it is actually good at: reading the whole evidence set and
writing the connective tissue a human wants at 08:15 — what changed, what it
means, and what would falsify it.

The prompt forbids inventing figures, and the caller passes only computed
values, so the model has nothing to hallucinate from. If no API key is present
the agent runs fully and simply omits the narrative.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("marketswarm.llm")

SYSTEM = """You are the senior analyst on a quantitative pre-market research desk.

You are given the complete, already-computed output of an evidence-gathering
swarm: prices, probabilities, statistics, filings and headlines. Your job is to
write the morning narrative a portfolio manager reads before the open.

Hard rules:
- Use ONLY numbers present in the input. Never invent, round differently, or
  extrapolate a figure. If something is not in the input, do not mention it.
- Distinguish observation from inference. "ES is down 0.4%" is an observation.
  "Risk appetite is deteriorating" is an inference and must be labelled as one.
- Lead with what would make the read wrong. A morning note that only argues one
  side is worthless.
- Note explicitly where the evidence is thin, single-sourced, or contradictory.
- No hype, no "poised to", no price predictions beyond the supplied
  probabilities. Plain professional English.
- This is research, not advice. Do not tell the reader what to do with money.

Write 3-5 short paragraphs. No headings, no bullet lists, no preamble."""


class Narrator:
    def __init__(self, api_key: str | None, model: str = "claude-opus-4-5", enabled: bool = True):
        self.api_key = api_key
        self.model = model
        self.enabled = enabled and bool(api_key)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            log.info("anthropic package not installed — narrative disabled")
            return None
        try:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not create Anthropic client: %s", exc)
            return None
        return self._client

    def build_payload(self, result) -> dict:
        """Compact, numbers-only view of the run for the model."""
        payload = {
            "date": result.run_date.isoformat(),
            "index_probability_up": round(result.probability, 4),
            "confidence_0_100": result.confidence,
            "agents_ok": result.agents_ok,
            "agents_failed": result.agents_failed,
            "sections": {},
        }
        for name, rep in result.reports.items():
            if not rep.usable:
                payload["sections"][name] = {"status": rep.status, "error": rep.error}
                continue
            payload["sections"][name] = {
                "headline": rep.headline,
                "findings": rep.findings[:15],
                "signals": [
                    {"name": s.name, "p": round(s.probability, 3), "note": s.note}
                    for s in rep.signals[:8]
                ],
            }
        payload["ideas"] = {
            k: [
                {
                    "symbol": i["symbol"], "direction": i["direction"],
                    "probability": i["probability"], "expected_r": i["expected_r"],
                    "confidence": i["confidence"], "entry": i["entry"],
                    "target": i["target"], "stop": i["stop"],
                }
                for i in v
            ]
            for k, v in (result.ideas or {}).items()
        }
        return payload

    def narrate(self, result) -> str | None:
        if not self.enabled:
            return None
        client = self._get_client()
        if client is None:
            return None

        payload = self.build_payload(result)
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=1400,
                system=SYSTEM,
                messages=[{
                    "role": "user",
                    "content": "Swarm output for this morning:\n\n"
                               + json.dumps(payload, indent=2, default=str)[:60_000],
                }],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n\n".join(parts).strip() or None
        except Exception as exc:  # noqa: BLE001 — narrative is optional, never fatal
            log.warning("narrative generation failed: %s", exc)
            return None

    def postmortem(self, calibration: dict, recent: list[dict]) -> str | None:
        """Written after scoring: what the agent got wrong and why."""
        if not self.enabled:
            return None
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=900,
                system=(
                    "You review a forecasting system's scored track record. Be blunt and specific. "
                    "Identify patterns in the losses, say plainly whether the system has demonstrated "
                    "skill or is within noise of a coin flip, and name the single change most likely "
                    "to improve calibration. Use only the supplied numbers. No encouragement."
                ),
                messages=[{
                    "role": "user",
                    "content": json.dumps(
                        {"calibration": calibration, "recent_resolved": recent[:60]},
                        indent=2, default=str
                    )[:40_000],
                }],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n\n".join(parts).strip() or None
        except Exception as exc:  # noqa: BLE001
            log.warning("postmortem failed: %s", exc)
            return None
