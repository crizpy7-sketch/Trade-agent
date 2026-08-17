"""Adversarial review.

Every other agent is looking for reasons to have a view. This one exists to
attack the conclusion, and it runs last so it can see the finished playbook.

Two layers:
  1. Mechanical checks that catch the specific ways this system fails — all
     ideas on the same side of the market, targets beyond the implied move,
     conviction resting on one source, probabilities that contradict the
     historical base rate.
  2. An optional LLM pass whose only instruction is to argue the other side.

The mechanical layer runs with no API key and is the part that has teeth.
"""

from __future__ import annotations

from .base import AgentReport, BaseAgent, SwarmContext


class RedTeamAgent(BaseAgent):
    name = "red_team"
    description = "Argues against the playbook and flags what the swarm missed"
    depends_on = ("playbook", "cross_verify", "risk", "options_flow",
                  "econ_calendar", "volatility_regime")

    async def run(self, ctx: SwarmContext) -> AgentReport:
        rep = AgentReport(agent=self.name, headline="Red team")

        pb = ctx.report_of("playbook")
        if not pb:
            rep.status = "skipped"
            rep.headline = "Red team: no playbook to attack"
            return rep

        calls = pb.data.get("calls", [])
        puts = pb.data.get("puts", [])
        stocks = pb.data.get("stocks", [])
        ideas = calls + puts + stocks

        objections: list[dict] = []

        # --- 1. one-sided book ---
        if stocks:
            longs = sum(1 for i in stocks if i["direction"] == "long")
            shorts = len(stocks) - longs
            if longs == 0 or shorts == 0:
                side = "long" if shorts == 0 else "short"
                objections.append({
                    "severity": "high",
                    "objection": f"Every stock setup is {side}. This is one bet on market direction "
                                 f"wearing {len(stocks)} costumes.",
                    "test": "If the index gaps against you, all of them lose together. Size the "
                            "whole book as a single position, not as several.",
                })

        # --- 2. targets beyond what the option market is pricing ---
        flows = ctx.data_of("options_flow", "flows", {}) or {}
        for i in ideas:
            f = flows.get(i["symbol"])
            if not f or not f.get("implied_move_pct"):
                continue
            move_needed = abs(i["target"] - i["entry"]) / i["entry"] * 100
            implied = f["implied_move_pct"]
            if move_needed > implied * 1.05:
                objections.append({
                    "severity": "medium",
                    "objection": f"{i['symbol']} target requires a {move_needed:.2f}% move, but the "
                                 f"option market prices only ±{implied:.2f}% for the session.",
                    "test": "Either the chain is mispriced or the target is. The chain is usually "
                            "right; name the specific catalyst that makes today different.",
                })

        # --- 3. probability vs. the learned base rate ---
        lessons = ctx.lessons or []
        regime = ctx.data_of("volatility_regime", "regime", "unknown")
        for l in lessons:
            if l.get("scope") == f"regime={regime}" and l.get("effect_size", 0) < -0.1:
                objections.append({
                    "severity": "high",
                    "objection": f"The agent's own scored history says this regime underperforms: "
                                 f"{l['lesson']}",
                    "test": "The morning narrative is arguing against the track record. The track "
                            "record has more observations than the narrative.",
                })

        # --- 4. thin evidence dressed as conviction ---
        eff_n = ctx.data_of("cross_verify", "effective_n", 0)
        raw_n = ctx.data_of("cross_verify", "raw_signal_count", 0)
        if raw_n and eff_n < 2.5:
            objections.append({
                "severity": "high",
                "objection": f"{raw_n} signals collapse to {eff_n:.1f} independent ones. The apparent "
                             f"agreement is mostly the same information counted repeatedly.",
                "test": "Ask what would have to be true for the read to be wrong. If the answer is "
                        "'one thing' — the overnight risk tone — then there is one signal here.",
            })

        # --- 5. probabilities near the coin flip carrying real size ---
        marginal = [i for i in ideas if 0.45 <= i["probability"] <= 0.55]
        if marginal:
            objections.append({
                "severity": "medium",
                "objection": f"{len(marginal)} of {len(ideas)} ideas sit within 5 points of a coin "
                             f"flip on P(target before stop).",
                "test": "At those odds the outcome is decided by execution and costs, not by the "
                        "analysis. That is a fee, not an edge.",
            })

        # --- 6. event risk the playbook ignored ---
        very_high = ctx.data_of("econ_calendar", "very_high_impact", []) or []
        if very_high and ideas:
            objections.append({
                "severity": "high",
                "objection": f"{', '.join(very_high)} prints today, yet {len(ideas)} directional "
                             f"ideas are proposed anyway.",
                "test": "Pre-release ranges compress and the first post-release move reverses often. "
                        "Every technical level in this report has a shorter half-life than usual.",
            })

        # --- 7. the strongest counter-argument to the directional read ---
        p_up = ctx.data_of("cross_verify", "probability", 0.5)
        contributions = ctx.data_of("cross_verify", "contributions", {}) or {}
        if contributions:
            against = {k: v for k, v in contributions.items()
                       if (v < 0) == (p_up >= 0.5) and abs(v) > 0.01}
            if against:
                worst = max(against.items(), key=lambda kv: abs(kv[1]))
                objections.append({
                    "severity": "info",
                    "objection": f"The strongest dissenting signal is {worst[0]} at "
                                 f"{worst[1]:+.3f} log-odds against the conclusion.",
                    "test": "It was outvoted, not refuted. If it is the one signal with a real "
                            "mechanism behind it today, the vote is wrong.",
                })

        # --- 8. concentration ---
        if ideas:
            syms = [i["symbol"] for i in ideas]
            dupes = {s: syms.count(s) for s in set(syms) if syms.count(s) > 1}
            if dupes:
                objections.append({
                    "severity": "medium",
                    "objection": f"Repeated exposure to {', '.join(dupes)} across the option and "
                                 f"stock sections.",
                    "test": "These are the same trade expressed twice, not two ideas. Count the "
                            "risk once.",
                })

        for o in sorted(objections, key=lambda x: {"high": 0, "medium": 1, "info": 2}[x["severity"]]):
            rep.add(f"[{o['severity']}] {o['objection']} → {o['test']}")

        if not objections:
            rep.add("No structural objection found. That is unusual and is itself worth a second "
                    "look — it more often means the checks missed something than that the analysis "
                    "is airtight.")

        highs = [o for o in objections if o["severity"] == "high"]
        rep.data = {
            "objections": objections,
            "high_severity": len(highs),
            "recommend_stand_down": len(highs) >= 2,
        }
        rep.headline = (
            f"{len(objections)} objections ({len(highs)} high severity)"
            + (" — consider standing down" if len(highs) >= 2 else "")
        )
        rep.confidence = 0.7

        if len(highs) >= 2:
            rep.add(
                "Two or more high-severity objections stand. The honest read is that today's "
                "conviction is not supported by independent evidence — trade smaller than the "
                "risk budget suggests, or not at all."
            )
        return rep
