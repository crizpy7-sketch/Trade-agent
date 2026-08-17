"""Tests for the Review Gate and revision loop.

The defect these exist to prevent: red-team criticism that cannot change the
output. Every test here asserts that an objection actually *bites*.
"""

from __future__ import annotations

import pytest

from marketswarm.review.gate import (
    Finding,
    GatePolicy,
    ReviewDecision,
    ReviewGate,
    ReviewStatus,
    Severity,
)
from marketswarm.review.loop import ReviewLoop, findings_from_redteam_report


def idea(confidence=70, **kw):
    base = {
        "symbol": "NVDA", "direction": "long", "entry": 100.0,
        "target": 103.0, "stop": 98.0, "confidence": confidence,
        "probability": 0.55, "expected_r": 0.2, "invalidation": "loses 98",
    }
    base.update(kw)
    return base


# ---------------- the four required behaviours ----------------

def test_severe_finding_lowers_confidence():
    gate = ReviewGate()
    d = gate.review([Finding("thin evidence", Severity.HIGH)], original_confidence=80)
    assert d.revised_confidence < 80
    assert d.status is not ReviewStatus.APPROVE
    assert d.confidence_delta < 0


def test_severe_finding_can_request_more_research():
    gate = ReviewGate()
    d = gate.review(
        [Finding("gap may be an earnings artefact", Severity.HIGH,
                 requires_evidence="confirm whether NVDA reported last night")],
        original_confidence=75, iteration=0, max_iterations=2,
    )
    assert d.status is ReviewStatus.REQUEST_MORE_RESEARCH
    assert d.required_followups == ["confirm whether NVDA reported last night"]


def test_finding_can_modify_an_idea():
    gate = ReviewGate()
    d = gate.review(
        [Finding("target beyond the implied move", Severity.MEDIUM,
                 suggested_modification={"target": 101.5, "size_multiplier": 0.5})],
        original_confidence=60,
    )
    assert d.status is ReviewStatus.MODIFY
    out = gate.apply(idea(60), d)
    assert out is not None
    assert out["target"] == 101.5
    assert out["size_multiplier"] == 0.5


def test_critical_finding_rejects_the_idea():
    gate = ReviewGate()
    d = gate.review([Finding("the underlying already reported and gapped the other way",
                             Severity.CRITICAL, invalidates_claim="earnings continuation")],
                    original_confidence=90)
    assert d.status is ReviewStatus.REJECT
    assert d.revised_confidence == 0
    assert gate.apply(idea(90), d) is None


# ---------------- policy guarantees ----------------

def test_critical_can_never_be_approved():
    """No configuration and no confidence level lets a CRITICAL through."""
    gate = ReviewGate()
    for conf in (1, 50, 99):
        d = gate.review([Finding("fatal", Severity.CRITICAL)], original_confidence=conf)
        assert d.status is ReviewStatus.REJECT


def test_enough_high_findings_force_rejection():
    gate = ReviewGate(GatePolicy(high_findings_to_reject=3))
    findings = [Finding(f"problem {i}", Severity.HIGH) for i in range(3)]
    assert gate.review(findings, 95).status is ReviewStatus.REJECT


def test_high_finding_caps_confidence():
    gate = ReviewGate(GatePolicy(max_confidence_after_high=55))
    d = gate.review([Finding("one-sided book", Severity.HIGH)], original_confidence=99)
    assert d.revised_confidence <= 55


def test_confidence_floor_rejects_rather_than_publishing_noise():
    gate = ReviewGate(GatePolicy(min_confidence_to_survive=20))
    d = gate.review(
        [Finding("a", Severity.MEDIUM), Finding("b", Severity.MEDIUM),
         Finding("c", Severity.MEDIUM)],
        original_confidence=25,
    )
    assert d.status is ReviewStatus.REJECT
    assert "floor" in " ".join(d.reasons)


def test_no_findings_approves_untouched():
    gate = ReviewGate()
    d = gate.review([], original_confidence=64)
    assert d.status is ReviewStatus.APPROVE
    assert d.revised_confidence == 64


def test_only_minor_findings_still_costs_something():
    gate = ReviewGate()
    d = gate.review([Finding("nitpick", Severity.LOW)], original_confidence=70)
    assert d.status is ReviewStatus.APPROVE_WITH_REDUCED_CONFIDENCE
    assert d.revised_confidence < 70


def test_explicit_confidence_cap_overrides_computed_value():
    gate = ReviewGate()
    d = gate.review(
        [Finding("stale data", Severity.MEDIUM,
                 suggested_modification={"confidence_cap": 12})],
        original_confidence=90,
    )
    # Cap of 12 is under the survival floor, so the idea must not be published.
    assert d.status is ReviewStatus.REJECT


def test_applied_idea_carries_the_audit_fields():
    gate = ReviewGate()
    d = gate.review([Finding("thin", Severity.HIGH)], original_confidence=80)
    out = gate.apply(idea(80), d)
    assert out["original_confidence"] == 80
    assert out["confidence"] == d.revised_confidence
    assert out["review_status"] == d.status.value
    assert "Red team raised a high objection" in out["invalidation"]


# ---------------- the loop ----------------

def test_loop_terminates_and_records_history():
    calls = {"n": 0}

    def critic(current, iteration):
        calls["n"] += 1
        return [Finding("always complaining", Severity.MEDIUM)]

    loop = ReviewLoop(max_iterations=2)
    out = loop.run(idea(70), critic=critic)
    assert out.final_idea is not None
    assert out.iterations_used <= 2
    assert calls["n"] <= 2
    assert out.history[0].stage == "original"
    assert out.history[-1].stage == "final"


def test_loop_cannot_run_forever_when_critic_never_relents():
    """A critic that always demands more research must still terminate."""
    def critic(current, iteration):
        return [Finding("need more", Severity.HIGH, requires_evidence="anything")]

    loop = ReviewLoop(max_iterations=3)
    out = loop.run(idea(80), critic=critic)
    assert out.iterations_used <= 3
    assert out.final_idea is None or out.final_idea["confidence"] <= 80


def test_research_is_requested_at_most_once():
    """An unrelenting critic must not be able to demand research forever.

    After the first REQUEST_MORE_RESEARCH the gate latches, so the second pass
    resolves instead of asking again — terminating well inside the budget.
    """
    def critic(current, iteration):
        return [Finding("same objection", Severity.HIGH, requires_evidence="x")]

    out = ReviewLoop(max_iterations=5).run(idea(80), critic=critic)
    assert out.iterations_used == 2, "should settle on the second pass, not spin"
    verdicts = [r.decision.status for r in out.history
                if r.decision and r.stage == "red_team"]
    assert verdicts.count(ReviewStatus.REQUEST_MORE_RESEARCH) == 1
    assert out.final_decision.revised_confidence < 80


def test_repeated_identical_modification_breaks_the_loop():
    """The signature guard: the same MODIFY verdict twice means no progress."""
    def critic(current, iteration):
        return [Finding("always the same fix", Severity.MEDIUM,
                        suggested_modification={"size_multiplier": 0.5})]

    out = ReviewLoop(max_iterations=6).run(idea(70), critic=critic)
    assert out.iterations_used < 6
    assert "no progress" in out.terminated_because


def test_loop_investigator_is_invoked_and_can_resolve():
    seen = {}

    def critic(current, iteration):
        if current.get("resolved"):
            return []                       # satisfied after investigation
        return [Finding("unknown catalyst", Severity.HIGH,
                        requires_evidence="identify the catalyst")]

    def investigator(current, questions):
        seen["questions"] = questions
        current["resolved"] = True
        return current

    loop = ReviewLoop(max_iterations=3)
    out = loop.run(idea(70), critic=critic, investigator=investigator)
    assert seen["questions"] == ["identify the catalyst"]
    assert out.final_idea is not None
    assert out.final_decision.status in (ReviewStatus.APPROVE,
                                         ReviewStatus.APPROVE_WITH_REDUCED_CONFIDENCE)


def test_loop_without_investigator_publishes_open_questions():
    def critic(current, iteration):
        if iteration == 0:
            return [Finding("unclear", Severity.HIGH, requires_evidence="check the filing")]
        return []

    out = ReviewLoop(max_iterations=2).run(idea(70), critic=critic)
    assert out.final_idea is not None
    assert "check the filing" in out.final_idea.get("open_questions", [])


def test_loop_rejection_produces_no_idea_but_full_trail():
    def critic(current, iteration):
        return [Finding("fatal flaw", Severity.CRITICAL)]

    out = ReviewLoop(max_iterations=2).run(idea(90), critic=critic)
    assert out.rejected
    assert out.final_idea is None
    trail = out.audit_trail()
    assert trail[0]["stage"] == "original"
    assert any(r["decision"] and r["decision"]["status"] == "REJECT" for r in trail)
    assert "fatal flaw" in out.explain()


def test_loop_requires_at_least_one_iteration():
    with pytest.raises(ValueError):
        ReviewLoop(max_iterations=0)


def test_audit_trail_preserves_the_original_snapshot():
    def critic(current, iteration):
        return [Finding("reduce it", Severity.MEDIUM,
                        suggested_modification={"target": 101.0})]

    original = idea(70)
    out = ReviewLoop(max_iterations=2).run(original, critic=critic)
    assert out.history[0].snapshot["target"] == 103.0     # untouched original
    assert out.final_idea["target"] == 101.0              # revised final


# ---------------- adapter from the existing agent ----------------

class _FakeReport:
    def __init__(self, objections):
        self.data = {"objections": objections}


def test_adapter_maps_existing_redteam_output():
    rep = _FakeReport([
        {"severity": "high", "objection": "Every stock setup is long.", "test": "size as one"},
        {"severity": "medium", "objection": "NVDA target beyond implied move", "test": "check chain"},
        {"severity": "info", "objection": "minor note", "test": ""},
    ])
    findings = findings_from_redteam_report(rep, symbol="NVDA")
    sevs = {f.severity for f in findings}
    assert Severity.HIGH in sevs        # book-level objection applies to every idea
    assert Severity.MEDIUM in sevs      # symbol-specific objection matched
    assert all(isinstance(f, Finding) for f in findings)


def test_adapter_filters_objections_about_other_symbols():
    rep = _FakeReport([
        {"severity": "high", "objection": "AAPL target beyond implied move", "test": "t"},
    ])
    assert findings_from_redteam_report(rep, symbol="NVDA") == []


def test_adapter_handles_empty_report():
    assert findings_from_redteam_report(None) == []
    assert findings_from_redteam_report(_FakeReport([])) == []


def test_general_critical_objection_is_never_dropped():
    """The bug this guards: an objection that names no ticker must apply to
    every idea. Requiring an explicit symbol match silently discarded critical
    findings — exactly the class that must always bite."""
    rep = _FakeReport([{
        "severity": "critical",
        "objection": "the underlying already reported and gapped the other way",
        "test": "check the tape",
    }])
    for sym in ("NVDA", "AAPL", "SPY"):
        findings = findings_from_redteam_report(rep, symbol=sym)
        assert len(findings) == 1, f"critical finding dropped for {sym}"
        assert findings[0].severity is Severity.CRITICAL


def test_objection_naming_another_symbol_is_filtered():
    rep = _FakeReport([{"severity": "high",
                        "objection": "AAPL target beyond the implied move", "test": "t"}])
    assert findings_from_redteam_report(rep, symbol="NVDA") == []
    assert len(findings_from_redteam_report(rep, symbol="AAPL")) == 1


def test_non_ticker_uppercase_words_do_not_look_like_symbols():
    rep = _FakeReport([{"severity": "high",
                        "objection": "VIX spiked and CPI prints today", "test": "t"}])
    assert len(findings_from_redteam_report(rep, symbol="NVDA")) == 1
