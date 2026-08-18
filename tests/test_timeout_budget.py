"""How `timeout_seconds` resolves against an agent's own declared timeout.

The old rule was `min(declared, budget * 2)`. With the agent default and the
config default both at 45s, that could only ever *lower* a timeout: setting
`timeout_seconds: 120` computed min(45, 240) = 45 and changed nothing. It was
offered as the fix for agents timing out at 45s, and it was silently a no-op —
the worst kind of configuration knob, one that looks connected.
"""

from __future__ import annotations

import pytest

from marketswarm.agents.base import DEFAULT_AGENT_TIMEOUT
from marketswarm.orchestrator import scaled_timeout


def test_an_untouched_config_changes_nothing():
    assert scaled_timeout(DEFAULT_AGENT_TIMEOUT, DEFAULT_AGENT_TIMEOUT) == \
        DEFAULT_AGENT_TIMEOUT


def test_raising_the_budget_actually_raises_the_timeout():
    """The regression. This is what `timeout_seconds: 120` was supposed to do."""
    raised = scaled_timeout(DEFAULT_AGENT_TIMEOUT, 120.0)
    assert raised == pytest.approx(120.0)
    assert raised > DEFAULT_AGENT_TIMEOUT, (
        "raising timeout_seconds did not raise the agent timeout — the knob is "
        "connected to nothing, which is how the original advice failed")


def test_lowering_the_budget_still_bites():
    assert scaled_timeout(DEFAULT_AGENT_TIMEOUT, 10.0) == pytest.approx(10.0)


def test_an_agents_relative_need_survives_a_budget_change():
    """An agent asking for double the standard should keep asking for double."""
    hungry = DEFAULT_AGENT_TIMEOUT * 2
    for budget in (10.0, 45.0, 120.0, 300.0):
        standard = scaled_timeout(DEFAULT_AGENT_TIMEOUT, budget)
        assert scaled_timeout(hungry, budget) == pytest.approx(standard * 2), (
            f"at budget={budget} the hungry agent lost its relative need")


def test_the_result_is_monotonic_in_the_budget():
    previous = 0.0
    for budget in (1.0, 10.0, 45.0, 90.0, 120.0, 600.0):
        current = scaled_timeout(DEFAULT_AGENT_TIMEOUT, budget)
        assert current > previous, "a larger budget must never mean less time"
        previous = current


@pytest.mark.parametrize("budget", [0.0, -1.0, -45.0])
def test_a_nonsense_budget_falls_back_rather_than_zeroing_the_timeout(budget):
    """A misconfigured budget must not give every agent a 0s timeout, which
    would fail the whole swarm instantly and look like a total outage."""
    assert scaled_timeout(DEFAULT_AGENT_TIMEOUT, budget) == DEFAULT_AGENT_TIMEOUT


def test_a_tiny_budget_never_produces_an_unrunnable_timeout():
    assert scaled_timeout(DEFAULT_AGENT_TIMEOUT, 0.001) >= 1.0
