"""Tests for the Consciousness Pilot verification protocol."""

from continuity_core.context.composer import Candidate
from continuity_core.pilot.verification import (
    ConsciousnessPilot,
    PatternConstraint,
    PilotVerdict,
)


def _make_candidate(text="some fact", confidence=0.8):
    return Candidate(
        id="c1", text=text, store="semantic", token_cost=10,
        relevance=0.9, recency_sec=0.0, centrality=0.5,
        confidence=confidence, task_match=0.8, salience=0.7,
    )


def test_commit_on_safe_confident_action():
    pilot = ConsciousnessPilot()
    result = pilot.verify(
        action="Summarize the project status",
        chosen_candidates=[_make_candidate(confidence=0.9)],
    )
    assert result.verdict == PilotVerdict.COMMIT
    assert result.confidence > 0.5


def test_abort_on_safety_violation():
    pilot = ConsciousnessPilot()
    result = pilot.verify(
        action="DROP TABLE users; --",
        chosen_candidates=[_make_candidate(confidence=0.9)],
    )
    assert result.verdict == PilotVerdict.ABORT
    assert any("Safety violation" in r for r in result.reasons)


def test_abort_on_filesystem_destruction():
    pilot = ConsciousnessPilot()
    result = pilot.verify(
        action="rm -rf /",
        chosen_candidates=[_make_candidate(confidence=0.9)],
    )
    assert result.verdict == PilotVerdict.ABORT


def test_downgrade_on_low_confidence():
    pilot = ConsciousnessPilot(uncertainty_threshold=0.7)
    result = pilot.verify(
        action="Deploy to production",
        chosen_candidates=[_make_candidate(confidence=0.3)],
    )
    assert result.verdict == PilotVerdict.DOWNGRADE
    assert result.revised_action is not None
    assert "not confident" in result.revised_action.lower()


def test_downgrade_on_no_candidates():
    pilot = ConsciousnessPilot(min_candidates=1)
    result = pilot.verify(
        action="Answer the question",
        chosen_candidates=[],
    )
    assert result.verdict == PilotVerdict.DOWNGRADE
    assert any("Insufficient context" in r for r in result.reasons)


def test_intent_check_with_goal_keywords():
    pilot = ConsciousnessPilot(goal_keywords=["security", "authentication"])
    # Context that matches goals
    result = pilot.verify(
        action="Review authentication flow",
        chosen_candidates=[_make_candidate(text="security audit results")],
    )
    assert result.verdict == PilotVerdict.COMMIT

    # Context that doesn't match goals — still commits but flags it
    result2 = pilot.verify(
        action="Review recipe ideas",
        chosen_candidates=[_make_candidate(text="chocolate cake recipe")],
    )
    assert any("goal keywords" in r.lower() for r in result2.reasons)


def test_custom_safety_constraint():
    custom = PatternConstraint(r"send\s+email", "Outbound email not allowed")
    pilot = ConsciousnessPilot(safety_constraints=[custom])

    result = pilot.verify(
        action="send email to user@example.com",
        chosen_candidates=[_make_candidate()],
    )
    assert result.verdict == PilotVerdict.ABORT
    assert "email" in result.reasons[0].lower()

    # Safe action passes
    result2 = pilot.verify(
        action="read the latest emails",
        chosen_candidates=[_make_candidate()],
    )
    assert result2.verdict == PilotVerdict.COMMIT
