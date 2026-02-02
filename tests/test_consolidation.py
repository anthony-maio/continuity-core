from continuity_core.memory.consolidation import ConsolidationFeedback, ConsolidationPolicy, RecallGatedConsolidator
from continuity_core.memory.gates import ThresholdGate


def test_nested_learning_updates_policy():
    policy = ConsolidationPolicy(pattern_depth=3, cross_memory_integration=0.6)
    consolidator = RecallGatedConsolidator(gate=ThresholdGate(theta=0.1), policy=policy)

    feedback = ConsolidationFeedback(retrieval_accuracy=0.5, coherence_gain=0.05, eviction_count=2)
    consolidator.nested_learning_cycle(feedback)

    assert consolidator.policy.pattern_depth >= 3
    assert consolidator.policy.cross_memory_integration >= 0.6
    assert consolidator.policy.gate_bias > 0.0
