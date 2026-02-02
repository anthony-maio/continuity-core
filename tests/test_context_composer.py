from continuity_core.context import Candidate, ContextComposer


def test_context_composer_respects_budget():
    composer = ContextComposer(token_budget=10, epsilon=0.0, lambda_penalty=0.0)
    candidates = [
        Candidate("1", "alpha", "store", 5, 0.9, 10.0, 0.5, 0.8, 0.8, 0.7),
        Candidate("2", "beta", "store", 6, 0.8, 10.0, 0.4, 0.7, 0.7, 0.6),
        Candidate("3", "gamma", "store", 4, 0.2, 10.0, 0.2, 0.5, 0.5, 0.4),
    ]
    chosen = composer.select(candidates)
    assert sum(c.token_cost for c in chosen) <= 10
    assert chosen
