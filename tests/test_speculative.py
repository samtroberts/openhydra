from coordinator.speculative import (
    DraftTokenModel,
    select_verified_token_ids,
    select_verified_tokens,
)


def test_draft_model_is_deterministic():
    model = DraftTokenModel(seed=9)
    a = model.propose("hello hydra", max_tokens=5)
    b = model.propose("hello hydra", max_tokens=5)
    assert a == b
    assert len(a) == 5


def test_select_verified_tokens_full_match_accepts_all():
    selected = select_verified_tokens(
        verified_tokens=["a", "b", "c"],
        draft_tokens=["a", "b", "c"],
    )
    assert selected.accepted_tokens == ["a", "b", "c"]
    assert selected.matched_prefix == 3
    assert selected.mismatch is False


def test_select_verified_tokens_mismatch_accepts_prefix_and_strong_token():
    selected = select_verified_tokens(
        verified_tokens=["a", "b", "c"],
        draft_tokens=["a", "x", "y"],
    )
    assert selected.accepted_tokens == ["a", "b"]
    assert selected.matched_prefix == 1
    assert selected.mismatch is True


def test_select_verified_token_ids_mismatch_accepts_prefix_and_target_correction():
    selected = select_verified_token_ids(
        verified_token_ids=[10, 20, 30],
        draft_token_ids=[10, 99, 88],
    )
    assert selected.accepted_token_ids == [10, 20]
    assert selected.matched_prefix == 1
    assert selected.mismatch is True
