"""State machine coverage: every legal edge plus a sample of illegal jumps."""

import pytest

from app.states import (
    TERMINAL_STATES,
    TRANSITIONS,
    CreativeState,
    InvalidTransition,
    advance,
    can_transition,
)

ALL_LEGAL_EDGES = [
    (current, target) for current, targets in TRANSITIONS.items() for target in targets
]

ILLEGAL_SAMPLE = [
    (CreativeState.DRAFT, CreativeState.GENERATING),
    (CreativeState.DRAFT, CreativeState.PUBLISHED),
    (CreativeState.RESEARCHED, CreativeState.SCRIPT_APPROVED),
    (CreativeState.SCRIPT_READY, CreativeState.GENERATING),
    (CreativeState.SCRIPT_APPROVED, CreativeState.QC_REQUIRED),
    (CreativeState.GENERATING, CreativeState.READY),
    (CreativeState.GENERATING, CreativeState.GENERATING),
    (CreativeState.QC_REQUIRED, CreativeState.FINAL_APPROVED),
    (CreativeState.READY, CreativeState.SCHEDULED),
    (CreativeState.FINAL_APPROVED, CreativeState.PUBLISHING),
    (CreativeState.SCHEDULED, CreativeState.PUBLISHED),
    (CreativeState.PUBLISHING, CreativeState.SCHEDULED),
    (CreativeState.PARTIAL, CreativeState.FAILED),
    (CreativeState.NEEDS_ACTION, CreativeState.FAILED),
    (CreativeState.PUBLISHED, CreativeState.DRAFT),
    (CreativeState.PUBLISHED, CreativeState.PUBLISHING),
    (CreativeState.FAILED, CreativeState.GENERATING),
    (CreativeState.FAILED, CreativeState.DRAFT),
]


def test_transition_table_covers_every_state() -> None:
    assert set(TRANSITIONS) == set(CreativeState)


@pytest.mark.parametrize(("current", "target"), ALL_LEGAL_EDGES)
def test_every_legal_edge_advances(current: CreativeState, target: CreativeState) -> None:
    assert can_transition(current, target)
    assert advance(current, target) == target


@pytest.mark.parametrize(("current", "target"), ILLEGAL_SAMPLE)
def test_illegal_edges_raise(current: CreativeState, target: CreativeState) -> None:
    assert not can_transition(current, target)
    with pytest.raises(InvalidTransition) as excinfo:
        advance(current, target)
    assert excinfo.value.current == current
    assert excinfo.value.target == target
    assert current.value in str(excinfo.value)
    assert target.value in str(excinfo.value)


def test_illegal_sample_edges_are_actually_illegal() -> None:
    legal = set(ALL_LEGAL_EDGES)
    assert not legal.intersection(ILLEGAL_SAMPLE)


def test_terminal_states_have_no_outgoing_edges() -> None:
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == frozenset()


def test_happy_path_walk_through_the_full_pipeline() -> None:
    path = [
        CreativeState.DRAFT,
        CreativeState.RESEARCHED,
        CreativeState.SCRIPT_READY,
        CreativeState.SCRIPT_APPROVED,
        CreativeState.GENERATING,
        CreativeState.QC_REQUIRED,
        CreativeState.READY,
        CreativeState.FINAL_APPROVED,
        CreativeState.SCHEDULED,
        CreativeState.PUBLISHING,
        CreativeState.PUBLISHED,
    ]
    state = path[0]
    for target in path[1:]:
        state = advance(state, target)
    assert state == CreativeState.PUBLISHED


def test_scene_retry_loop_generating_qc() -> None:
    state = advance(CreativeState.GENERATING, CreativeState.QC_REQUIRED)
    state = advance(state, CreativeState.GENERATING)  # failed-scene retry
    assert state == CreativeState.GENERATING


def test_partial_can_retry_remaining_targets() -> None:
    assert advance(CreativeState.PARTIAL, CreativeState.PUBLISHING) == CreativeState.PUBLISHING
    assert advance(CreativeState.PARTIAL, CreativeState.PUBLISHED) == CreativeState.PUBLISHED
