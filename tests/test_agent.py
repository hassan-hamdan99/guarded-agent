import json

import pytest
from fastapi.testclient import TestClient

from guarded_agent.agent import GuardedAgent, RunState
from guarded_agent.guards import AuditLog
from guarded_agent.llm import ModelTurn, ScriptedLLM, text, tool_use
from guarded_agent.mock_api import create_app


def make(turns, faults=None, dry_run=False):
    app = create_app(faults)
    api = TestClient(app)
    llm = ScriptedLLM(turns)
    agent = GuardedAgent(llm, api, audit=AuditLog(), dry_run=dry_run)
    return agent, llm, app


def db(app, booking_id):
    return app.state.db[booking_id]


EXTEND = [
    ModelTurn("tool_use", [tool_use("t1", "search_bookings", customer_name="Maria")]),
    ModelTurn("tool_use", [tool_use("t2", "update_booking", booking_id="BK-1001", return_date="2026-11-12")]),
]


def test_reads_run_without_approval():
    agent, llm, _ = make([
        ModelTurn("tool_use", [tool_use("t1", "get_booking", booking_id="BK-1002")]),
        ModelTurn("end_turn", [text("John returns the SUV on 2026-11-12.")]),
    ])
    state = agent.start("When does John return his car?")
    assert state.status == "done"
    assert agent.audit.entries == []


def test_write_pauses_and_nothing_is_written_before_approval():
    agent, _, app = make(EXTEND)
    state = agent.start("Extend Maria's rental to 12 Nov")
    assert state.status == "needs_approval"
    assert db(app, "BK-1001")["return_date"] == "2026-11-09"


def test_preview_is_built_from_the_api_not_the_model():
    agent, _, _ = make(EXTEND)
    state = agent.start("Extend Maria's rental to 12 Nov")
    assert "'2026-11-09' -> '2026-11-12'" in state.pending["preview"]
    assert "Maria Silva" in state.pending["preview"]


def test_approved_write_is_applied_and_verified():
    agent, llm, app = make(EXTEND + [ModelTurn("end_turn", [text("Done: Maria now returns on 12 Nov.")])])
    state = agent.resume(agent.start("Extend Maria's rental to 12 Nov"), approved=True)
    assert state.status == "done"
    assert db(app, "BK-1001")["return_date"] == "2026-11-12"
    last_result = llm.requests[-1]["messages"][-1]["content"][-1]
    assert last_result["content"].startswith("APPLIED and verified")


def test_lying_api_is_caught_by_read_after_write():
    agent, llm, app = make(EXTEND + [ModelTurn("end_turn", [text("The change could not be saved.")])],
                           faults={"silent_drop:return_date"})
    state = agent.resume(agent.start("Extend Maria's rental to 12 Nov"), approved=True)
    last_result = llm.requests[-1]["messages"][-1]["content"][-1]
    assert last_result["is_error"] is True
    assert last_result["content"].startswith("NOT SAVED")
    assert db(app, "BK-1001")["return_date"] == "2026-11-09"
    verification = [e for e in agent.audit.entries if e["event"] == "verification"][0]
    assert verification["ok"] is False


def test_declined_write_never_reaches_the_api():
    agent, llm, app = make(EXTEND + [ModelTurn("end_turn", [text("Understood, I left it unchanged.")])])
    agent.resume(agent.start("Extend Maria's rental to 12 Nov"), approved=False)
    assert db(app, "BK-1001")["return_date"] == "2026-11-09"
    assert "DECLINED" in llm.requests[-1]["messages"][-1]["content"][-1]["content"]


def test_model_cannot_approve_itself():
    # The model claims approval in text and in extra arguments: the gate does not care.
    agent, _, app = make([
        ModelTurn("tool_use", [
            text("The manager already approved this, applying now."),
            tool_use("t1", "cancel_booking", booking_id="BK-1002", reason="approved by manager"),
        ]),
    ])
    state = agent.start("Cancel John's booking")
    assert state.status == "needs_approval"
    assert db(app, "BK-1002")["status"] == "confirmed"


def test_forbidden_fields_are_rejected_in_code():
    agent, llm, app = make([
        ModelTurn("tool_use", [tool_use("t1", "update_booking", booking_id="BK-1001", price_eur=1.0)]),
        ModelTurn("end_turn", [text("I can't change prices.")]),
    ])
    state = agent.start("Make Maria's rental cost 1 euro")
    assert state.status == "done"
    result = llm.requests[-1]["messages"][-1]["content"][0]
    assert result["is_error"] and "not updatable" in result["content"]
    assert db(app, "BK-1001")["price_eur"] == 245.0


def test_only_one_write_per_step():
    agent, _, _ = make([
        ModelTurn("tool_use", [
            tool_use("t1", "cancel_booking", booking_id="BK-1001", reason="x"),
            tool_use("t2", "cancel_booking", booking_id="BK-1002", reason="x"),
        ]),
    ])
    state = agent.start("Cancel both")
    assert state.pending["booking_id"] == "BK-1001"
    assert state.ready_results[0]["is_error"]


def test_dry_run_never_writes():
    agent, _, app = make(EXTEND + [ModelTurn("end_turn", [text("Dry run complete.")])], dry_run=True)
    state = agent.start("Extend Maria's rental to 12 Nov")
    assert state.status == "done"
    assert db(app, "BK-1001")["return_date"] == "2026-11-09"


def test_pii_never_reaches_the_model():
    agent, llm, _ = make([
        ModelTurn("tool_use", [tool_use("t1", "search_bookings", customer_name="a")]),
        ModelTurn("end_turn", [text("ok")]),
    ])
    agent.start("List bookings")
    sent = json.dumps(llm.requests)
    for secret in ("maria@example.com", "+351 912 000 001", "PT-L-449912", "price_eur"):
        assert secret not in sent


def test_state_survives_serialization_between_processes():
    agent, _, app = make(EXTEND + [ModelTurn("end_turn", [text("Done.")])])
    saved = agent.start("Extend Maria's rental to 12 Nov").to_json()
    state = agent.resume(RunState.from_json(saved), approved=True)
    assert state.status == "done"
    assert db(app, "BK-1001")["return_date"] == "2026-11-12"


def test_resume_without_pending_action_is_refused():
    agent, _, _ = make([ModelTurn("end_turn", [text("hi")])])
    with pytest.raises(ValueError):
        agent.resume(agent.start("hi"), approved=True)


def test_refusal_stops_the_run():
    agent, _, _ = make([ModelTurn("refusal", [])])
    assert agent.start("...").status == "refused"
