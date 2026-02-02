from continuity_core.event_log import EventLog


def test_event_log_append_and_query():
    log = EventLog()
    log.log(actor="user", intent="ask", inp="hi", out="hello", tags=["greeting"])
    log.log(actor="agent", intent="respond", inp="hi", out="hello", tags=["response"])

    tail = log.tail(1)
    assert len(tail) == 1
    assert tail[0].actor == "agent"

    tagged = log.query(tag="greeting")
    assert len(tagged) == 1
    assert tagged[0].intent == "ask"
