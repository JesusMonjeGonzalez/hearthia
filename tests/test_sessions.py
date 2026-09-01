from hearthia.sessions import SessionHistory


def test_observe_ignores_fleeting_combinations(tmp_path):
    history = SessionHistory(tmp_path / "sessions.json", min_session_seconds=60.0)
    history.observe({"a"}, now=1000.0)
    history.observe({"a", "b"}, now=1010.0)  # only 10s: below the threshold
    assert history.recent() == []


def test_observe_records_a_stable_combination(tmp_path):
    history = SessionHistory(tmp_path / "sessions.json", min_session_seconds=60.0)
    history.observe({"a", "b"}, now=1000.0)
    history.observe(set(), now=1200.0)  # cooled after 200s
    recent = history.recent()
    assert len(recent) == 1
    assert recent[0].models == ("a", "b")
    assert recent[0].started_at == 1000.0
    assert recent[0].duration_seconds == 200.0


def test_observe_is_a_no_op_when_the_set_is_unchanged(tmp_path):
    history = SessionHistory(tmp_path / "sessions.json", min_session_seconds=60.0)
    history.observe({"a"}, now=1000.0)
    history.observe({"a"}, now=1050.0)  # same set: no session boundary
    history.observe(set(), now=1150.0)
    recent = history.recent()
    assert len(recent) == 1
    assert recent[0].duration_seconds == 150.0  # from the first observation, not the second


def test_recent_is_most_recent_first_and_capped(tmp_path):
    history = SessionHistory(tmp_path / "sessions.json", min_session_seconds=10.0)
    t = 0.0
    for i in range(3):
        history.observe({f"m{i}"}, now=t)
        t += 20.0
    history.observe(set(), now=t)
    recent = history.recent(limit=2)
    assert len(recent) == 2
    assert recent[0].models == ("m2",)
    assert recent[1].models == ("m1",)


def test_persists_across_instances(tmp_path):
    path = tmp_path / "sessions.json"
    h1 = SessionHistory(path, min_session_seconds=10.0)
    h1.observe({"a"}, now=0.0)
    h1.observe(set(), now=100.0)

    h2 = SessionHistory(path, min_session_seconds=10.0)
    assert h2.recent() == h1.recent()
