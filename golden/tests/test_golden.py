"""Offline tests for the golden dataset builder (no API calls).

    .venv/bin/python -m pytest golden/tests -q
"""

import json
from collections import Counter

import pytest

from golden.export import seg_rows, tax_rows
from golden.generate import check_turns
from golden.seeds import MAX_SESSION_TURNS, make_seeds, write_seeds
from golden.sessions import Session
from golden.taxonomy import OPERATIONS_FILE, category_names, load_compatibility, load_tasks

PILOT = dict(n_sessions=60, seed=1)


@pytest.fixture(scope="module")
def pilot():
    return make_seeds(**PILOT)


@pytest.fixture(scope="module")
def pilot_v2():
    return make_seeds(500, 3, min_per_length=40)


def test_taxonomy_tree():
    tasks = load_tasks()
    assert len(tasks) == 64
    by_industry = Counter(t.industry for t in tasks.values())
    assert len(by_industry) == 8 and set(by_industry.values()) == {8}
    for t in tasks.values():
        assert t.use_case.startswith(t.industry + ".use_case.")
        assert t.workflow.startswith(t.industry + ".workflow.")
    for industry in by_industry:
        mine = [t for t in tasks.values() if t.industry == industry]
        assert len({t.use_case for t in mine}) == 2 and len({t.workflow for t in mine}) == 4


def test_operations_table_validates_and_rejects_bad_tables(tmp_path):
    assert len(load_compatibility(require_reviewed=False)) == 64
    table = json.loads(OPERATIONS_FILE.read_text())
    with pytest.raises(SystemExit):
        bad = tmp_path / "unreviewed.json"
        bad.write_text(json.dumps({**table, "reviewed": False}))
        load_compatibility(bad)
    first = next(iter(table["tasks"]))
    for broken in ({first: {"primary": "code", "allowed": ["extract"]}},       # primary not allowed
                   {first: {"primary": "dance", "allowed": ["dance"]}}):       # unknown operation
        path = tmp_path / "broken.json"
        path.write_text(json.dumps({**table, "reviewed": True, "tasks": {**table["tasks"], **broken}}))
        with pytest.raises(ValueError):
            load_compatibility(path)


def test_seeds_are_deterministic(tmp_path):
    write_seeds(make_seeds(**PILOT), tmp_path / "a.jsonl")
    write_seeds(make_seeds(**PILOT), tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_seed_coverage(pilot):
    segments = [g for s in pilot for g in s.segments]
    lengths = Counter(g.n_turns for g in segments)
    assert set(lengths) == set(range(1, 21)) and min(lengths.values()) >= 6
    assert min(Counter(g.task for g in segments).values()) >= 2 and len({g.task for g in segments}) == 64
    assert min(Counter(g.switch_in for g in segments if g.switch_in).values()) >= 30
    assert set(Counter(len(s.segments) for s in pilot)) == {1, 2, 3, 4, 5}
    assert max(sum(g.n_turns for g in s.segments) for s in pilot) <= MAX_SESSION_TURNS
    assert len({g.object_ref for g in segments}) == len(segments)


def test_seed_slots_are_consistent(pilot):
    tasks, compatibility = load_tasks(), load_compatibility(require_reviewed=False)
    for session in pilot:
        assert session.segments[0].switch_in is None
        for prev, cur in zip(session.segments, session.segments[1:]):
            if cur.switch_in == "same_task_new_object":
                assert cur.task == prev.task
            elif cur.switch_in == "same_industry_other_task":
                assert cur.industry == prev.industry and cur.task != prev.task
            else:
                assert cur.switch_in == "cross_industry" and cur.industry != prev.industry
        for g in session.segments:
            t = tasks[g.task]
            assert (g.industry, g.use_case, g.workflow) == (t.industry, t.use_case, t.workflow)
            assert g.operation in compatibility[g.task]["allowed"]
            assert len(g.followup_kinds) == g.n_turns - 1


def _fake_session(pilot):
    seed = next(s for s in pilot if len(s.segments) == 3)
    generation = {"session_id": seed.session_id, "segments": [
        {"status": "ok", "turns": [{"user": f"{g.segment_id} u{i}", "assistant": f"a{i}"} for i in range(g.n_turns)]}
        for g in seed.segments]}
    return Session(seed, generation)


def test_export_gold_is_true_exactly_at_segment_starts(pilot):
    session = _fake_session(pilot)
    rows = seg_rows(session, "eval")
    starts = set(session.gold_starts()) - {0}
    assert {r["k"] for r in rows if r["gold"] == "true"} == starts
    assert len(rows) == sum(g.n_turns for g in session.seed.segments) - 1
    for r in rows:
        assert len(r["conversation_so_far"]) == r["k"]               # full prefix from turn 0
        assert r["conversation_so_far"][0].startswith("[0] USER:")
    gold = [r["gold"] for r in tax_rows(session, "eval")]
    assert [g["task"] for g in gold] == [g.task for g in session.seed.segments]


def test_check_turns():
    names = category_names()
    ok = json.dumps({"turns": [{"user": "Here is invoice INV-1: ... Which lines lack a receipt?", "assistant": "Lines 2 and 4."}]})
    assert check_turns(ok, 1, names) == ([{"user": json.loads(ok)["turns"][0]["user"], "assistant": "Lines 2 and 4."}], [])
    assert check_turns(ok, 2, names)[0] is None                                   # wrong turn count
    assert check_turns("not json", 1, names)[0] is None
    leak = json.dumps({"turns": [{"user": "New task: please Summarize an encounter.", "assistant": "ok"}]})
    problems = check_turns(leak, 1, names)[1]
    assert any("new task" in p for p in problems) and any("category name" in p for p in problems)
    long_reply = json.dumps({"turns": [{"user": "Check this.", "assistant": "word " * 2000}]})
    assert check_turns(long_reply, 1, names)[1] == []                            # replies have no length cap


def test_check_turns_rejects_internal_ref():
    text = json.dumps({"turns": [{"user": "Review claim REF-80603 please.", "assistant": "Done."}]})
    assert any("internal seed id" in p for p in check_turns(text, 1, category_names())[1])


def test_every_operation_has_a_tie_break():
    from golden.taxonomy import OPERATION_TIE_BREAKS, load_operations
    assert set(OPERATION_TIE_BREAKS) == set(load_operations())


def test_seed_coverage_at_full_size():
    sessions = make_seeds(500, 2, min_per_length=40)
    segments = [g for s in sessions for g in s.segments]
    assert min(Counter(g.n_turns for g in segments).values()) >= 40
    assert min(Counter(g.task for g in segments).values()) >= 20
    assert max(sum(g.n_turns for g in s.segments) for s in sessions) <= MAX_SESSION_TURNS


def _segment(segment_id, opener, status="ok", attempts=1, problems=(), reply="Done, see the table above."):
    return {"segment_id": segment_id, "status": status, "attempts": attempts, "problems": list(problems),
            "cost_usd": 0.01, "turns": [{"user": opener, "assistant": reply}] if status == "ok" else None}


def test_quality_checks_warn_on_bad_data():
    from golden.checks import generation_checks, verification_checks
    opener = "Please compare invoice INV-204 for 3 pallets of steel bolts with the delivery note listing 2 pallets"
    bad = [{"session_id": "a", "segments": [
        _segment("a-g0", opener), _segment("a-g1", opener),                                 # near-duplicate openers
        _segment("a-g2", "Route this claim using the rules below ...", reply="As an AI, I cannot do that."),
        _segment("a-g3", "", status="generation_failed", attempts=3, problems=["turn 1: user says 'moving on'"] * 3),
    ]}]
    warned = {c.name for c in generation_checks(bad) if not c.ok}
    assert {"failed segments", "segments needing a retry", "top rejection reason",
            "near-duplicate openers", "assistant meta/refusal replies"} <= warned

    good = [{"session_id": "b", "segments": [_segment(f"b-g{i}", f"Unique request number {i} about item {i * 7} with its own"
                                                                  f" pasted table and figures {i * 13} {i * 17}") for i in range(5)]}]
    assert all(c.ok for c in generation_checks(good))

    verification = [{"boundaries_gold": [0, 4], "boundaries_pred": [0, 2], "boundaries_match": False,
                     "segments": [{"task_match": False, "operation_match": False}]}]
    assert not any(c.ok for c in verification_checks(verification))


def test_check_turns_survives_malformed_turns():
    for turns in ([{"user": "only a user message"}], ["a bare string"], [None]):
        result, problems = check_turns(json.dumps({"turns": turns}), 1, category_names())
        assert any("empty user or assistant" in p for p in problems)


def test_check_turns_enforces_length_and_shape():
    names = category_names()
    def problems(user, length, shape):
        return check_turns(json.dumps({"turns": [{"user": user, "assistant": "ok"}]}), 1, names, [length], [shape])[1]
    assert problems("x" * 50, 50, "prose") == []
    assert any("chars" in p for p in problems("x" * 400, 50, "prose"))
    assert any("pasted block" in p for p in problems("y" * 400, 400, "pasted"))
    assert problems("line one\nline two\n" + "z" * 380, 400, "pasted") == []
    assert any("prose" in p for p in problems("a\nb\nc\nd" + "w" * 100, 100, "prose"))


def test_surface_slots_hide_the_boundary(pilot_v2):
    from golden.seeds import LENGTH_BUCKETS, PASTED_SHARE
    segments = [g for s in pilot_v2 for g in s.segments]
    first = Counter(g.user_lengths[0] for g in segments)
    later = Counter(n for g in segments for n in g.user_lengths[1:])
    for bucket in LENGTH_BUCKETS:                               # same length distribution for first and later turns
        assert abs(first[bucket] / sum(first.values()) - later[bucket] / sum(later.values())) < 0.03
    for g in segments:
        counts = Counter(g.user_lengths)
        assert max(counts.values()) - min(counts.get(b, 0) for b in LENGTH_BUCKETS[: min(g.n_turns, 5)]) <= 1 \
            or g.n_turns < 5                                     # balanced within each segment
        assert len(g.user_lengths) == len(g.turn_shapes) == g.n_turns
        assert all(PASTED_SHARE[n] > 0 or shape == "prose" for n, shape in zip(g.user_lengths, g.turn_shapes))
        assert (g.transition_style is None) == (g.switch_in is None)
    kinds = Counter(k for g in segments for k in g.followup_kinds)
    assert abs(kinds["decoy"] / sum(kinds.values()) - 0.25) < 0.01


def test_prompt_renders_every_slot(pilot_v2):
    from golden.generate import build_prompt
    from golden.taxonomy import load_operations
    session = next(s for s in pilot_v2 if len(s.segments) > 1)
    segment = session.segments[1]
    prompt = build_prompt(session, segment, load_tasks()[segment.task], load_operations(),
                          [{"user": "earlier", "assistant": "reply"}])
    assert f"exactly {segment.n_turns} user turn" in prompt and "Turn 1 [USER ~" in prompt
    assert segment.object_ref not in prompt and "EARLIER IN THIS CHAT" in prompt     # the seed id never reaches the model


def test_shortcut_checks_catch_a_length_leak(pilot):
    from golden.checks import shortcut_checks

    def session(seed, opener_len, later_len):
        segments = [{"status": "ok", "turns": [{"user": ("o" if i == 0 else "f") * (opener_len if i == 0 else later_len)
                                                + f" {g.segment_id} {i}", "assistant": "ok"} for i in range(g.n_turns)]}
                    for g in seed.segments]
        return Session(seed, {"session_id": seed.session_id, "segments": segments})

    leaky = [session(s, 800, 100) for s in pilot]
    assert not next(c for c in shortcut_checks(leaky) if c.name == "shortcut: user turn length").ok
    balanced = [session(s, 300, 300) for s in pilot]
    assert next(c for c in shortcut_checks(balanced) if c.name == "shortcut: user turn length").ok


def test_export_keeps_long_turns_whole(pilot_v2):
    seed = next(s for s in pilot_v2 if len(s.segments) == 3)
    long_text = "word " * 300
    generation = {"session_id": seed.session_id, "segments": [
        {"status": "ok", "turns": [{"user": long_text, "assistant": long_text} for _ in range(g.n_turns)]}
        for g in seed.segments]}
    rows = seg_rows(Session(seed, generation), "eval")
    assert long_text.strip() in rows[-1]["new_user_request"] and long_text.strip() in rows[-1]["conversation_so_far"][0]
    assert {r["k"] for r in rows if r["gold"] == "true"} == set(Session(seed, generation).gold_starts()) - {0}
    assert all((r["transition_style"] is None) == (r["gold"] == "false") for r in rows)


def test_repair_targets_only_length_and_shape_problems(pilot_v2):
    from golden.generate import repair_request
    segment = next(g for s in pilot_v2 for g in s.segments if g.n_turns >= 3)
    turns = [{"user": "x", "assistant": "y"} for _ in range(segment.n_turns)]
    prompt = repair_request(turns, ["turn 2: user message has 1 chars, needs 25-70",
                                    "turn 3: user message must be a pasted block (line breaks)"], segment)
    assert prompt and "- turn 2:" in prompt and "- turn 3:" in prompt and "- turn 1:" not in prompt
    assert repair_request(turns, ["turn 2: mentions the category name 'X'"], segment) is None


def test_check_turns_accepts_raw_line_breaks_in_json():
    raw = '{"turns": [{"user": "rules:\n- a\n- b", "assistant": "ok"}]}'.replace("\\n", "\n")
    assert check_turns(raw, 1, category_names())[0] is not None


def test_network_failures_are_not_saved_as_done():
    from golden.generate import network_failure
    outage = {"segments": [{"status": "generation_failed", "problems": ["API error: ConnectError"] * 4}]}
    rejected = {"segments": [{"status": "generation_failed", "problems": ["API error: x", "turn 1: user message has 9 chars"]}]}
    assert network_failure(outage) and not network_failure(rejected)


def test_lead_ins_hide_the_boundary_and_are_enforced(pilot_v2):
    from golden.generate import starts_with
    from golden.seeds import LEAD_IN_POOLS
    segments = [g for s in pilot_v2 for g in s.segments]
    def kind(p): return next(k for k, pool in LEAD_IN_POOLS.items() if p in pool)
    at_boundary = Counter(kind(g.lead_ins[0]) for g in segments if g.switch_in)
    follow_up = Counter(kind(x) for g in segments for x in g.lead_ins[1:])
    for k in LEAD_IN_POOLS:                                   # same lead-in mix on both sides of a boundary
        assert abs(at_boundary[k] / sum(at_boundary.values()) - follow_up[k] / sum(follow_up.values())) < 0.03
    assert starts_with("Oh and, the vendor sent a second invoice", "oh and")
    assert not starts_with("The vendor sent a second invoice", "oh and")
    raw = json.dumps({"turns": [{"user": "The vendor sent this", "assistant": "ok"}]})
    assert any("must begin" in p for p in check_turns(raw, 1, category_names(), lead_ins=["one more thing:"])[1])
