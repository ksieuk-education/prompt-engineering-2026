from copy import deepcopy
from hashlib import sha256
import json

import pytest

from agent_theatre.reading import SceneReading, apply_reading, read_performance, scene_take
from agent_theatre.scenario import load_scenario


def test_reading_preserves_speaker_ballot_and_tool_memory(tmp_path):
    story = load_scenario("extra")
    turn = {"event": "turn", "actor": "barbie", "index": 3, "line": "Я воздержусь. Почему я должна решать?",
            "stage": "", "to": "tony", "vote": "abstain"}
    tool = {"event": "skill", "tool": "recall_line", "result": {"turn": 1, "text": "Кто это решил?"}}
    take = SceneReading.model_validate({"turns": [
        {"index": 3, "реплика": "3. Барби: «Я воздержусь. Это решать не мне.»", "жест": "(Отступает.)"}]})
    events = [tool, turn, {"event": "ending", "outcome": "unresolved"}]
    original = deepcopy(events)
    reading = scene_take(story, [turn], take)
    edited = apply_reading(events, reading)
    assert edited[1] == turn | {"line": "Я воздержусь. Это решать не мне.", "stage": "Отступает."}
    assert edited[0] == tool and edited[2] == events[2] and events == original
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert read_performance(tmp_path) == events
    (tmp_path / "reading.json").write_text(json.dumps({"trace_sha256": sha256(trace.read_bytes()).hexdigest(),
                                                       "turns": reading}))
    assert read_performance(tmp_path) == edited
    trace.write_text(trace.read_text().replace("unresolved", "challenge"))
    with pytest.raises(ValueError, match="изменился"):
        read_performance(tmp_path)


def test_reading_refuses_swapped_turns_and_wrong_gender():
    turn = {"event": "turn", "actor": "hermione", "index": 2, "to": "", "vote": ""}
    take = SceneReading.model_validate({"turns": [{"index": 3, "реплика": "Я устал.", "жест": ""}]})
    with pytest.raises(ValueError, match="номера"):
        scene_take(load_scenario("lab404"), [turn], take)
    take.turns[0].index = 2
    with pytest.raises(ValueError, match="gender"):
        scene_take(load_scenario("lab404"), [turn], take)


def test_published_edits_cannot_replace_a_registered_ballot_or_another_reading(tmp_path):
    from agent_theatre.reading import publish_edition

    events = [{"event": "start", "scenario": "extra"},
              {"event": "turn", "index": 1, "name": "Барби", "actor": "barbie", "to": "",
               "line": "Я воздержусь. Это мой выбор.", "stage": "", "vote": "abstain"},
              {"event": "ending", "curtain": "Занавес."}]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(e) + "\n" for e in events))
    reading = tmp_path / "reading.json"
    reading.write_text(json.dumps({"trace_sha256": sha256(trace.read_bytes()).hexdigest(),
                                  "turns": [{k: events[1][k] for k in ("index","line","stage")}]}))
    with pytest.raises(ValueError, match="зарегистрировано"):
        publish_edition(tmp_path, {"1": {"line": "Я остаюсь здесь. Остальные могут идти."}}, "Вычитка")
    publish_edition(tmp_path, {"1": {"line": "Выбирать за вас я не стану."}}, "Вычитка")
    assert read_performance(tmp_path)[1]["line"] == "Я воздержусь. Выбирать за вас я не стану."
    assert read_performance(tmp_path, edition=False)[1]["line"] == events[1]["line"]
    assert (tmp_path / "edition.md").exists()
    reading.write_text(reading.read_text() + "\n")
    with pytest.raises(ValueError, match="другой читке"):
        read_performance(tmp_path)
