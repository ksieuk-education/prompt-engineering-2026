from collections import Counter

import pytest
from pydantic import ValidationError

from agent_theatre.archive import Archive
from agent_theatre.cli import markdown
from agent_theatre.engine import Turn, ballot_phrase, choose_speaker, named_addressee, perform, resolve
from agent_theatre.scenario import SCENARIOS, load_scenario
from agent_theatre.schema import Delivery, Intention, Plan, ScenePlan, Speech, ToolCall


class RehearsalModel:
    """Прогоняет протокол без сети. Тексты спектаклей для просмотра создаются реальными LLM."""

    def ask(self, model, system, prompt, form, *, choices=None, budget=500):
        if form is Plan:
            return Plan(scenes=[ScenePlan(title=str(n), opening="Вспышка.", objective="Спор.")
                                for n in range(4)])
        if form is ToolCall:
            action = "search_archive" if "search_archive" in choices["action"] else ""
            return ToolCall(action=action, query="проверка протокол кнопка ограничение", supporters=1)
        if form is Intention:
            return Intention(intent="Уточнить условия", to=choices["to"][-1],
                             source=choices["source"][-1], vote=choices["vote"][0])
        assert form is Delivery and ".txt" not in prompt
        return Delivery(line="Я помню условия.", stage="")


@pytest.mark.parametrize("key", SCENARIOS)
def test_full_run_delivers_four_distinct_ballots_and_traceable_evidence(key):
    story = load_scenario(key)
    events = []
    models = {actor: f"model-{i % 2}" for i, actor in enumerate(story.cast)}
    perform(story, RehearsalModel(), models, "planner", 5, events.append)
    turns = [e for e in events if e["event"] == "turn"]
    assert events[0]["event"] == "plan"
    assert Counter(t["scene"] for t in turns) == {1: 5, 2: 5, 3: 5, 4: 4}
    assert {t["actor"] for t in turns if t["scene"] == 1} == set(story.cast)
    assert set(events[-1]["votes"]) == set(story.cast)
    assert events[-1]["outcome"] == "challenge"
    assert {t["model"] for t in turns} == set(models.values())
    assert all(a["actor"] != b["actor"] for a, b in zip(turns, turns[1:]))
    found = {key: set() for key in story.cast}
    for event in events:
        if event["event"] == "search":
            for hit in event["hits"]:
                lines = (story.archive / hit["file"]).read_text().splitlines()
                assert lines[hit["line"] - 1] == hit["text"]
                found[event["actor"]].add(f"{hit['file']}:{hit['line']}")
        elif event["event"] == "turn" and event["source"]:
            assert event["source"] in found[event["actor"]]
    assert any(t["source"] for t in turns)
    script = markdown(story.title, events)
    assert script.count("На табло у имени") == 4
    assert ".txt" not in script


def test_archive_misses_and_different_worlds():
    extra = Archive(load_scenario("extra").archive)
    lab = Archive(load_scenario("lab404").archive)
    assert extra.search("красную кнопку")[0].file == "button.txt"
    assert lab.search("проверка памяти")[0].file == "memory.txt"
    assert extra.search("../../несуществующий") == []
    assert extra.search("воспоминания") == []


def test_direct_address_and_starvation():
    cast = list("abcd")
    def turn(who, to):
        return Turn(who, 1, Speech(line="Ответь.", to=to, source="", stage="", vote=""))
    assert choose_speaker(cast, [turn("a", "d")], cast)[0] == "d"
    history = [turn("c", ""), turn("d", ""), turn("a", "b"), turn("b", "a"),
               turn("a", "b"), turn("b", "a")]
    assert choose_speaker(cast, history, cast)[0] == "c"


def test_different_votes_really_change_the_ending():
    story = load_scenario("extra")
    votes = dict.fromkeys(story.cast, "select:tony")
    votes["tony"] = "challenge"
    assert resolve(story, votes) == ("select:tony", story.nomination.format(name="Тони Старк"))
    votes["barbie"] = "abstain"
    assert resolve(story, votes)[0] == "unresolved"
    votes.pop("chandler")
    with pytest.raises(ValueError):
        resolve(story, votes)


@pytest.mark.parametrize("line", ["Смотрите, source: rules.txt:2", "Гермиона, 这是一个按钮。"])
def test_backstage_metadata_and_language_switch_get_a_retake(line):
    with pytest.raises(ValidationError):
        Delivery(line=line, stage="")


def test_extraneous_english_direction_does_not_lose_the_spoken_turn():
    line = "Можно приоткрыть дверь, пока все ещё в комнате?"
    assert Delivery(line=line, stage="University Lecture Hall").model_dump() == {"line": line, "stage": ""}


def test_spoken_address_and_self_nomination_depend_on_the_actual_scene():
    extra = load_scenario("extra")
    lab = load_scenario("lab404")
    assert named_addressee("Мистер Старк, кто решил за остальных?", extra, ["tony", "barbie"]) == "tony"
    assert ballot_phrase(extra, "barbie", "select:barbie") == "Я остаюсь здесь."
    assert ballot_phrase(lab, "barbie", "select:barbie") == "Я называю себя."
