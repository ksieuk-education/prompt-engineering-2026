from collections import Counter

import pytest

from agent_theatre.engine import Turn, perform, resolve
from agent_theatre.scenario import load_scenario
from agent_theatre.schema import Speech, ToolCall
from agent_theatre.stage_tools import quorum, recall_line
from test_rehearsal import RehearsalModel


def test_memory_can_find_an_older_line_by_its_actual_speaker():
    def turn(index, actor, line):
        return Turn(actor, 1, Speech(line=line, stage="", source="", vote="", to=""), index)
    history = [turn(1, "tony", "Проверим пропуск у двери."), turn(2, "barbie", "А если пропуск один?"),
               turn(3, "tony", "Кого ты ждёшь?")]
    assert recall_line(history, "tony", "пропуска").index == 1
    assert recall_line(history, "tony", "последняя реплика").index == 3
    assert recall_line(history, "barbie", "двери") is None


@pytest.mark.parametrize("scenario", ["extra", "lab404"])
def test_a_hypothetical_ballot_matches_the_real_terminal(scenario):
    story = load_scenario(scenario)
    cast = list(story.cast)
    for option, vote in [("challenge", "challenge"), ("nomination", f"select:{cast[0]}")]:
        for supporters in range(1, len(cast) + 1):
            ballot = {a: vote if i < supporters else "abstain" for i, a in enumerate(cast)}
            outcome, _ = resolve(story, ballot)
            assert (outcome == vote) == (supporters >= quorum(option, len(cast)))


def test_tool_results_arrive_before_intention_and_never_cast_a_vote():
    class Players(RehearsalModel):
        def ask(self, model, system, prompt, form, **options):
            if form is ToolCall:
                actions = options["choices"]["action"]
                if "search_archive" in actions:
                    return ToolCall(action="search_archive", query="датчик проверки", supporters=1)
                if "test_ballot:nomination" in actions:
                    return ToolCall(action="test_ballot:nomination", query="", supporters=2)
                return ToolCall(action=actions[-1], query="", supporters=1)
            if "Мысленный пример" in prompt:
                assert "Порог — 3. Голосов пока мало." in prompt
            return super().ask(model, system, prompt, form, **options)
    story = load_scenario("extra")
    events = []
    perform(story, Players(), {a: str(i % 2) for i, a in enumerate(story.cast)}, "p", 4, events.append)
    skills = Counter(e["tool"] for e in events if e["event"] in ("search", "skill"))
    assert set(skills) == {"recall_line", "test_ballot", "search_archive"}
    played = {}
    for event in events:
        if event["event"] == "skill" and event["tool"] == "recall_line":
            result = event["result"]
            assert played[result["turn"]]["line"] == result["text"]
            assert played[result["turn"]]["actor"] == event["arguments"]["speaker"]
        elif event["event"] == "turn":
            anchor = event["reply_to"]
            if anchor is not None:
                assert anchor in played and played[anchor]["scene"] == event["scene"]
            played[event["index"]] = event
    assert events[-1]["outcome"] == "challenge"
