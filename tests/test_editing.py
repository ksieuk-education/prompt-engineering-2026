from pathlib import Path

import pytest

from agent_theatre.cli import read_saved_plan
from agent_theatre.engine import Turn, named_addressee, pending_question, perform
from agent_theatre.llm import GenerationError
from agent_theatre.rehearsal import deliver, draft_issues
from agent_theatre.scenario import load_scenario
from agent_theatre.schema import Delivery, Intention, Plan, Review, Speech
from test_rehearsal import RehearsalModel


EXTRA = load_scenario("extra")


@pytest.mark.parametrize(("actor", "text", "expected"), [
    ("tony", "Мой брат умер, зная, что я вернусь.", ["family"]),
    ("barbie", "У меня была младшая сестра.", ["family"]),
    ("hermione", "Мои родители этого не помнят.", []),
    ("chandler", "Меня ждут моя жена и мои дети.", []),
    ("tony", "Гермиона, твои родители тебя ждут?", []),
    ("tony", "Ты сказал «мой брат», я верно услышал?", []),
    ("tony", "Проверим дверь. (Поднимаю руку.)", ["aside"]),
])
def test_feedback_distinguishes_the_speakers_own_story(actor, text, expected):
    assert draft_issues(Delivery(line=text, stage=""), EXTRA.cast[actor], [], False) == expected


def test_self_vote_allows_acknowledging_the_cost():
    actor = EXTRA.cast["chandler"]
    conflict = Delivery(line="Дома ждут Моника и дети. Я не могу их оставить.", stage="")
    cost = Delivery(line="Мне страшно не увидеть Монику, но за вас я выбирать не хочу.", stage="")
    assert draft_issues(conflict, actor, [], True) == ["vote"]
    assert draft_issues(conflict, actor, [], False) == []
    assert draft_issues(cost, actor, [], True) == []


def test_repetition_needs_more_than_a_shared_prop():
    prior = ["Мистер Старк, если часы единственный индикатор, то как мы проверим его точность?"]
    repeated = Delivery(line="Мистер Старк, если часы единственный индикатор, то как мы узнаем его цену?", stage="")
    reply = Delivery(line="Часы забери себе, а мне скажи, ты сам согласишься остаться?", stage="")
    assert draft_issues(repeated, EXTRA.cast["hermione"], prior, False) == ["repeat"]
    assert draft_issues(reply, EXTRA.cast["chandler"], prior, False) == []
    assert draft_issues(reply, EXTRA.cast["chandler"], prior, False, True) == ["clock"]


def test_interrupted_question_and_vocative():
    def turn(who, to, line):
        return Turn(who, 2, Speech(line=line, to=to, stage="", source="", vote=""))
    asked = turn("tony", "barbie", "Ты согласна рискнуть, Барби?")
    history = [asked, turn("hermione", "", "Послушайте.")]
    assert named_addressee(asked.speech.line, EXTRA, list(EXTRA.cast)) == "barbie"
    assert named_addressee("Я выбираю Барби.", EXTRA, list(EXTRA.cast)) is None
    assert pending_question("barbie", history) == asked
    history.append(turn("barbie", "", "За всех решать не буду."))
    assert pending_question("barbie", history) is None


def test_retake_precedes_public_history_and_a_saved_plan_skips_the_planner():
    class RetakeModel(RehearsalModel):
        def __init__(self):
            self.drafted = False

        def ask(self, model, system, prompt, form, **options):
            assert form is not Plan
            if form is Intention:
                assert "Мой брат умер" not in prompt
            if form is Delivery and not self.drafted:
                self.drafted = True
                return Delivery(line="Мой брат умер.", stage="")
            return super().ask(model, system, prompt, form, **options)

    run = Path(__file__).resolve().parents[1] / "examples/extra"
    plan, author = read_saved_plan(run, "extra")
    with pytest.raises(ValueError, match="другой постановке"):
        read_saved_plan(run, "lab404")
    events = []
    models = {key: str(i % 2) for i, key in enumerate(EXTRA.cast)}
    perform(EXTRA, RetakeModel(), models, author, 4, events.append, plan=plan)
    assert events[0]["reused"] and events[0]["model"] == author
    take = next(e for e in events if e["event"] == "retake")
    turn = next(e for e in events if e["event"] == "turn")
    assert take["issues"] == ["family"] and turn["actor"] == take["actor"]
    assert turn["index"] == take["index"] and turn["edited"] and turn["quality_issues"] == []


def test_repeated_hard_violation_stops_after_three_drafts():
    class Stubborn:
        def ask(self, *args, **kwargs):
            return Delivery(line="Мой брат всё решил.", stage="")
    records = []
    with pytest.raises(GenerationError, match="трёх дублей"):
        deliver(Stubborn(), "m", "r", "p", EXTRA.cast["tony"], [], False, False, records.append)
    assert [r["take"] for r in records] == [1, 2, 3]


def test_semantic_editor_can_send_an_unlisted_memory_back_to_the_same_actor():
    class ActorModel:
        def ask(self, model, system, prompt, form, **kwargs):
            assert model == "actor-model" and form is Delivery
            if "Чистый лист" in prompt:
                assert "Париж" not in prompt
                return Delivery(line="Я уже уходила из мира, где всё решили за меня.", stage="")
            return Delivery(line="Я помню, как упала с моста в Париже.", stage="")
    def review(words):
        if "Париже" in words.line:
            return Review(issue="biography", note="Замечание редактора о Париже: используй факт о своём мире.")
        return Review(issue="", note="")
    records = []
    line, edited, remaining = deliver(ActorModel(), "actor-model", "Барби", "Ход", EXTRA.cast["barbie"],
                                      [], False, False, records.append, review=review, repair_prompt="Чистый лист")
    assert edited and "решили за меня" in line.line and remaining == []
    assert records[0]["issues"] == ["biography"] and "о Париже" in records[0]["note"]


def test_editor_must_ground_a_contradiction_in_words_actually_spoken():
    from agent_theatre.rehearsal import review_line
    from agent_theatre.schema import EditorialCheck

    class Editor:
        def __init__(self, quote):
            self.quote = quote

        def ask(self, model, system, prompt, form, *, choices, budget):
            assert form is EditorialCheck and "reply" in choices["issue"]
            return EditorialCheck(issue="logic", excerpt="ты выбрал Барби", anchor=self.quote, note="Сверь имя.")

    partner = "Я выбрал себя. Что скажешь?"
    response = Delivery(line="Значит, ты выбрал Барби.", stage="")
    def check(quote):
        return review_line(Editor(quote), "m", EXTRA.cast["tony"], partner, "", response,
                           addressed_line=partner, final=False)
    assert check("Я выбрал себя").issue == "logic"
    assert check("Я выбрал Гермиону").issue == ""


def test_verbatim_input_echo_is_removed_before_grammar_check():
    prompt = "Я не уверен, что это протокол."

    class Echoing:
        def ask(self, *args, **kwargs):
            return Delivery(line=prompt + " Кто оставил строку пустой?", stage="")

    words, edited, _ = deliver(Echoing(), "m", "роль", "ход", EXTRA.cast["hermione"], [], False,
                               False, lambda _: None, addressed_line=prompt)
    assert words.line == "Кто оставил строку пустой?" and not edited
