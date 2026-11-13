import pytest

from agent_theatre.audio import casting, cues_for
from agent_theatre.language import stage_direction, wrong_gender
from agent_theatre.scenario import load_scenario


@pytest.mark.parametrize(("who", "line", "wrong"), [
    ("tony", "Я тогда ошиблась.", True), ("barbie", "Я не уверен.", True),
    ("hermione", "Я решила, что Чендлер был прав.", False),
    ("tony", "Она сказала «я устала». Я верно услышал?", False),
    ("tony", "Барби, ты готов?", True), ("chandler", "Гермиона, ты права.", False),
    ("barbie", "Я врач, а он сказал, что дверь открыта.", False),
    ("barbie", "Не уверен, что табло понимает вопрос.", True),
    ("tony", "Готова?", False),
])
def test_agreement_is_about_the_right_person(who, line, wrong):
    story = load_scenario("extra")
    assert wrong_gender(line, story.cast[who], list(story.cast.values())) is wrong


def test_cast_snapshot_keeps_voices_after_display_names_change():
    start = {"event": "start", "cast": {"x": {"name": "Алекс", "gender": "female", "speaker": "xenia"}}}
    events = [start, {"event": "turn", "actor": "x", "stage": "Поправляю очки.", "line": "Я готова."},
              {"event": "ending", "curtain": "Занавес."}]
    cues = list(cues_for(events))
    assert [(c.speaker, c.text) for c in cues] == [
        ("baya", "Алекс."), ("baya", "Поправляет очки."), ("xenia", "Я готова."), ("baya", "Занавес.")]
    start["cast"]["x"]["speaker"] = "aidar"
    with pytest.raises(ValueError, match="пол роли"):
        casting(events)


def test_narrator_preserves_quotes_and_handles_an_irregular_verb():
    actor = load_scenario("lab404").cast["professor"]
    assert stage_direction("Я беру лист и говорю «Я жду».", actor) == "Профессор берёт лист и говорит «Я жду»."


def test_a_second_person_predicate_follows_the_known_addressee():
    cast = load_scenario("extra").cast
    assert wrong_gender("Ты решил за нас?", cast["tony"], list(cast.values()), cast["barbie"])
    assert not wrong_gender("Чендлер, ты решил за нас?", cast["tony"], list(cast.values()), cast["barbie"])
    assert not wrong_gender("Дверь закрыта. Барби, ты видела ещё что-то?", cast["hermione"],
                            list(cast.values()), cast["chandler"])
