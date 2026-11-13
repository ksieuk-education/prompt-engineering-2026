from copy import deepcopy
from pathlib import Path
import json

import pytest

from agent_theatre.analysis import read_trace
from agent_theatre.audio import cues_for
from agent_theatre.book_scene import Choice, check_comment, printed_quote, shorten_comment
from agent_theatre.cli import markdown
from agent_theatre.engine import perform, resolve
from agent_theatre.evidence import alternatives, benchmark, check_archive, compare_runs, dossier
from agent_theatre.library import DIRECTORY, Library
from agent_theatre.scenario import load_scenario
from agent_theatre.schema import Delivery, Lookup
from test_rehearsal import RehearsalModel


@pytest.fixture(scope="module")
def lib():
    return Library()


def test_full_works_have_citable_text_and_bounded_context(lib):
    assert len(lib.entries) == len(lib.positions) > 4000
    assert sum(len(p.text.split()) for p in lib.entries) > 40_000
    assert sum(len({p.act for p in lib.entries if p.book == key}) for key in lib.books) == 13
    hit = lib.search("Фирс заперто уехали", book="garden")[0]
    card = lib.card(hit)
    assert card["speaker"] == "Фирс" and "Заперто" in card["quote"]
    lo, hi = card["lines"]
    file = DIRECTORY / lib.books[hit.book].file
    assert "Заперто" in " ".join(file.read_text().splitlines()[lo-1:hi])
    assert card["url"].endswith(f"#L{lo}-L{hi}")
    context = lib.context(hit.ref, radius=3)
    assert any("Про меня забыли" in p.text for p in context)
    assert all((p.act, p.section, p.book) == (hit.act, hit.section, hit.book) for p in context)
    assert lib.context("garden:1", radius=5)[0].ref == "garden:1"
    with pytest.raises(ValueError):
        lib.context("garden:1", radius=200)
    assert lib.search("купец, монета, орёл и решётка", book="dowry")
    assert lib.search("Хлестаков вычисляет квантовый кубит токенизатором") == []
    assert lib.search("Фирс взятка", book="garden") == []
    assert lib.search("и /../../", book="garden") == []


def test_checksum_checked_before_indexing(tmp_path):
    catalog = json.loads((DIRECTORY / "catalog.json").read_text())
    catalog["works"] = catalog["works"][:1]
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / catalog["works"][0]["file"]).write_bytes(b"<TEI/>")
    with pytest.raises(ValueError, match="сумма"):
        Library(tmp_path)


class Readers(RehearsalModel):
    def __init__(self, refuse=False):
        self.refuse = refuse

    def ask(self, model, system, prompt, form, **options):
        if form is Lookup:
            return Lookup(query="Вожеватов монета орёл решетка" if "Ларис" in prompt else "Фирс заперто уехали")
        if form is Choice:
            self.checking_context = "Сейчас проверь чужое чтение." in prompt
            return Choice(ref="" if self.refuse else options["choices"]["ref"][0], stance="уточнить")
        if form is Delivery and "Детали для разговора" in prompt:
            line = "Вы согласны на жребий?" if "Ларис" in prompt else "Фирс остался у двери. Мы тоже решим за другого?"
            if self.checking_context:
                line = "Спросим саму Ларису про монету?" if "Ларис" in prompt else "Кто спросил Фирса, хотел ли он остаться?"
            return Delivery(line=line, stage="")
        return super().ask(model, system, prompt, form, **options)


def show(lib, refuse=False, model=None):
    story = load_scenario("extra")
    events = [{"event": "start", "scenario": story.key, "seed": 1, "library_revision": lib.revision}]
    perform(story, model or Readers(refuse), {key: f"m{i%2}" for i,key in enumerate(story.cast)},
            "planner", 4, events.append, library=lib)
    return events


def test_independent_reader_checks_and_performs_a_verifiable_passage(lib):
    events = show(lib)
    refs = check_archive(events, lib)
    assert [e["kind"] for e in refs] == ["read","check","read","check"]
    assert all(a["actor"] != b["actor"] and b["origin"] == a["card"]["ref"]
               for a,b in (refs[:2],refs[2:]))
    assert len([c for c in cues_for(events) if c.name.endswith(" читает")]) == 4
    assert all(printed_quote(e["card"]) in markdown("Пьеса", events) for e in refs)
    assert "Ближайшая альтернативная развязка" in dossier(events, lib)
    broken = deepcopy(events)
    next(e for e in broken if e["event"]=="book_search")["hits"][0]["quote"] = "Несуществующая цитата."
    with pytest.raises(ValueError, match="Искажена"):
        check_archive(broken, lib)
    broken = deepcopy(events)
    next(e for e in broken if e["event"]=="book_context")["actor"] = refs[0]["actor"]
    with pytest.raises(ValueError, match="другого актёра"):
        check_archive(broken, lib)


def test_no_suitable_evidence_produces_no_public_quote(lib):
    events = show(lib, refuse=True)
    assert [e["attempt"] for e in events if e["event"]=="book_search"] == [1,2,1,2]
    assert check_archive(events, lib) == []
    assert not any(e["event"]=="book_context" for e in events)


def test_a_verbatim_source_does_not_validate_an_invented_comment(lib):
    card = lib.card(lib.search("жребий монета орёл", book="dowry")[0])
    story = load_scenario("extra")
    citation = {"card":card, **story.library_tasks[0]}
    cast = " ".join(a.name + " " + a.stake for a in story.cast.values())
    for line in ("Элизабет велела бросить жребий.", "Войтовский тоже брал монету.",
                 "Монета значит addressed to Barbie.", "Монета. «Дайте все билеты мне»."):
        assert check_comment(Delivery(line=line,stage=""), citation, cast).issue == "book"
    assert not check_comment(Delivery(line="Барби, спросим у каждого, согласен ли он на жребий?",stage=""),
                             citation, cast).issue
    assert not check_comment(Delivery(line="Монета у Вожеватова. Ты бы согласилась на такой жребий?", stage=""),
                             citation, cast).issue


def test_counterfactual_finds_a_unique_pivotal_vote_and_preserves_the_record():
    story = load_scenario("lab404")
    votes = {k: ("abstain" if k=="chandler" else "challenge") for k in story.cast}
    original = votes.copy()
    candidates = alternatives(story,votes)
    assert candidates["challenge"] == {"distance":1,"ways":1,"example":{"chandler":"challenge"}}
    for outcome, value in candidates.items():
        assert resolve(story, votes | value["example"])[0] == outcome
        assert len(value["example"]) == value["distance"]
    assert votes == original
    assert "unresolved" in alternatives(story,dict.fromkeys(story.cast,"challenge"))


def test_queries_are_ranked_without_passing_the_expected_book(lib):
    result = benchmark(lib)
    assert result["positive"]==15 and result["negative"]==3
    assert result["scores"]["bm25"]["empty"]==3
    assert result["scores"]["bm25"]["top3"]>=result["scores"]["bm25"]["top1"]
    row=next(r for r in result["queries"] if "шубе" in r["query"])
    assert row["bm25"]["rank"] and lib.get(row["bm25"]["hits"][row["bm25"]["rank"]-1]).book=="garden"


def test_actor_can_withdraw_without_leaking_an_unplayed_passage(lib):
    class MistakenReader(Readers):
        def ask(self, model, system, prompt, form, **options):
            if form is Delivery and "Детали для разговора" in prompt:
                return Delivery(line="Элизабет требует бросить монету. Вы согласны?", stage="")
            return super().ask(model, system, prompt, form, **options)

    events = show(lib, model=MistakenReader())
    assert [e["index"] for e in events if e["event"] == "book_withdrawn"] == [5, 9]
    assert len([e for e in events if e["event"] == "turn"]) == 16
    assert not check_archive(events, lib)
    assert not any(e["event"] == "book_context" for e in events)
    assert "остался за кулисами" in dossier(events, lib)


def test_shortened_take_uses_an_intact_sentence_and_accepts_declined_names(lib):
    story = load_scenario("lab404")
    task = story.library_tasks[0]
    card = lib.card(lib.search("Хлестаков", book="inspector")[0])
    citation = {"card": card, **task}
    sentence = "Вы бы доверили Хлестакову эту проверку?"
    raw = Delivery(line="Мне трудно ответить на это сразу. " * 7 + sentence, stage="")
    brief = shorten_comment(raw, citation)
    assert brief.line == sentence
    assert not check_comment(brief, citation, "").issue
    garden = {"card": lib.card(lib.get("garden:1468")), **load_scenario("extra").library_tasks[1]}
    named_book = Delivery(line="В «Вишневом саде» Фирс остался в запертом доме. Мы это заметили?", stage="")
    assert not check_comment(named_book, garden, "").issue


def test_book_editor_keeps_a_complete_new_answer_instead_of_an_echo_or_ellipsis(lib):
    story = load_scenario("extra")
    citation = {"card": lib.card(lib.get("dowry:1572")), **story.library_tasks[0]}
    heard = "Кто вообще спрашивал Ларису, согласна ли она на жребий?"
    broken = "«Что мы решим с учетом её..."
    reply = "Пусть она сама решит, согласна ли бросать монету."
    long_take = " ".join([heard, broken, reply, "Мы ещё можем поговорить об этом за столом. " * 4])
    cut = shorten_comment(Delivery(line=long_take, stage=""), citation, previous=heard)
    assert cut.line == reply
    assert check_comment(Delivery(line=broken, stage=""), citation, "").issue == "book"
    quoted = Delivery(line=f"«{heard}»", stage="")
    assert check_comment(quoted, citation, "", previous=heard).issue == "book"
    assert not check_comment(cut, citation, "", previous=heard).issue


def test_reading_falls_back_only_to_a_verified_spoken_turn(lib):
    from agent_theatre.reading import rehearse_scene

    story = load_scenario("extra")
    events = read_trace(Path(__file__).resolve().parents[1] / "examples/library/extra")
    scene = next(e for e in events if e["event"] == "scene" and e["scene"] == 3)
    turn = next(e for e in events if e["event"] == "turn" and e["index"] == 9)
    citation = next(e for e in events if e["event"] == "citation" and e["index"] == 9)

    class FaultyDramaturg:
        def ask(self, model, system, prompt, form, **options):
            return form.model_validate({"turn_9_chandler": {"реплика": "Элизабет сама заперла дом.", "жест": ""}})

    kept = {}
    result = rehearse_scene(story, scene, [turn], {}, [], FaultyDramaturg(), "model",
                            citations={9: citation}, kept_turns=kept)
    assert result[0]["line"] == turn["line"] and "Элизабет" in kept[9]


def test_paired_experiment_checks_setup_and_counts_only_published_quotes(lib):
    root = Path(__file__).resolve().parents[1] / "examples" / "library"
    a = read_trace(root / "control")
    b = read_trace(root / "extra")
    comparison = compare_runs(a, b, lib)
    assert comparison["control"]["citations"] == 0
    assert comparison["with_books"]["citations"] == 4
    assert set(comparison["changed_votes"]) == {"Барби"}
    with pytest.raises(ValueError, match="контрольным"):
        compare_runs(b, a, lib)
    with pytest.raises(ValueError, match="условия"):
        compare_runs(a, [b[0] | {"seed": -1}, *b[1:]], lib)
