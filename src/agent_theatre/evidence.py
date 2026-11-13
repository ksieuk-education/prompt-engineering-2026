"""Проверка книжных ссылок, небольшая разметка поиска и контрфактический разбор голосов."""
from itertools import product
import json

from .archive import terms
from .book_scene import printed_quote
from .engine import ballot_phrase, resolve
from .library import DIRECTORY, Library
from .scenario import Scenario, load_scenario


def benchmark(lib: Library) -> dict:
    cases = json.loads((DIRECTORY / "queries.json").read_text(encoding="utf-8"))
    simple = [terms(p.speaker + " " + p.text) for p in lib.entries]
    rows = []
    for case in cases:
        wanted = terms(case["query"])
        scores = [(len(wanted & s), i) for i, s in enumerate(simple) if wanted & s]
        scores.sort(key=lambda item: (-item[0], item[1]))
        result = {"query": case["query"], "answerable": bool(case["lines"])}
        methods = {"prefix": [lib.entries[i] for _, i in scores[:3]], "bm25": lib.search(case["query"])}
        for method, hits in methods.items():
            rank = next((k for k, p in enumerate(hits, 1) if p.book == case["book"] and any(
                p.first_line <= end and start <= p.last_line for start, end in case["lines"])), None)
            result[method] = {"rank": rank, "hits": [p.ref for p in hits]}
        rows.append(result)
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    metrics = {}
    for name in ("prefix", "bm25"):
        ranks = [r[name]["rank"] for r in positive]
        metrics[name] = {"top1": ranks.count(1), "top3": sum(r is not None for r in ranks),
                         "mrr3": sum(1/r for r in ranks if r)/len(positive),
                         "empty": sum(not r[name]["hits"] for r in negative)}
    return {"positive": len(positive), "negative": len(negative), "scores": metrics, "queries": rows}


def alternatives(story: Scenario, votes: dict[str, str]) -> dict:
    """Минимальное расстояние до каждого другого финала, в числе изменённых бюллетеней."""
    actual = resolve(story, votes)[0]
    actors = list(story.cast)
    options = ["abstain", "challenge", *(f"select:{actor}" for actor in actors)]
    best = {}
    for values in product(options, repeat=len(actors)):
        candidate = dict(zip(actors, values))
        outcome = resolve(story, candidate)[0]
        if outcome == actual:
            continue
        changes = {key: vote for key, vote in candidate.items() if votes[key] != vote}
        distance = len(changes)
        if outcome not in best or distance < best[outcome]["distance"]:
            best[outcome] = {"distance": distance, "ways": 0, "example": changes}
        if distance == best[outcome]["distance"]:
            best[outcome]["ways"] += 1
    return best


def check_archive(events: list[dict], lib: Library) -> list[dict]:
    """Проверяем адрес, текст, автора, порядок открытия и связь с реальным ходом."""
    if not events or events[-1]["event"] != "ending":
        raise ValueError("Нужен завершённый журнал спектакля")
    selected, public, turns = {}, {}, {}
    for e in events:
        kind = e["event"]
        if kind == "turn":
            turns[e["index"]] = e
        if kind == "book_withdrawn":
            selected.pop((e["index"], e["actor"]), None)
        if kind not in {"book_search", "book_context", "citation"}:
            continue
        if events[0].get("library_revision") != lib.revision:
            raise ValueError("Журнал относится к другой редакции корпуса")
        key = (e["index"], e["actor"])
        if kind == "citation":
            if (e["index"] in public or e["index"] in turns
                    or selected.get(key) != (e["kind"], e["card"], e.get("origin"), e.get("origin_turn"))):
                raise ValueError("Для публичной цитаты нет соответствующего выбора актёра")
            public[e["index"]] = e
            continue
        cards = e["hits"]
        for card in cards:
            p = lib.get(card["ref"])
            identity = lib.card(p)
            stable = ("ref", "book", "author", "title", "speaker", "act", "section", "lines", "url")
            quote = card["quote"]
            if (any(identity[k] != card.get(k) for k in stable) or not quote or quote not in p.text
                    or not 1 <= len(quote.split()) <= 20
                    or "cut_before" in card and card["cut_before"] != (not p.text.startswith(quote))
                    or "cut_after" in card and card["cut_after"] != (not p.text.endswith(quote))):
                raise ValueError(f"Искажена цитата или адрес фрагмента {card['ref']}")
        if kind == "book_context":
            prior = public.get(e["origin_turn"])
            if (not prior or e["origin_turn"] not in turns or e["origin_turn"] >= e["index"] or prior["actor"] == e["actor"]
                    or prior["card"]["ref"] != e["origin"] or prior["scene"] != e["scene"]):
                raise ValueError("Отсутствует ранее опубликованная цитата другого актёра")
            if [c["ref"] for c in cards] != [p.ref for p in lib.context(e["origin"], radius=3)]:
                raise ValueError("Открыт посторонний контекст")
            if e["selected"] == e["origin"]:
                raise ValueError("Для проверки нужна соседняя строка")
        elif any(c["book"] != e["book"] for c in cards):
            raise ValueError("Поиск вернул другую книгу")
        if e["selected"]:
            matches = [c for c in cards if c["ref"] == e["selected"]]
            if len(matches) != 1:
                raise ValueError("Выбран отсутствующий или неоднозначный фрагмент")
            selected[key] = ("read" if kind == "book_search" else "check", matches[0],
                             e.get("origin"), e.get("origin_turn"))
    for index, citation in public.items():
        turn = turns.get(index, {})
        if any(turn.get(field) != citation[field] for field in ("actor", "scene", "name")):
            raise ValueError("Цитата не связана с принятым ходом")
    return list(public.values())


def compare_runs(control: list[dict], with_books: list[dict], lib: Library) -> dict:
    """Парный опыт требует совпадения актёров, плана и условий генерации."""
    if not control or not with_books or any(run[-1]["event"] != "ending" for run in (control, with_books)):
        raise ValueError("Для сравнения нужны два завершённых журнала.")
    settings = ("scenario", "cast", "models", "seed", "editor", "planner", "turns_per_scene")
    if any(control[0].get(k) != with_books[0].get(k) for k in settings):
        raise ValueError("Различаются условия постановок.")
    plans = [next((e["scenes"] for e in run if e["event"] == "plan"), None) for run in (control, with_books)]
    if plans[0] is None or plans[0] != plans[1]:
        raise ValueError("Для парного опыта нужен один макроплан.")
    if control[0].get("library_revision") or with_books[0].get("library_revision") != lib.revision:
        raise ValueError("Первый журнал должен быть контрольным, второй — с действующей библиотекой.")
    story = load_scenario(control[0]["scenario"])
    turns = [[e for e in run if e["event"] == "turn"] for run in (control, with_books)]
    common = 0
    for a, b in zip(*turns):
        if (a["actor"], a["line"]) != (b["actor"], b["line"]):
            break
        common += 1
    output = {"scenario": story.key, "seed": control[0]["seed"], "same_opening_turns": common}
    for key, run, accepted in zip(("control", "with_books"), (control, with_books), turns, strict=True):
        votes = run[-1]["votes"]
        output[key] = {"turns": len(accepted), "citations": len(check_archive(run, lib)),
                       "outcome": run[-1]["outcome"],
                       "votes_to_other_ending": min(v["distance"] for v in alternatives(story, votes).values())}
    a, b = control[-1]["votes"], with_books[-1]["votes"]
    output["changed_votes"] = {story.cast[k].name: [ballot_phrase(story, k, a[k]), ballot_phrase(story, k, b[k])]
                               for k in story.cast if a[k] != b[k]}
    return output


def dossier(events: list[dict], lib: Library) -> str:
    citations = check_archive(events, lib)
    story = load_scenario(events[0]["scenario"])
    turns = {e["index"]: e for e in events if e["event"] == "turn"}
    searches = {(e["index"], e["selected"]): e for e in events if e["event"] == "book_search" and e["selected"]}
    lines = [f"# Свидетельства в спектакле «{story.title}»", "", f"Зерно прогона {events[0]['seed']}. "
             f"Принятых реплик {len(turns)}, прочитанных фрагментов {len(citations)}.", ""]
    for e in citations:
        card = e["card"]
        lines.extend([f"## Ход {e['index']}. {e['name']}", ""])
        if e["kind"] == "read":
            search = searches[e["index"], card["ref"]]
            lines.append(f"Запрос актёра — «{search['query']}». Из {len(search['hits'])} предложенных фрагментов выбран `{card['ref']}`.")
        else:
            lines.append(f"Актёр открывает контекст фрагмента `{e['origin']}`, прозвучавшего на ходу {e['origin_turn']}.")
        lines.extend(["", f"Вопрос на сцене — {e['question']}", "",
                      f"Выбранное актёром намерение — {e['stance']}.", "",
                      f"> {card['speaker']}. {printed_quote(card)}", "",
                      f"[{card['author']}, «{card['title']}», действие {card['act']}, {card['section'].lower()}]({card['url']})",
                      "", f"**{e['name']}.** {turns[e['index']]['line']}", ""])
    withdrawn = [e for e in events if e["event"] == "book_withdrawn"]
    if withdrawn:
        lines.extend(["## Где актёр отложил книгу", ""])
        for e in withdrawn:
            lines.append(f"- Ход {e['index']}, {e['name']}. Фрагмент `{e['ref']}` остался за кулисами. "
                         f"Последнее замечание режиссёра — «{e['reason']}»")
        lines.append("")
    ending = events[-1]
    lines.extend(["## Действительный финал", "", ending["curtain"], ""])
    for a, v in ending["votes"].items():
        lines.append(f"- {story.cast[a].name} — {ballot_phrase(story, a, v)}")
    possibilities = alternatives(story, ending["votes"])
    distance = min(item["distance"] for item in possibilities.values())
    lines.extend(["", "## Ближайшая альтернативная развязка", "",
                  f"Минимальное число изменений бюллетеней — {distance}. Это пересчёт тех же правил после спектакля.", ""])
    for outcome, possibility in possibilities.items():
        if possibility["distance"] != distance:
            continue
        edits = possibility["example"]
        _, curtain = resolve(story, ending["votes"] | edits)
        who = " ".join(f"{story.cast[k].name}: «{ballot_phrase(story, k, v)}»." for k, v in edits.items())
        lines.append(f"- {who} Тогда: {curtain} Минимальных вариантов для этого исхода: {possibility['ways']}.")
    return "\n".join(lines) + "\n"
