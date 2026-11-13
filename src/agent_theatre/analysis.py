"""Наблюдаемые признаки сыгранной пьесы для сравнения прогонов в ноутбуке."""

import json
from pathlib import Path

from .rehearsal import draft_issues, mentions_clock
from .scenario import load_scenario
from .schema import Delivery


def read_trace(directory: Path) -> list[dict]:
    with (directory / "trace.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def summarize(events: list[dict]) -> dict[str, int]:
    story = load_scenario(events[0]["scenario"])
    turns = [event for event in events if event["event"] == "turn"]
    searches = [event for event in events if event["event"] == "search"]
    recalls = [e for e in events if e["event"] == "skill" and e["tool"] == "recall_line"]
    issues = []
    for turn in turns:
        actor = story.cast[turn["actor"]]
        staying = story.key == "extra" and turn["vote"] == "select:" + turn["actor"]
        words = Delivery(line=turn["line"], stage=turn["stage"])
        issues.extend(draft_issues(words, actor, [], staying, partners=list(story.cast.values()),
                                   addressee=story.cast.get(turn["to"])))
    return {
        "Реплик": len(turns),
        "Актёров": len({turn["actor"] for turn in turns}),
        "Успешных поисков": sum(bool(e["hits"]) for e in searches),
        "Поисков всего": len(searches),
        "Обращений к памяти диалога": len(recalls),
        "Воспоминаний с дословной цитатой": sum(e["result"] is not None for e in recalls),
        "Проверок условного голосования": sum(e.get("tool") == "test_ballot" for e in events),
        "Ходов с указанной ответной репликой": sum(t.get("reply_to") is not None for t in turns),
        "Реплик с выбранной строкой архива": sum(bool(t["source"]) for t in turns),
        "Следом говорит адресат в той же сцене": sum(
            a["scene"] == b["scene"] and a["to"] == b["actor"] for a, b in zip(turns, turns[1:])),
        "Реплик со словом «часы» в разных падежах": sum(mentions_clock(t["line"]) for t in turns),
        "Заявлений о родственнике вне карточки по шаблону": issues.count("family"),
        "Явных противоречий самовыдвижению по шаблону": issues.count("vote"),
        "Реплик с жестом внутри прямой речи по шаблону": issues.count("aside"),
        "Несогласований рода по проверяемым формам": issues.count("gender"),
        "Записанных отклонённых дублей": sum(e["event"] == "retake" for e in events),
    }
