import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

from .book_scene import citation_intro, printed_quote
from .engine import perform
from .evidence import dossier
from .library import Library
from .llm import DEFAULT_A, DEFAULT_B, Gateway, GenerationError
from .scenario import SCENARIOS, load_scenario
from .schema import Plan


def read_saved_plan(directory: Path, scenario: str) -> tuple[Plan, str]:
    """Берём только первые два события: актёры не получат диалог прежнего состава."""
    try:
        with (directory / "trace.jsonl").open(encoding="utf-8") as journal:
            start, event = json.loads(journal.readline()), json.loads(journal.readline())
        if start["event"] != "start" or start["scenario"] != scenario:
            raise ValueError("папка относится к другой постановке")
        if event["event"] != "plan" or not isinstance(event["model"], str) or not event["model"]:
            raise ValueError("отсутствует план с именем драматурга")
        return Plan(scenes=event["scenes"]), event["model"]
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать план из {directory}: {exc}") from exc


def markdown(title: str, events: list[dict]) -> str:
    lines = [f"# {title}", ""]
    ballot_labels = {"challenge": "оспорить условие", "abstain": "воздержаться"}
    ballot_labels.update({f"select:{e['actor']}": e["name"] for e in events if e["event"] == "turn"})
    for e in events:
        if e["event"] == "scene":
            lines.extend([f"## Сцена {e['scene']}. {e['title']}", "", f"*{e['opening']}*", ""])
            if e.get("action"):
                lines.extend([f"*{e['action']}*", ""])
            if e["notice"]:
                lines.extend([f"**Табло.** {e['notice']}", ""])
        elif e["event"] == "citation":
            card = e["card"]
            lines.extend([f"*{citation_intro(e)}*", "", f"> {printed_quote(card)}", "",
                          f"[Отрывок в издании]({card['url']})", ""])
        elif e["event"] == "turn":
            if e["stage"]:
                lines.extend([f"*{e['stage']}*", ""])
            lines.extend([f"**{e['name']}.** {e['line']}", ""])
            if e["vote"]:
                lines.extend([f"*На табло у имени «{e['name']}» — «{ballot_labels[e['vote']]}».*", ""])
        elif e["event"] == "ending":
            lines.extend([f"*{e['curtain']}*", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сыграть многоагентную пьесу на русском языке")
    parser.add_argument("scenario", choices=SCENARIOS, help="extra — Лишний персонаж, lab404 — Лабораторная № 404")
    parser.add_argument("--model-a", default=DEFAULT_A, help="модель первого и третьего актёра")
    parser.add_argument("--model-b", default=DEFAULT_B, help="модель второго и четвёртого актёра")
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument("--planner", help="по умолчанию model-a; для GPT укажите openai/ID")
    planning.add_argument("--plan-from", type=Path, help="папка прежнего прогона этой пьесы, чей план сыграть заново")
    editing = parser.add_mutually_exclusive_group()
    editing.add_argument("--editor", help="модель для смысловой проверки черновика (по умолчанию model-a)")
    editing.add_argument("--no-editor", action="store_true", help="оставить только проверки Python")
    parser.add_argument("--no-library", action="store_true", help="контрольная постановка без чтения больших пьес")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--turns", type=int, choices=range(4, 11), default=6,
                        help="ходов в каждой из первых трёх сцен, в финале всегда четыре")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--out", type=Path, help="новая папка прогона (по умолчанию runs/...)")
    args = parser.parse_args(argv)
    story = load_scenario(args.scenario)
    if args.model_a == args.model_b:
        parser.error("укажите две разные модели")
    planner = args.planner or args.model_a
    plan = None
    if args.plan_from:
        try:
            plan, planner = read_saved_plan(args.plan_from, story.key)
        except ValueError as exc:
            parser.error(str(exc))
    models = {key: (args.model_a, args.model_b)[i % 2] for i, key in enumerate(story.cast)}
    editor = None if args.no_editor else args.editor or args.model_a
    library = None if args.no_library else Library()
    out = args.out or Path("runs") / f"{story.key}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"папка {out} уже существует; укажите новую, чтобы сохранить предыдущий прогон")

    events: list[dict] = []
    url = args.ollama_url if "://" in args.ollama_url else "http://" + args.ollama_url
    with (out / "trace.jsonl").open("x", encoding="utf-8") as journal:
        def emit(e: dict) -> None:
            journal.write(json.dumps(e, ensure_ascii=False) + "\n")
            journal.flush()
            events.append(e)
            if e["event"] == "plan":
                (out / "plan.json").write_text(json.dumps(e, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            elif e["event"] == "scene":
                print(f"{e['scene']}. {e['title']}", flush=True)

        emit({"event": "start", "scenario": story.key, "seed": args.seed,
              "models": models, "planner": planner, "turns_per_scene": args.turns,
              "library_revision": library.revision if library else None,
              "cast": {key: {"name": a.name, "gender": a.gender, "speaker": a.speaker}
                       for key, a in story.cast.items()},
              "plan_from": str(args.plan_from) if args.plan_from else None, "editor": editor})
        with httpx.Client(timeout=httpx.Timeout(180, connect=10)) as http:
            try:
                perform(story, Gateway(http, url, args.seed), models, planner, args.turns, emit,
                        plan=plan, editor=editor, library=library)
            except GenerationError as exc:
                emit({"event": "error", "message": str(exc)})
                print(f"Прогон прерван. Журнал: {out}\n{exc}", file=sys.stderr)
                return 1
    (out / "play.md").write_text(markdown(story.title, events), encoding="utf-8")
    if library:
        (out / "dossier.md").write_text(dossier(events, library), encoding="utf-8")
    print(f"Готово: {out / 'play.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
