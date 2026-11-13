"""Читка по актам. Импровизация, вызовы инструментов и голоса остаются в исходном журнале."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import httpx
from pydantic import Field, create_model

from .analysis import read_trace
from .book_scene import acting_note, check_comment
from .engine import ballot_phrase
from .language import stage_direction
from .llm import Gateway, GenerationError, Generator
from .rehearsal import draft_issues
from .scenario import Scenario, load_scenario
from .schema import Delivery, Form


class RoleLine(Form):
    line: str = Field(alias="реплика")
    stage: str = Field(alias="жест")


class ReadingLine(RoleLine):
    index: int


class SceneReading(Form):
    turns: list[ReadingLine]


# Темы реплик, которые можно перенести в читку без выдуманного факта из черновика.
TOPICS = {
    "семья и цена возвращения": "моник|дет(?:ей|и|ям)|семь|домой",
    "чья память может служить доказательством": "памят|помн|воспомина",
    "можно ли отказаться от навязанного выбора": "отказ|согласи|воздерж|навяз",
    "что на самом деле измеряет прибор": "датчик|измер|фиксир|показан",
    "кто имеет право судить участников": "наблюдател|судить|подпис|критер",
    "условия совместной попытки": "вместе|сообща|четвером|пересмотр",
    "что проверено у двери": "ручк|замок|замка|двер",
    "возможная проверка предложения": "провер|опыт|гипотез",
}


def scene_take(story: Scenario, turns: list[dict], take: SceneReading,
               citations: dict[int, dict] | None = None) -> list[dict]:
    if [line.index for line in take.turns] != [turn["index"] for turn in turns]:
        raise ValueError("Сохрани точное число, порядок и номера реплик.")
    result = []
    for turn, words in zip(turns, take.turns, strict=True):
        actor = story.cast[turn["actor"]]
        staying = story.key == "extra" and turn["vote"] == "select:" + turn["actor"]
        pledge = ballot_phrase(story, turn["actor"], turn["vote"]) if turn["vote"] else ""
        names = "|".join(re.escape(n) for n in [actor.name, actor.name.split()[0]])
        line = re.sub(rf"^\s*(?:{turn['index']}[.)]\s*)?(?:{names})\s*[:.]\s*", "", words.line)
        line = line.removeprefix("—").strip()
        if line.startswith("«") and line.endswith("»"):
            line = line[1:-1].strip()
        if re.match(r"^\d+[.)]\s|^[А-ЯЁ][а-яё]+(?: [А-ЯЁ][а-яё]+)?:\s", line):
            raise ValueError(f"В поле реплики {turn['index']} должны быть только слова этого актёра.")
        if pledge:
            options = ["challenge", "abstain", *(f"select:{k}" for k in story.cast)]
            if any(other != pledge and line.startswith(other) for other in
                   (ballot_phrase(story, turn["actor"], option) for option in options)):
                raise ValueError(f"Ход {turn['index']}: решение уже было зарегистрировано как «{pledge}».")
        line = line.removeprefix(pledge).strip() if pledge else line
        clean = Delivery(line=line, stage=words.stage.strip("()* "))
        if citation := (citations or {}).get(turn["index"]):
            cast_text = " ".join(a.name + " " + a.stake for a in story.cast.values())
            verdict = check_comment(clean, citation, cast_text)
            if verdict.issue:
                raise ValueError(f"Ход {turn['index']}: {verdict.note}")
        issues = draft_issues(clean.model_copy(update={"stage": ""}), actor, [], staying,
                              partners=list(story.cast.values()), addressee=story.cast.get(turn["to"]))
        if issues:
            raise ValueError(f"Ход {turn['index']} ({actor.name}, {actor.agreement}): {', '.join(issues)}.")
        gesture = clean.stage
        stage = stage_direction(gesture, actor) if len(gesture.split()) <= 12 else ""
        result.append({"index": words.index, "line": (pledge + " " + line).strip(), "stage": stage})
    return result


def apply_reading(events: list[dict], reading: list[dict]) -> list[dict]:
    turns = [e for e in events if e["event"] == "turn"]
    if [t["index"] for t in turns] != [r["index"] for r in reading]:
        raise ValueError("Номера реплик читки не совпадают с журналом.")
    replacements = iter(reading)
    return [e | {k: v for k, v in next(replacements).items() if k in {"line", "stage"}}
            if e["event"] == "turn" else e for e in events]


def read_performance(run: Path, *, edition: bool = True) -> list[dict]:
    events = read_trace(run)
    file = run / "reading.json"
    if not file.is_file():
        return events
    reading = json.loads(file.read_text(encoding="utf-8"))
    if reading["trace_sha256"] != hashlib.sha256((run / "trace.jsonl").read_bytes()).hexdigest():
        raise ValueError("Журнал изменился после читки. Для записи нужен соответствующий ему текст.")
    performance = apply_reading(events, reading["turns"])
    edited = run / "edition.json"
    if edition and edited.is_file():
        changes = json.loads(edited.read_text(encoding="utf-8"))
        if changes["reading_sha256"] != hashlib.sha256(file.read_bytes()).hexdigest():
            raise ValueError("Финальная редактура относится к другой читке.")
        return revise_lines(performance, changes["changes"])
    return performance


def revise_lines(events: list[dict], changes: dict[str, dict]) -> list[dict]:
    story = load_scenario(events[0]["scenario"])
    turns = [e for e in events if e["event"] == "turn"]
    if not isinstance(changes, dict) or not set(changes) <= {str(t["index"]) for t in turns}:
        raise ValueError("Исправить можно только существующий ход читки.")
    revised = []
    citations = {e["index"]: e for e in events if e["event"] == "citation"}
    for turn in turns:
        edited = changes.get(str(turn["index"]))
        if edited is None:
            revised.append({key: turn[key] for key in ("index", "line", "stage")})
            continue
        if not edited or set(edited) - {"line", "stage"}:
            raise ValueError("В правке допустимы только поля line и stage.")
        text = ReadingLine(index=turn["index"], реплика=edited.get("line", turn["line"]),
                           жест=edited.get("stage", turn["stage"]))
        revised.extend(scene_take(story, [turn], SceneReading(turns=[text]), citations))
    return apply_reading(events, revised)


def publish_edition(run: Path, changes: dict, credit: str) -> None:
    from .cli import markdown

    reading = run / "reading.json"
    digest = hashlib.sha256(reading.read_bytes()).hexdigest()
    events = revise_lines(read_performance(run, edition=False), changes)
    payload = {"editor": credit, "reading_sha256": digest, "changes": changes}
    publish_pair(run / "edition.md", markdown(load_scenario(events[0]["scenario"]).title, events),
                 run / "edition.json", payload)


def rehearse_scene(story: Scenario, scene: dict, turns: list[dict], available: dict,
                   previous: list[str], gen: Generator, model: str,
                   notes: dict[str, str] | None = None, citations: dict[int, dict] | None = None,
                   kept_turns: dict[int, str] | None = None) -> list[dict]:
    cast = "\n".join(f"{a.name} ({a.agreement}): {a.voice} Известно: {a.stake}" for a in story.cast.values())
    lines = []
    for i, t in enumerate(turns):
        ref = (f" Опора: {available[t['actor'], t['source']]}" if t["source"] else "")
        vote = ballot_phrase(story, t["actor"], t["vote"]) if t["vote"] else ""
        themes = [label for label, pattern in TOPICS.items() if re.search(pattern, t["line"], re.I)]
        following = f" Затем говорит {turns[i+1]['name']}." if i+1 < len(turns) else " Конец действия."
        lines.append(f"{t['index']}. {t['name']}. Мотивы её или его импровизации: "
                     f"{', '.join(themes[:2]) or 'реакция на партнёра'}. "
                     f"Отвечает на ход {t.get('reply_to') or 'открытие сцены'}.{ref}"
                     + (f" Обязательный голос: {vote}" if vote else "") + following)
        if direction := (notes or {}).get(str(t["index"])):
            lines.append("Режиссёрское задание для этой реплики: " + direction)
    system = ("Ты драматург русской камерной пьесы. Актёры уже сыграли и приняли решения. "
              "Сделай устную читку: четыре разных живых человека слушают друг друга. Коротко, по-русски. "
              "Произнеси ответ на конкретный вопрос. Один собеседник вправе возразить, уступить или "
              "признать незнание. Если другой герой верно его понял, не начинай ответ с пересказа. "
              "Береги сухой юмор Чендлера, точность Гермионы, прямоту Барби. "
              "Пример ритма (не копируй): «Сколько ты теряешь, если остаёшься?» — «Ужин дома. "
              "Смешно звучит, пока не представишь пустую тарелку». — «Тогда почему ты отодвинул стул?» "
              "Каждая фраза обращена к конкретному человеку. Дай разговору начаться, измениться и закончиться.")
    canon = ("Терминал гарантирует три пропуска. У него нет измерения ценности людей. Тони не управляет "
             "магией Гермионы. Четвёртый человек может не получить пропуск."
             if story.key == "extra" else
             "Терминал не знает, кто именно настоящий, и никому не присвоил этот статус. На экране только "
             "условие задачи. Никто из этих четверых до этой встречи не знал другого лично.")
    if scene["scene"] == 1:
        canon = ("Четверо встретились впервые. Дверь существует и остаётся закрытой после проверки. "
                 "Экран тёмный. О выборе, голосовании и пропусках никто ещё не слышал. "
                 "Начни с обычного разговора: кому нужно вернуться, что стоящий рядом может сделать для него.")
    direction = " ".join(story.cues[scene["scene"]-1])
    prompt = (f"«{story.title}». {story.setting}\nВерные условия мира: {canon}\nРоли:\n{cast}\n"
              "Уже прозвучало до этого действия:\n"
              + ("\n".join(previous[-5:]) or "(пьеса только началась)")
              + f"\nСцена {scene['scene']}. {story.arc[scene['scene']-1]}\n"
                f"На сцене: {scene['opening']} {scene.get('action', '')}\n"
              + (f"Объявление: {story.notice}\n" if scene["scene"] >= 2 else "")
              + f"Задача драматурга в этом действии: {direction}\nХоды актёров в фактическом порядке:\n" + "\n".join(lines) +
              "\nЖёсткие условия — говорящие, их решения, правила терминала, полученные улики и мизансцена. "
              "Свободно перепиши неудачные доводы. Если в черновике путаются причины и следствия, "
              "придумывают измерения, чужие способности или тянут случайную метафору, выброси это целиком. "
              "Опирайся на наблюдаемое действие, личный интерес конкретного героя и услышанный вопрос. "
              "Если герой предлагает измерить неизмеримое, пусть другой спросит, как именно, и получит ответ. "
              "Исправь ложные цитаты и несовместимые утверждения. Гипотезы остаются гипотезами. "
              "Если актёр отвечает за другого, верни каждому его собственную мысль. Никаких новых "
              "событий прошлого, родственников, чисел датчика, надписей на экране или открытий двери. "
              "Проверку ручки уже видели. Помни, кто выдвинул какой довод. В каждой строке точный index, "
              "Избегай отвлечённых метафор и загадок. Ручка, стул, семейный ужин, пустая строка "
              "эксперимента — это буквальные предметы и обстоятельства. Если задаёшь прямой вопрос, дай ответить тому, кто действительно говорит следующим. "
              "Обращение «ты» и род согласуй с ним. У персонажей нет общих приключений в прошлом. "
              "одна-две законченные произносимые фразы (до 32 слов) и необязательный короткий жест "
              "от третьего лица. В реплике только произносимые слова, без номера и имени. В финале пиши только обоснование, "
              "формулу голоса программа прикрепит сама. У актёров не может быть чужих реплик.")
    # Идентичность говорящего принадлежит журналу. Отдельное обязательное поле
    # на каждый ход исключает перепутанные номера и пропущенные реплики в JSON.
    keys = [f"turn_{turn['index']}_{turn['actor']}" for turn in turns]
    sheet_type = create_model("ActScript", __base__=Form, **{key: (RoleLine, ...) for key in keys})
    prompt += "\nИмена полей в порядке исполнения: " + ", ".join(keys) + "."
    correction = ""
    for attempt in range(3):
        sheet = gen.ask(model, system, prompt + correction, sheet_type, budget=max(850, len(turns) * 190))
        take = SceneReading(turns=[ReadingLine(index=t["index"], **getattr(sheet, k).model_dump(by_alias=True))
                                   for t, k in zip(turns, keys, strict=True)])
        try:
            return scene_take(story, turns, take, citations)
        except ValueError as exc:
            if attempt == 2:
                # Когда монтаж портит книжную реплику, можно взять уже сыгранный
                # дубль. Он проходит те же проверки; происхождение сохраняем.
                final = []
                for t, line in zip(turns, take.turns, strict=True):
                    try:
                        segment = scene_take(story, [t], SceneReading(turns=[line]), citations)
                    except ValueError as mistake:
                        if t["index"] not in (citations or {}):
                            raise GenerationError(f"Не принята читка сцены {scene['scene']}: {mistake}") from mistake
                        original = ReadingLine(index=t["index"], реплика=t["line"], жест=t["stage"])
                        try:
                            segment = scene_take(story, [t], SceneReading(turns=[original]), citations)
                        except ValueError as invalid:
                            raise GenerationError(f"Не принят и исходный дубль: {invalid}") from invalid
                        if kept_turns is not None:
                            kept_turns[t["index"]] = str(mistake)
                    final.extend(segment)
                return final
            correction = f"\nИсправь предыдущую попытку и верни все строки сцены. {exc}"
    raise AssertionError("unreachable")


def prepare_reading(run: Path, gen: Generator, model: str) -> None:
    from .cli import markdown

    events = read_trace(run)
    if not events or events[-1]["event"] != "ending":
        raise ValueError("Для читки нужен завершённый спектакль.")
    manifest, script = run / "reading.json", run / "reading.md"
    if manifest.exists() or script.exists():
        raise FileExistsError("В папке уже есть читка.")
    story = load_scenario(events[0]["scenario"])
    direction_file = run / "directions.json"
    notes = json.loads(direction_file.read_text(encoding="utf-8")) if direction_file.exists() else {}
    indices = {str(e["index"]) for e in events if e["event"] == "turn"}
    if not isinstance(notes, dict) or not set(notes) <= indices or any(
            not isinstance(text, str) or not text.strip() for text in notes.values()):
        raise ValueError("В режиссёрском листе ожидаются номера сыгранных ходов и непустые задания.")
    available = {(e["actor"], f"{h['file']}:{h['line']}"): h["text"]
                 for e in events if e["event"] == "search" for h in e["hits"]}
    scene_notes = dict(notes)
    citations = {e["index"]: e for e in events if e["event"] == "citation"}
    for index, event in citations.items():
        key = str(index)
        scene_notes[key] = (scene_notes.get(key, "") + " " + acting_note(event)).strip()
    previous, reading, kept_turns = [], [], {}
    for scene in (e for e in events if e["event"] == "scene"):
        turns = [e for e in events if e["event"] == "turn" and e["scene"] == scene["scene"]]
        new = rehearse_scene(story, scene, turns, available, previous, gen, model, scene_notes, citations, kept_turns)
        reading.extend(new)
        previous.extend(f"{t['name']}: {v['line']}" for t, v in zip(turns, new, strict=True))
    data = {"model": model, "trace_sha256": hashlib.sha256((run / "trace.jsonl").read_bytes()).hexdigest(),
            "turns": reading}
    if notes:
        data["directions_sha256"] = hashlib.sha256(direction_file.read_bytes()).hexdigest()
    if kept_turns:
        data["retained_from_improvisation"] = kept_turns
    publish_pair(script, markdown(story.title, apply_reading(events, reading)), manifest, data)


def publish_pair(script: Path, text: str, manifest: Path, data: dict) -> None:
    # Публикуем только полную читку. Повторный процесс не заменяет ранее созданные файлы.
    with NamedTemporaryFile(dir=script.parent, mode="w", encoding="utf-8") as sheet, \
            NamedTemporaryFile(dir=manifest.parent, mode="w", encoding="utf-8") as journal:
        sheet.write(text)
        journal.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        sheet.flush()
        journal.flush()
        os.link(sheet.name, script)
        try:
            os.link(journal.name, manifest)
        except OSError:
            script.unlink()
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Провести литературную читку завершённой импровизации")
    parser.add_argument("run", type=Path)
    parser.add_argument("--model", default="gpt-oss:20b", help="литературный редактор")
    parser.add_argument("--edits", type=Path, help="применить JSON правок к существующей читке, без LLM")
    parser.add_argument("--credit", default="Редактор постановки", help="автор финальной вычитки")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    args = parser.parse_args(argv)
    try:
        if args.edits:
            publish_edition(args.run, json.loads(args.edits.read_text(encoding="utf-8")), args.credit)
            print(f"Готова редакция: {args.run / 'edition.md'}")
            return 0
        start = read_trace(args.run)[0]
        url = args.ollama_url if "://" in args.ollama_url else "http://" + args.ollama_url
        with httpx.Client(timeout=httpx.Timeout(240, connect=10)) as http:
            prepare_reading(args.run, Gateway(http, url, start["seed"]), args.model)
    except (GenerationError, OSError, ValueError) as exc:
        parser.exit(1, f"Читка не завершена. {exc}\n")
    print(f"Читка готова: {args.run / 'reading.md'}")
    return 0
