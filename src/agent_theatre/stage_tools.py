"""Сценические инструменты. Актёр выбирает вызов до того, как решит, что сказать."""

from collections.abc import Callable
from dataclasses import asdict

from .archive import Archive, Hit, terms
from .llm import Generator
from .scenario import Scenario
from .schema import ToolCall, Turn


def quorum(option: str, cast_size: int) -> int:
    if option == "challenge":
        return cast_size
    if option == "nomination":
        return cast_size // 2 + 1
    raise ValueError(f"Неизвестный вид голосования: {option}")


def recall_line(history: list[Turn], speaker: str, query: str) -> Turn | None:
    """Ищем среди принятых реплик этого героя за все действия. При ничьей берём последнюю."""
    matches = [t for t in reversed(history) if t.actor == speaker]
    words = terms(query) - {"после", "репли", "фраза", "фразу", "слова", "сказа"}
    if not matches or not words:
        return next(iter(matches), None)
    scored = [(len(words & terms(t.speech.line)), t) for t in matches]
    score, turn = max(scored, key=lambda item: item[0])
    return turn if score else None


def use_skill(gen: Generator, model: str, story: Scenario, actor: str, scene: int, context: str,
              history: list[Turn], archive: Archive, searched: set[str], seen: dict[str, Hit],
              emit: Callable[[dict], None]) -> str:
    speakers = list(dict.fromkeys(t.actor for t in history))
    actions = ["", *(f"recall_line:{key}" for key in speakers)]
    menu = []
    if scene in (2, 3) and actor in story.researchers and actor not in searched:
        actions.append("search_archive")
        menu.append("search_archive — уточнить условие по файлу. Опись: " + " / ".join(archive.titles))
    if speakers:
        menu.append("recall_line:код — уточнить, что этот человек действительно сказал. "
                    "В query укажи тему или оставь пустым для его последней реплики. Коды: "
                    + ", ".join(f"{key} = {story.cast[key].name}" for key in speakers))
    if scene >= 2:
        actions.extend(("test_ballot:challenge", "test_ballot:nomination"))
        menu.append("test_ballot:challenge — проверить условную поддержку совместного пересмотра, "
                    "test_ballot:nomination — поддержку одного имени. В supporters поставь число от 1 до 4.")
    if not menu:
        return ""
    call = gen.ask(model, f"Ты — {story.cast[actor].name}. Выбираешь сценический инструмент перед ходом.",
        context + "\n" + "\n".join(menu) +
        "\nКакую одну деталь стоит проверить для ответа партнёру? Выбери action. "
        "Для поиска в архиве нужны ключевые слова из описи. Для проверки голосов достаточно числа. "
        "Если проверять нечего, оставь action пустым.", ToolCall, choices={"action": actions}, budget=175)
    if not call.action:
        return ""
    index = len(history) + 1
    event = {"scene": scene, "index": index, "actor": actor, "model": model}
    if call.action == "search_archive":
        searched.add(actor)
        hits = archive.search(call.query)
        seen.update((hit.ref, hit) for hit in hits)
        emit({**event, "event": "search", "tool": call.action, "query": call.query,
              "hits": [asdict(h) for h in hits]})
        return "В найденном документе: " + " ".join(h.text for h in hits) if hits else "Архив не нашёл подтверждения."
    tool, argument = call.action.split(":", 1)
    if tool == "recall_line":
        turn = recall_line(history, argument, call.query)
        args = {"speaker": argument, "query": call.query}
        result = {"turn": turn.index, "scene": turn.scene, "text": turn.speech.line} if turn else None
        message = (f"Точные слова {story.cast[argument].name} из действия {turn.scene}: «{turn.speech.line}»."
                   if turn else "Подходящей реплики этого участника в записи нет.")
    else:
        needed = quorum(argument, len(story.cast))
        args = {"option": argument, "supporters": call.supporters}
        result = {"required": needed, "enough": call.supporters >= needed}
        option = "совместный пересмотр" if argument == "challenge" else "выбор одного имени"
        message = (f"Мысленный пример: за {option} — {call.supporters} из {len(story.cast)}. "
                   f"Порог — {needed}. Голосов {'достаточно' if result['enough'] else 'пока мало'}. "
                   "Это условный расчёт. Фактические голоса регистрируются отдельно.")
    emit({**event, "event": "skill", "tool": tool, "arguments": args, "result": result})
    return message
