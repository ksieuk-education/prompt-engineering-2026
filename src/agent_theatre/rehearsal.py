"""Проверки черновика до выхода на сцену. Замечания адресуются той же модели."""

import re
from collections.abc import Callable
from difflib import SequenceMatcher

from .archive import terms
from .llm import GenerationError, Generator
from .scenario import Actor
from .schema import Delivery, Review


FAMILY_FORMS = {
    "брат": r"брат(?:а|у|ом|е|ья|ьев)?", "сестра": r"сестр(?:а|у|ы|е|ой|ёнка)",
    "мать": r"(?:мать|матери|мам(?:а|у|ы|е|ой))", "отец": r"(?:отец|отц[ауы]|пап[ауы])",
    "родители": r"родител(?:и|ей|ям|ях)", "дети": r"(?:дети|детей|детям|ребёнок|ребёнка)",
    "жена": r"жен[ауые]|супруга", "муж": r"муж(?:а|у|ем)?",
    "сын": r"сын(?:а|у|ом|овья)?", "дочь": r"(?:дочь|доч(?:ка|ери|ку))",
}
FAMILY_CLAIM = re.compile(
    r"\b(?:(?:мо[йяиюё]|моего|моей|моих)\s+|у меня (?:есть |был[аи]? )?)"
    r"(?:(?:старш|младш|родн)\w+\s+)?(?:"
    + "|".join(f"(?P<{relation}>{forms})" for relation, forms in FAMILY_FORMS.items())
    + r")\b", re.I,
)
INLINE_ACTION = re.compile(
    r"[(（*]\s*(?:я\s+)?(?:показ|подним|подня|покач|поворач|наклон|смотр|гляд|"
    r"сжим|указ|вздых|пожим|улыба|прикас|отступ|подход|шепч|кива)\w*[\s,.)*]", re.I,
)
SELF_VOTE_CONFLICT = re.compile(
    r"\b(?:не могу (?:их|семью|детей|Монику) (?:оставить|бросить)|"
    r"(?:долж(?:ен|на)|обязан[а]?) (?:сам[а]? )?вернуться домой)\b", re.I,
)

# Формулировки адресованы актёру. Причины и исходные реплики также попадут в журнал.
NOTES = {
    "family": "Убери личную историю о родственнике, которого нет в твоей карточке. Возьми мотив из роли.",
    "vote": "Ты согласился остаться здесь. Объясни, ради чего принимаешь разлуку с домом.",
    "repeat": "Ты уже высказывал этот довод почти теми же словами. Ответь партнёру поступком, уступкой или контрпримером.",
    "clock": "Часы уже обсуждали двое. Оставь этот предмет за кадром. Что объявленное условие делает с конкретным человеком?",
    "aside": "Перенеси описание движения из прямой речи в поле жеста. Вслух оставь только слова.",
    "length": "Сократи реплику до двух коротких предложений, жест — до одного движения.",
}


def mentions_clock(line: str) -> bool:
    return bool(re.search(r"\bчас(?:ы|ов|ам|ами|ах)\b", line, re.I))


def draft_issues(words: Delivery, actor: Actor, prior: list[str], staying: bool,
                 clock_discussed: bool = False) -> list[str]:
    """Узкие признаки для переигровки, а не оценка литературного качества."""
    issues = []
    # Упоминание чужих родителей или цитирование партнёра само по себе допустимо.
    unquoted = re.sub(r'«[^»]*»|"[^"]*"', " ", words.line)
    for claim in FAMILY_CLAIM.finditer(unquoted):
        if claim.lastgroup not in actor.family:
            issues.append("family")
            break
    if staying and SELF_VOTE_CONFLICT.search(unquoted):
        issues.append("vote")
    vocab = terms(words.line)
    tokens = re.findall(r"\w+", words.line.casefold())
    for previous in prior[-3:]:
        earlier = terms(previous)
        shared = len(vocab & earlier)
        overlap = SequenceMatcher(None, tokens, re.findall(r"\w+", previous.casefold())).find_longest_match()
        if overlap.size >= 7 or (shared >= 4 and shared / len(vocab | earlier) >= 0.65):
            issues.append("repeat")
            break
    if clock_discussed and mentions_clock(words.line):
        issues.append("clock")
    if INLINE_ACTION.search(words.line):
        issues.append("aside")
    if len(words.line.split()) > 38 or len(words.stage.split()) > 12 or "?" in words.stage:
        issues.append("length")
    return issues


def deliver(gen: Generator, model: str, role: str, prompt: str, actor: Actor,
            prior: list[str], staying: bool, clock_discussed: bool, record: Callable[[dict], None],
            *, review: Callable[[Delivery], Review] | None = None,
            repair_prompt: str | None = None) -> tuple[Delivery, bool, list[str]]:
    correction = ""
    for take in range(1, 4):
        words = gen.ask(model, role, prompt + correction, Delivery, budget=340)
        issues = draft_issues(words, actor, prior, staying, clock_discussed)
        note = ""
        if review and (not issues or take > 1):
            verdict = review(words)
            if verdict.issue:
                issues.append(verdict.issue)
                note = verdict.note
        hard = bool({"family", "vote", "biography", "ballot", "evidence"} & set(issues))
        if not issues or (take >= 2 and not hard):
            return words, take > 1, issues
        record({"take": take, "line": words.line, "stage": words.stage, "issues": issues, "note": note})
        if take == 3:
            raise GenerationError(f"{actor.name}: после трёх дублей остаётся замечание {', '.join(issues)}")
        if hard and repair_prompt:
            # Повтор забракованного анекдота в подсказке закреплял его у малой модели.
            # При фактической ошибке актёр сохраняет адресата, источник и голос,
            # но получает чистый лист вместо черновика и своего прежнего намерения.
            prompt = repair_prompt
            hints = {
                "biography": "Используй только дословно подтверждаемую личную деталь из карточки роли. Можно прямо отказаться предъявлять воспоминание.",
                "evidence": "Скажи лишь то, что действительно установлено. Следующий шаг сформулируй как предложение или вопрос.",
                "ballot": "Выскажи одну причину уже принятого решения и назови его цену лично для тебя.",
            }
            correction = "\n" + "\n".join(hints.get(issue, NOTES.get(issue, "")) for issue in issues)
            if take == 2:
                correction += "\nОставь только одно простое предложение, которое можно обосновать карточкой."
            continue
        instructions = [NOTES[issue] for issue in issues if issue in NOTES]
        if note:
            instructions.append(note)
        correction = (f"\nРедактор возвращает твой черновик: «{words.line}»\n"
                      + "\n".join(instructions) + "\nСыграй заново одной короткой фразой.")
    raise AssertionError("unreachable")


def review_line(gen: Generator, editor: str, actor: Actor, context: str, cue: str,
                words: Delivery) -> Review:
    system = ("Ты редактор пьесы. Проверяй только произнесённый текст. Допускай юмор, гипотезы, "
              "спор, чувства и перемену мнения. Возвращай одну обоснованную поправку. "
              "При отсутствии явного нарушения оставь issue и note пустыми.")
    return gen.ask(editor, system,
        f"Роль: {actor.name}. Известная биография: {actor.stake}\n"
        f"Условия и разговор на сцене:\n{context}\nМизансцена и выбор актёра: {cue}\n"
        f"Проверяемая реплика: «{words.line}».\n"
        "biography — герой добавил конкретное личное событие, место, занятие или родственника, "
        "которых нет в известной биографии. evidence — герой объявил новым измеренным фактом то, "
        "чего прибор не сообщал, или исказил текст полученной карточки. ballot — актёр призывает "
        "к решению, несовместимому с собственным голосом. echo — просто повторил предыдущие слова "
        "без ответа или нового шага. Выбирай категорию только при явной ошибке. "
        "В поправке напиши по-русски одну короткую инструкцию. Назови лишнюю деталь и опору в сцене. "
        "Не сочиняй за него реплику. Если он неточно цитирует другого героя, не выдавай эту цитату за факт.",
        Review, choices={"issue": ["", "biography", "evidence", "ballot", "echo"]}, budget=420)
