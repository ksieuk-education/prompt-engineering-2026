"""Проверки черновика до выхода на сцену. Замечания адресуются той же модели."""

import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher

from .archive import terms
from .llm import GenerationError, Generator
from .language import wrong_gender
from .scenario import Actor
from .schema import Delivery, EditorialCheck, Review


class BookCommentRejected(GenerationError):
    """Актёр не сумел подтвердить свой комментарий и может отказаться от публичной цитаты."""


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
    "gender": "Согласуй род слов о себе и адресате с карточками ролей.",
    "unfinished": "Мысль оборвалась. Произнеси законченное короткое предложение с точкой или вопросительным знаком.",
}


def mentions_clock(line: str) -> bool:
    return bool(re.search(r"\bчас(?:ы|ов|ам|ами|ах)\b", line, re.I))


def draft_issues(words: Delivery, actor: Actor, prior: list[str], staying: bool,
                 clock_discussed: bool = False, partners: Sequence[Actor] = (),
                 addressee: Actor | None = None) -> list[str]:
    """Узкие признаки для переигровки, а не оценка литературного качества."""
    issues = []
    if wrong_gender(words.line, actor, partners, addressee):
        issues.append("gender")
    ending = words.line.rstrip('»"”’*) ')
    dangling = r"\b(?:если|чтобы|потому что|но|и|что я|если (?:вы собираетесь|ты собираешься))[.!?…]+$"
    if not ending.endswith((".", "?", "!", "…")) or re.search(dangling, ending, re.I):
        issues.append("unfinished")
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
            repair_prompt: str | None = None, partners: Sequence[Actor] = (),
            addressee: Actor | None = None, addressed_line: str = "",
            shorten: Callable[[Delivery], Delivery] | None = None) -> tuple[Delivery, bool, list[str]]:
    correction = ""
    for take in range(1, 4):
        words = gen.ask(model, role, prompt + correction, Delivery, budget=340)
        # Некоторые модели дословно печатают всю входную реплику перед своим ответом.
        if len(addressed_line.split()) >= 4 and words.line.startswith(addressed_line):
            own = words.line[len(addressed_line):].lstrip(" \n—")
            if own:
                words = words.model_copy(update={"line": own})
        if len(words.stage.split()) > 12:
            words = words.model_copy(update={"stage": ""})
        if shorten:
            brief = shorten(words)
            if words.line != brief.line:
                record({"take": take, "line": words.line, "stage": words.stage, "shortened_to": brief.line,
                        "issues": ["book_length"], "note": "Для чтения выбрано полное предложение из дубля."})
                words = brief
        issues = draft_issues(words, actor, prior, staying, clock_discussed, partners, addressee)
        note = ""
        if review and (not issues or take > 1):
            verdict = review(words)
            if verdict.issue:
                issues.append(verdict.issue)
                note = verdict.note
        hard = bool({"family", "vote", "biography", "ballot", "evidence", "gender", "unfinished", "book"} & set(issues))
        disputed = bool({"logic", "reply"} & set(issues))
        if not issues or (take >= 2 and not hard and (not disputed or take == 3)):
            return words, take > 1, issues
        record({"take": take, "line": words.line, "stage": words.stage, "issues": issues, "note": note})
        if take == 3:
            if "book" in issues:
                raise BookCommentRejected(note)
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
                "reply": "Ответь на конкретный вопрос партнёра одной фразой. Можно признать незнание или несогласие.",
                "logic": "Исправь противоречие с уже сказанным. Объясни, что в словах партнёра меняет твой выбор.",
            }
            correction = "\n" + "\n".join(hints.get(issue, NOTES.get(issue, "")) for issue in issues)
            if note and any(issue in {"reply", "logic", "ballot", "book"} for issue in issues):
                correction += "\nУточнение редактора: " + note
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
                words: Delivery, *, addressed_line: str, final: bool) -> Review:
    categories = ["ok", "biography", "evidence"]
    if addressed_line:
        categories.extend(["logic", "echo"])
        if "?" in addressed_line:
            categories.append("reply")
    if final:
        categories.append("ballot")
    verdict = gen.ask(editor,
        "Ты принимаешь актёрские дубли. По умолчанию ставь ok. Менять дубль стоит только при "
        "конкретной ошибке. Персонажи вправе спорить, строить гипотезы, шутить и задавать встречные вопросы.",
        f"Роль: {actor.name}. Биография: {actor.stake}\nСцена и известные записи:\n{context}\n"
        f"Выбор актёра: {cue}\nСлова, на которые отвечаем: {addressed_line or '(сцену только открыли)'}\n"
        f"Оцени только эту прямую речь: «{words.line}».\n"
        "ok — принять. biography — новое неподтверждённое конкретное событие личного прошлого. "
        "evidence — объявлено фактом измерение, которого в сцене не было. logic — явное "
        "противоречие с конкретным утверждением или приписывание партнёру чужих слов. "
        "reply — прямой вопрос проигнорирован. Признание незнания и уточнение по теме допустимы. "
        "echo — дословный повтор без развития. ballot — призыв, противоположный принятому голосу. "
        "При ok оставь excerpt, anchor и note пустыми. При ошибке в excerpt дословно скопируй "
        "проблемные слова актёра, в anchor — точную цитату вопроса, условия или голоса. "
        "В note коротко укажи, что исправить. Не оценивай жест и литературный вкус.",
        EditorialCheck, choices={"issue": categories}, budget=390)
    def normalize(value: str) -> str:
        return " ".join(value.casefold().replace("ё", "е").strip(' «»"').split())
    excerpt = normalize(verdict.excerpt)
    reference = normalize(verdict.anchor)
    source = addressed_line if verdict.issue == "reply" else context + "\n" + cue
    if (verdict.issue == "ok" or not excerpt or excerpt not in normalize(words.line)
            or (verdict.issue in {"reply", "logic", "ballot"}
                and (not reference or reference not in normalize(source)))):
        return Review(issue="", note="")
    return Review(issue=verdict.issue, note=verdict.note)
