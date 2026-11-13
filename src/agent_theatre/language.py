"""Согласование русской речи с ролью и перевод сценических ремарок в третье лицо."""

import re
from collections.abc import Sequence
from functools import lru_cache

from pymorphy3 import MorphAnalyzer

from .scenario import Actor, VoiceRole


@lru_cache(maxsize=1)
def morphology() -> MorphAnalyzer:
    return MorphAnalyzer(lang="ru")


@lru_cache(maxsize=4096)
def parses(word: str):
    return morphology().parse(word)


QUOTED = re.compile(r'«[^»]*»|„[^“]*“|“[^”]*”|"[^"]*"')
PREDICATE = r"(?:\s+(?:не|бы|уже|тоже|сама?|тогда|просто|точно|верно|лично|сейчас|никогда|ещё))*\s+([а-яё]+)"


def wrong_gender(line: str, actor: Actor, partners: Sequence[Actor] = (),
                 addressee: Actor | None = None) -> bool:
    """Проверяем явные «я была» и «Тони, ты готова». У чужой цитаты собственный говорящий."""
    plain = QUOTED.sub(" ", line.casefold())
    subjects = [(r"\bя", actor)]
    named = False
    for partner in partners:
        names = "|".join(re.escape(s.casefold()) for s in [partner.name, *partner.name.split()])
        subjects.append((rf"\b(?:{names})\s*[,!]\s*ты", partner))
        if re.search(rf"(?:^|[,.!?…]\s*)(?:{names})\s*[,!?.]", plain):
            named = True
    if addressee and not named:
        subjects.append((r"\bты", addressee))
    for subject, role in subjects:
        expected = "femn" if role.gender == "female" else "masc"
        for match in re.finditer(subject + PREDICATE, plain):
            # «Я голос» — существительное. Согласуем только предикаты с различимым родом.
            genders = {p.tag.gender for p in parses(match[1]) if p.tag.number == "sing"
                       and (p.tag.POS == "VERB" and p.tag.tense == "past"
                            or p.tag.POS in {"ADJS", "PRTS"} and p.tag.gender in {"masc", "femn"})}
            if genders and expected not in genders:
                return True
    # Нулевой субъект распространён в устном ответе «Не уверена, что ...».
    # Вопросительные фразы могут быть обращением ко второму лицу.
    for sentence in re.split(r"(?<=[.!?…])\s+", plain):
        if "?" in sentence:
            continue
        implied = re.match(r"^(?:не )?(уверен[а]?|соглас(?:ен|на)|готов[а]?|рад[а]?)\b", sentence)
        if implied and (implied[1].endswith("а") != (actor.gender == "female")):
            return True
    return False


def stage_direction(stage: str, actor: VoiceRole) -> str:
    """В старых журналах встречается «Поправляю очки». Рассказчица описывает действие актёра."""
    def third_person(match: re.Match) -> str:
        word = match[0]
        if word.casefold() == "я":
            return actor.name.split()[0]
        best = parses(word.casefold())[0]
        if best.tag.POS == "VERB" and best.tag.person == "1per" and best.tag.number == "sing":
            form = best.inflect({"3per", "sing"})
            if form:
                return form.word.capitalize() if word[0].isupper() else form.word
        return word
    # Кавычки могут содержать указание на чужую прямую речь.
    parts = re.split(r'(«[^»]*»|"[^"]*")', stage)
    return "".join(part if i % 2 else re.sub(r"[А-Яа-яЁё]+", third_person, part)
                   for i, part in enumerate(parts))
