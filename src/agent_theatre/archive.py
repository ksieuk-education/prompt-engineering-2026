"""Поиск по реквизиту текущей пьесы. Модель передаёт слова, а не путь к файлу."""

import re
from dataclasses import dataclass
from pathlib import Path

STOP = set("как кто что это для или при есть был была какую какой один всех этих меня нас них наш чем если каких можно проверить найти узнать".split())


def terms(text: str) -> set[str]:
    # Префиксы сближают «кнопку» и «кнопка», сохраняя поиск прозрачным для отчёта.
    words = re.findall(r"[а-яa-z0-9]+", text.casefold().replace("ё", "е"))
    return {word[:5] for word in words if len(word) >= 3 and word not in STOP}


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.file}:{self.line}"


class Archive:
    def __init__(self, directory: Path):
        self.entries: list[tuple[Hit, set[str]]] = []
        self.titles: list[str] = []
        for path in sorted(directory.glob("*.txt")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            self.titles.append(lines[0])
            for number, text in enumerate(lines[1:], 2):
                if text.strip():
                    hit = Hit(path.name, number, text.strip())
                    self.entries.append((hit, terms(lines[0] + " " + text)))

    def search(self, query: str, limit: int = 2) -> list[Hit]:
        wanted = terms(query)
        ranked = [(len(wanted & words), hit) for hit, words in self.entries]
        ranked.sort(key=lambda item: -item[0])  # порядок файлов устойчив и при равном счёте
        return [hit for score, hit in ranked if score][:limit]
