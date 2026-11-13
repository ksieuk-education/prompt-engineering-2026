"""Локальный поиск по TEI. Адрес результата указывает на строки неизменяемой редакции."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from xml.etree.ElementTree import TreeBuilder
from xml.parsers import expat

from .language import parses
from .scenario import DATA

DIRECTORY = DATA / "library"
WORDS = re.compile(r"[а-яёa-z0-9]+", re.I)
STOP = set("быть сказать говорить спросить мочь хотеть найти показать пьеса книга произведение сцена "
           "эпизод место отрывок герой реплика слово вспомнить именно очень который какой такой этот".split())
GROUPS = ("жребий монета орлянка решка решётка орёл", "тайный инкогнито тайком секрет",
          "забыть оставить позабыть", "письмо конверт корреспонденция", "торги аукцион торг",
          "инспектор проверяющий ревизор", "запереть запирать замок", "больница лечебница госпиталь")


@lru_cache(maxsize=32768)
def lemma(word: str) -> str:
    p = parses(word)[0]
    if p.tag.POS in {"PREP", "CONJ", "PRCL", "NPRO", "INTJ"}:
        return ""
    return p.normal_form.replace("ё", "е")


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    return [term for word in WORDS.findall(text) if len(word) > 1
            if (term := lemma(word)) and term not in STOP]


def xml_with_lines(raw: bytes):
    # ElementTree из стандартной библиотеки не хранит номера строк. Expat отдаёт их
    # в момент чтения исходных байтов, включая вложенные авторские ремарки.
    builder = TreeBuilder()
    parser = expat.ParserCreate()
    spans = {}
    def opening(tag, attrs):
        node = builder.start(tag, attrs)
        spans[node] = [parser.CurrentLineNumber, 0]
    def closing(tag):
        node = builder.end(tag)
        spans[node][1] = parser.CurrentLineNumber
    parser.StartElementHandler = opening
    parser.EndElementHandler = closing
    parser.CharacterDataHandler = builder.data
    parser.Parse(raw, True)
    return builder.close(), spans


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split()))


def chunks(text: str, size: int = 56):
    words = clean(text).split()
    while words:
        n = min(size, len(words))
        if n < len(words):
            n = next((i for i in range(n, n // 2, -1) if words[i-1].rstrip('»")').endswith((".", "!", "?", "…"))), n)
        yield " ".join(words[:n])
        words = words[n:]


@dataclass(frozen=True)
class Book:
    key: str
    author: str
    title: str
    file: str
    sha256: str


@dataclass(frozen=True)
class Passage:
    ref: str
    book: str
    act: int
    section: int
    heading: str
    speaker: str
    text: str
    first_line: int
    last_line: int


def passages(book: Book, raw: bytes):
    root, spans = xml_with_lines(raw)
    body = root.find("./text/body")
    if body is None:
        raise ValueError(f"{book.file}: отсутствует текст пьесы")
    serial = 0

    def parts(node, speaker):
        start, end = spans[node]
        if node.tag in {"p", "l"}:
            # В «Ревизоре» речь нередко прерывается ремаркой внутри того же абзаца.
            buf = node.text or ""
            for sub in node:
                if sub.tag == "stage":
                    yield speaker, buf, start, spans[sub][0]
                    yield "Ремарка", "".join(sub.itertext()), *spans[sub]
                    start, buf = spans[sub][1], ""
                else:
                    buf += "".join(sub.itertext())
                buf += sub.tail or ""
            yield speaker, buf, start, end
        elif node.tag == "stage":
            yield "Ремарка", "".join(node.itertext()), start, end
        elif node.tag in {"sp", "spGrp", "lg"}:
            if node.tag == "sp":
                speaker = clean(node.findtext("speaker") or node.get("who", "").replace("#", "")).rstrip(" .")
            for sub in node:
                yield from parts(sub, speaker)

    for act_no, act in enumerate(body.findall("div[@type='act']"), 1):
        section = 0
        for block in act:
            if block.tag == "div":
                section += 1
                head = clean(block.findtext("head") or f"Мизансцена {section}")
                nodes = list(block)
                section_no = section
            else:
                head, nodes, section_no = "Начало действия", [block], 0
            for node in nodes:
                for speaker, text, start, end in parts(node, ""):
                    for fragment in chunks(text):
                        serial += 1
                        yield Passage(f"{book.key}:{serial}", book.key, act_no, section_no, head,
                                      speaker or "Голос", fragment, start, end)


class Library:
    def __init__(self, directory: Path = DIRECTORY):
        catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        self.revision = catalog["revision"]
        self.repository = catalog["repository"]
        self.books = {b.key: b for b in (Book(**entry) for entry in catalog["works"])}
        self.entries: list[Passage] = []
        for book in self.books.values():
            raw = (directory / book.file).read_bytes()
            if hashlib.sha256(raw).hexdigest() != book.sha256:
                raise ValueError(f"Контрольная сумма не совпала: {book.file}")
            self.entries.extend(passages(book, raw))
        self.positions = {p.ref: n for n, p in enumerate(self.entries)}
        self.postings = defaultdict(dict)
        self.lengths = []
        self.bodies = []
        self.names = set()
        self.book_terms = defaultdict(set)
        for i, p in enumerate(self.entries):
            body = tokens(p.text)
            counts = Counter(body)
            if p.speaker != "Ремарка":
                name = tokens(p.speaker)
                counts.update(name)
                self.names.update(name)
            self.bodies.append(set(counts))
            self.book_terms[p.book].update(counts)
            self.lengths.append(sum(counts.values()))
            for token, freq in counts.items():
                self.postings[token][i] = freq
        self.avg_length = sum(self.lengths) / max(1, len(self.entries))
        self.synonyms = {}
        for group in GROUPS:
            words = set(tokens(group))
            for word in words:
                self.synonyms[word] = words - {word}

    def get(self, ref: str) -> Passage:
        if ref not in self.positions:
            raise ValueError(f"Такого фрагмента нет в каталоге: {ref}")
        return self.entries[self.positions[ref]]

    def context(self, ref: str, radius: int = 2) -> list[Passage]:
        at = self.positions[self.get(ref).ref]
        center = self.entries[at]
        if not 0 <= radius <= 5:
            raise ValueError("Радиус контекста должен быть от 0 до 5")
        return [p for p in self.entries[max(0, at-radius):at+radius+1]
                if (p.book, p.act, p.section) == (center.book, center.act, center.section)]

    def source(self, passage: Passage) -> str:
        name = self.books[passage.book].file
        return f"{self.repository}/blob/{self.revision}/tei/{name}#L{passage.first_line}-L{passage.last_line}"

    def search(self, query: str, *, book: str = "", limit: int = 3) -> list[Passage]:
        if book and book not in self.books:
            raise ValueError(f"Неизвестная книга: {book}")
        if not 1 <= limit <= 20:
            raise ValueError("Число результатов должно быть от 1 до 20")
        words = set(tokens(query))
        if book:
            meta = self.books[book]
            words -= set(tokens(meta.title + " " + meta.author))
        weights = {v: 0.62 for w in words for v in self.synonyms.get(w, ())}
        weights.update(dict.fromkeys(words, 1.0))
        vocabulary = self.book_terms[book] if book else self.postings.keys()
        if words - self.names and not (weights.keys() - self.names) & vocabulary:
            return []  # Знакомая фамилия не даёт подтверждения неизвестному предмету.
        if words - self.names:
            for name in self.names & words:
                weights[name] = 0.5
        scores = Counter()
        matched = defaultdict(set)
        info = {}
        total = len(self.entries)
        for term, weight in sorted(weights.items()):
            posting = self.postings.get(term, {})
            idf = math.log1p((total - len(posting) + 0.5) / (len(posting) + 0.5))
            info[term] = idf * weight
            for i, tf in posting.items():
                if book and self.entries[i].book != book:
                    continue
                norm = 1.2 * (0.25 + 0.75 * self.lengths[i] / self.avg_length)
                scores[i] += weight * idf * tf * 2.2 / (tf + norm)
                if term in words:
                    matched[i].add(term)
        # Два редких слова в соседних репликах могут описывать один обмен.
        # Ограничиваем прибавку и никогда не перескакиваем в другое явление.
        for i in scores:
            nearby = set()
            a = self.entries[i]
            for j in range(max(0, i-3), min(total, i+4)):
                b = self.entries[j]
                if (a.book, a.act, a.section) == (b.book, b.act, b.section):
                    nearby.update(self.bodies[j] & weights.keys())
            scores[i] += sum(weights[v] for v in sorted(matched[i]))
            scores[i] += 0.5 * sum(info[v] for v in sorted(nearby - self.bodies[i]))
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return [self.entries[i] for i in ranked[:limit]]

    def card(self, p: Passage, query: str = "") -> dict:
        book = self.books[p.book]
        words = p.text.split()
        # Цитата в карточке — непрерывная подстрока. Для длинной реплики берём
        # короткое окно вокруг слов запроса, сохраняя авторское написание.
        wanted = set(tokens(query))
        hit = next((n for n, w in enumerate(words) if wanted.intersection(tokens(w))), 0)
        left = min(max(0, hit-5), max(0, len(words)-20))
        quote = " ".join(words[left:left+20])
        if len(words) > 20:
            sentences = [s for s in re.split(r"(?<=[.!?…])\s+", p.text)
                         if 2 <= len(s.split()) <= 20 and s.rstrip('»")').endswith((".", "!", "?", "…"))]
            if sentences:
                best = max(sentences, key=lambda s: (len(wanted.intersection(tokens(s))), len(s.split())))
                if not wanted or wanted.intersection(tokens(best)):
                    quote = best
        return {"ref": p.ref, "book": p.book, "title": book.title, "author": book.author,
                "act": p.act, "section": p.heading, "speaker": p.speaker,
                "quote": quote, "cut_before": not p.text.startswith(quote), "cut_after": not p.text.endswith(quote),
                "lines": [p.first_line, p.last_line],
                "url": self.source(p)}
