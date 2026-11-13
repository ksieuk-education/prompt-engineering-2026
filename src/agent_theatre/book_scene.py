"""Два хода с книгой: актёр выбирает свидетельство, его партнёр проверяет соседний текст."""
from collections.abc import Callable
import re

from .language import parses
from .library import Library, tokens
from .llm import Generator
from .scenario import Actor
from .schema import Delivery, Form, Lookup, Review


class Choice(Form):
    ref: str
    stance: str


def choose(gen: Generator, model: str, actor: Actor, prompt: str, cards: list[dict]) -> Choice:
    extracts = "\n".join(f"{c['ref']} · {c['speaker']}: «{c['quote']}»" for c in cards)
    return gen.ask(model, f"Ты — {actor.name}. Читаешь пьесу и готовишь короткий ответ партнёру.",
        prompt + "\nДоступные строки:\n" + extracts +
        "\nref — ровно один подходящий адрес. Если все строки посторонние, оставь пусто. "
        "stance — что сделаешь со свидетельством собеседника.",
        Choice, choices={"ref": [c["ref"] for c in cards] + [""],
                         "stance": ["уточнить", "возразить", "поддержать", "указать предел аналогии"]}, budget=160)


def find_citation(lib: Library, gen: Generator, model: str, actor: Actor, task: dict, dialogue: str,
                  event: dict, emit: Callable[[dict], None]) -> dict | None:
    book = lib.books[task["book"]]
    prompt = (f"На репетиции книги доступны целиком. Найди у {book.author} в «{book.title}» эпизод. "
              f"Задача для спора: {task['question']}\n{dialogue}\n"
              "В query дай 2–6 поисковых слов: участники книжного эпизода, предмет или действие. "
              "Не используй имена актёров нынешней постановки.")
    for attempt in (1, 2):
        query = gen.ask(model, f"Ты — {actor.name}, работаешь с театральной библиотекой.",
                        prompt, Lookup, budget=120).query.strip()
        hits = [lib.card(p, query) for p in lib.search(query, book=book.key)]
        picked = choose(gen, model, actor, task["question"], hits) if hits else None
        ref = picked.ref if picked else ""
        emit({**event, "event": "book_search", "tool": "search_library", "attempt": attempt,
              "book": book.key, "query": query, "hits": hits, "selected": ref})
        if ref:
            card = next(c for c in hits if c["ref"] == ref)
            return {**event, "event": "citation", "kind": "read", "card": card,
                    "stance": picked.stance, "question": task["question"], "motif": task["motif"]}
        prompt += f"\nПо запросу «{query}» подходящий фрагмент не выбран. Переформулируй другими словами."
    return None


def check_citation(lib: Library, gen: Generator, model: str, actor: Actor, published: dict,
                   event: dict, emit: Callable[[dict], None]) -> dict | None:
    first = published["card"]
    context = lib.context(first["ref"], radius=3)
    cards = [lib.card(p) for p in context]
    others = [c for c in cards if c["ref"] != first["ref"]]
    prompt = (f"Партнёр прочитал у {first['author']} «{first['quote']}». "
              f"Обсуждали: {published['question']} Сейчас проверь чужое чтение. "
              "Выбери соседнюю строку, которая уточняет, меняет или ограничивает вывод.")
    picked = choose(gen, model, actor, prompt, others) if others else None
    ref = picked.ref if picked else ""
    emit({**event, "event": "book_context", "tool": "open_context", "origin": first["ref"],
          "origin_turn": published["index"], "hits": cards, "selected": ref})
    if not ref:
        return None
    return {**event, "event": "citation", "kind": "check", "card": next(c for c in others if c["ref"] == ref),
            "origin": first["ref"], "origin_turn": published["index"], "stance": picked.stance,
            "question": published["question"], "motif": published["motif"]}


def repeats_partner(line: str, previous: str) -> bool:
    # Регистром или потерянной при монтаже кавычкой можно замаскировать целый
    # чужой вопрос. Короткое «Да, я согласен» остаётся допустимым ответом.
    def tokens_of_phrase(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold().replace("ё", "е"))
    heard = tokens_of_phrase(previous)
    return len(heard) >= 5 and tokens_of_phrase(line) == heard


def check_comment(words: Delivery, citation: dict, cast_text: str, previous: str = "") -> Review:
    card = citation["card"]
    if repeats_partner(words.line, previous):
        return Review(issue="book", note="Ты повторил весь вопрос партнёра. Ответь на него своими словами.")
    if (words.line.count("«") != words.line.count("»")
            or re.search(r"(?:с уч[её]том(?: е[её]| его| их)?|потому что|чтобы)\s*(?:\.{3}|…)\s*$", words.line, re.I)):
        return Review(issue="book", note="Фраза оборвалась. Договори мысль одним целым предложением.")
    if re.search(r"[a-z]{2,}", words.line, re.I | re.ASCII):
        return Review(issue="book", note="Весь комментарий должен прозвучать по-русски.")
    if len(words.line.split()) > 36:
        return Review(issue="book", note="Оставь одно-два коротких предложения на русском, всего до 30 слов.")
    def word_forms(text):
        forms = set(tokens(text))
        # У фамилии «Хлестаков» первый разбор без контекста бывает «хлестак»,
        # тогда как у «Хлестакову» первым идёт разбор фамилии.
        for word in re.findall(r"\b[А-ЯЁ][а-яё]+\b", text):
            forms.add(word.casefold().replace("ё", "е"))
            forms.update(p.normal_form.replace("ё", "е") for p in parses(word.casefold())
                         if {"Name", "Surn", "Patr"} & p.tag.grammemes)
        return forms
    support = word_forms(citation["motif"] + " " + card["speaker"] + " " + card["quote"])
    known = support | word_forms(cast_text + " " + citation["question"] + " " + card["title"] + " " + card["author"])
    for word in re.findall(r"\b[А-ЯЁ][а-яё]+\b", words.line):
        parsed = parses(word.casefold())[0]
        surname = re.search(r"(?:ов|ев|ин|ова|ева|ский|ская|ского|скому|овым|евым|ович|овна)$", word, re.I)
        if (({"Name", "Surn", "Patr"} & parsed.tag.grammemes or surname and parsed.tag.POS in {"NOUN", "ADJF"})
                and word.casefold().replace("ё", "е") not in known
                and not any(p.normal_form.replace("ё", "е") in known for p in parses(word.casefold()))):
            return Review(issue="book", note=f"Имя «{word}» не подтверждено открытой карточкой или составом сцены.")
    for quote in re.findall(r'«([^»]+)»|"([^"]+)"', words.line):
        text = next(q for q in quote if q)
        title = len(text.split()) == len(card["title"].split()) and tokens(text) == tokens(card["title"])
        if len(text.split()) > 1 and not title and text.casefold() not in card["quote"].casefold():
            return Review(issue="book", note=f"Убери неподтверждённую цитату «{text}». Выскажи свой довод своими словами, без кавычек.")
    if not support.intersection(word_forms(words.line)):
        return Review(issue="book", note=f"Свяжи свой ответ с конкретной деталью из книги: {citation['motif']}.")
    addressed = re.search(r"\b(я|ты|вы|мы|нам|мне|меня|нас|тебе|тебя|тобой|вам|вас|вами|здесь|наш\w*|ваш\w*|"
                          r"давайте|предлагаю|хочу|отказываюсь|надо|нужно|спросим|нельзя|пусть)\b", words.line, re.I)
    imperative = any("impr" in parses(w.casefold())[0].tag for w in re.findall(r"[А-Яа-яЁё]+", words.line))
    if "?" not in words.line and not addressed and not imperative:
        return Review(issue="book", note="Обратись к своим партнёрам: что именно нам здесь делать после прочитанного?")
    return Review(issue="", note="")


def shorten_comment(words: Delivery, citation: dict, previous: str = "") -> Delivery:
    if len(words.line.split()) <= 36:
        return words
    # Сохраняем полные предложения исходного актёрского дубля. Превращение
    # длинной фразы в оборванную цитату при монтаже слышно особенно отчётливо.
    sentences = [s.strip() for s in re.findall(r'[^.!?…]+(?:[.!?…]+[»"]*)', words.line)]
    candidates = [s for s in sentences if 5 <= len(s.split()) <= 30
                  and not s.rstrip('»"').endswith(("...", "…"))
                  and s.count("«") == s.count("»") and not repeats_partner(s, previous)]
    if not candidates:
        return words
    motif = set(tokens(citation["motif"]))
    best = max(candidates, key=lambda s: (bool(motif.intersection(tokens(s))), "?" in s,
                                          bool(re.search(r"\b(мы|нам|я|ты|вы|наш)\b", s, re.I))))
    return words.model_copy(update={"line": best})


def acting_note(citation: dict) -> str:
    card = citation["card"]
    return (f"Перед твоей репликой зрители услышат подлинный отрывок из «{card['title']}», "
            f"{card['speaker']}: «{card['quote']}». Задача: {citation['question']} "
            f"Твой способ ответа: {citation['stance']}. Одной живой фразой свяжи конкретную деталь из книги "
            "с выбором присутствующего. Если не согласен с аналогией, объясни почему. В книжном эпизоде "
            "вы не участвовали. Цитату прочитает постановка, свой комментарий скажи без кавычек.")


def citation_intro(event: dict) -> str:
    card = event["card"]
    action = "проверяет соседний фрагмент" if event["kind"] == "check" else "открывает книгу"
    speaker = "Авторская ремарка" if card["speaker"] == "Ремарка" else f"В книге говорит {card['speaker']}"
    return f"{event['name']} {action}. «{card['title']}», действие {card['act']}. {speaker}."


def printed_quote(card: dict) -> str:
    quote = card["quote"]
    before = "…" if card.get("cut_before") and quote[:1].islower() else ""
    after = "…" if card.get("cut_after") and not quote.rstrip('»")').endswith((".", "?", "!", "…")) else ""
    return before + quote + after
