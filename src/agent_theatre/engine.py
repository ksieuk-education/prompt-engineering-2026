"""Один вызов актёра — один ход. Актёры узнают чужие позиции из произнесённых реплик."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import partial
import re

from .archive import Archive, Hit
from .book_scene import acting_note, check_citation, check_comment, find_citation, shorten_comment
from .library import Library
from .language import stage_direction
from .llm import Generator
from .rehearsal import BookCommentRejected, deliver, mentions_clock, review_line
from .scenario import Scenario
from .schema import Intention, Plan, Speech, Turn
from .stage_tools import quorum, use_skill


def choose_speaker(cast: list[str], history: list[Turn], eligible: list[str],
                   opener: str | None = None, *, new_scene: bool = False) -> tuple[str, str]:
    """Прямой вопрос важнее очереди, пока он не вытесняет остальных из сцены."""
    if not eligible:
        raise ValueError("нет доступных актёров")
    last = history[-1] if history else None
    candidates = [a for a in eligible if last is None or a != last.actor] or eligible
    last_seen = dict.fromkeys(cast, -1)
    for index, turn in enumerate(history):
        last_seen[turn.actor] = index
    oldest = min(candidates, key=last_seen.__getitem__)
    if len(history) - 1 - last_seen[oldest] >= len(cast) + 1:
        return oldest, "давно не получал слова"
    if opener in candidates:
        return opener, "инициатива в новой сцене"
    if not new_scene and last and last.speech.to in candidates:
        return last.speech.to, f"обращение от {last.actor}"
    return oldest, "дольше других ждёт свободного слова"


def resolve(story: Scenario, votes: dict[str, str]) -> tuple[str, str]:
    if set(votes) != set(story.cast):
        raise ValueError("нужны решения всех четырёх персонажей")
    allowed = {"challenge", "abstain", *(f"select:{key}" for key in story.cast)}
    if not set(votes.values()) <= allowed:
        raise ValueError("неизвестный вариант решения")
    count = Counter(votes.values())
    if count["challenge"] >= quorum("challenge", len(votes)):
        return "challenge", story.unanimous
    choice, n = count.most_common(1)[0]
    if choice.startswith("select:") and n >= quorum("nomination", len(votes)):
        name = story.cast[choice.removeprefix("select:")].name
        return choice, story.nomination.format(name=name)
    return "unresolved", story.unresolved


def make_plan(story: Scenario, gen: Generator, model: str) -> Plan:
    cast = ", ".join(actor.name for actor in story.cast.values())
    return gen.ask(model, "Ты драматург русскоязычной камерной пьесы.",
        f"Пьеса «{story.title}». {story.setting}\nВ ролях: {cast}.\n"
        f"Проблема: {story.question}\nОбъявление во второй сцене: {story.notice}\n"
        "Составь макроплан до игры: ровно четыре сцены в этом порядке.\n"
        + "\n".join(f"{i}. {beat} Действие на сцене: {action}"
                    for i, (beat, action) in enumerate(zip(story.arc, story.blocking, strict=True), 1))
        + "\nДля каждой придумай title (заглавие), opening (одно короткое движение, свет или звук "
          "до слов героев, до 15 слов), objective (конкретная задача этого спора). Сохрани реквизит. "
          "Текст табло выведет сама программа. Оставь за кадром надписи, готовые реплики, "
          "новых героев и предсказания голосования. Во всех четырёх opening дверь ещё закрыта. "
          "Напиши по-русски объект со списком scenes.", Plan, budget=900)


def history_text(story: Scenario, history: list[Turn]) -> str:
    lines = []
    for t in history[-5:]:
        gesture = f" ({t.speech.stage})" if t.speech.stage else ""
        lines.append(f"{story.cast[t.actor].name}{gesture}: {t.speech.line}")
    return "\n".join(lines)


def pending_question(actor: str, history: list[Turn]) -> Turn | None:
    """Вопрос остаётся адресованным герою, даже если режиссёр дал слово третьему."""
    for turn in reversed(history):
        if turn.actor == actor:
            break
        if turn.speech.to == actor and "?" in turn.speech.line:
            return turn
    return None


def named_addressee(line: str, story: Scenario, candidates: list[str]) -> str | None:
    """Партнёр, которого позвали вслух, может отличаться от первоначального намерения."""
    for key in candidates:
        names = [story.cast[key].name, *story.cast[key].name.split()]
        forms = "|".join(re.escape(n) for n in names)
        if re.search(rf"(?:^|[.!?,]\s+)(?:мистер |мисс )?(?:{forms})\s*[,!?.]", line, re.I):
            return key
    return None


def ballot_phrase(story: Scenario, actor: str, vote: str) -> str:
    """Стандартная формула регистрации голоса. Обоснование пишет сам актёр."""
    if vote == "abstain":
        return "Я воздержусь."
    if vote == "challenge":
        return ("Я за совместный пересмотр." if story.key == "extra"
                else "Я отказываюсь определять, кто из нас настоящий.")
    nominee = vote.removeprefix("select:")
    if nominee == actor:
        return "Я остаюсь здесь." if story.key == "extra" else "Я называю себя."
    return f"Мой выбор — {story.cast[nominee].name}."


def play_turn(gen: Generator, model: str, story: Scenario, actor_id: str, context: str,
              seen: dict[str, Hit], eligible: list[str], final: bool, prior: list[str],
              clock_discussed: bool, record: Callable[[dict], None], editor: str | None,
              anchor: Turn | None = None, citation: dict | None = None) -> Speech:
    """Актёр сначала принимает решение, затем говорит, не видя служебных полей и путей."""
    actor = story.cast[actor_id]
    role = (f"Ты — {actor.name}. Играешь в русскоязычной камерной трагикомедии. "
            f"О себе говори в форме «я» ({actor.agreement}). {actor.voice}\nТвоя личная история в этой постановке: {actor.stake}\n"
            f"Твой способ вести спор: {actor.tactic}\n"
            f"Пример ритма речи: «{actor.sample}». "
            "Личное прошлое бери только из своей истории. Можно сомневаться, уступать и менять мнение.")
    labels = ", ".join(f"{key} = {a.name} ({a.agreement})" for key, a in story.cast.items())
    notes = "\n".join(f"{ref} — {h.text}" for ref, h in seen.items())
    decisions = {"challenge": story.challenge, "abstain": "воздержаться от решения"}
    for key, a in story.cast.items():
        meaning = "оставить в комнате" if story.key == "extra" else "признать настоящим"
        decisions[f"select:{key}"] = f"{meaning} одного участника ({a.name})"
    vote_rules = ""
    if final:
        vote_rules = ("Вспомни обещание, которое ты дал вслух, и предложение партнёра. Голосуй в "
                      "соответствии со своей позицией. Если передумал, в реплике потребуется причина. "
                      "Четыре возражения открывают пересмотр; "
                      "три одинаковых имени принимаются терминалом. Значения поля vote:\n"
                      + "\n".join(f"{k} — {v}" for k, v in decisions.items()))
    tactics = ["ответить прямо", "возразить", "предложить проверку", "уступить", "признать незнание",
               "уточнить сказанное", "объяснить личную цену"]
    move = gen.ask(model, role,
        context + "\nПодготовь свой ход за кулисами. В intent выбери способ ответа партнёру. "
        f"to — адресат или пусто. Участники: {labels}.\n"
        f"Твои архивные карточки:\n{notes or '(пока пусто)'}\n"
        f"source — одна карточка, на которую опираешься, иначе пусто. {vote_rules}",
        Intention, choices={"intent": tactics, "to": ["", *(key for key in eligible if key != actor_id)],
                            "source": ["", *seen], "vote": list(decisions) if final else [""]},
        budget=330)
    cue = f"Способ вести эту реплику — {move.intent}."
    if move.to:
        partner = story.cast[move.to]
        cue += f" После ответа можно передать слово: {partner.name} ({partner.agreement})."
    binding = f"Следующий адресат — {story.cast[move.to].name}." if move.to else "Обратись к присутствующим."
    if final:
        # В намерении могут фигурировать чужие предложения. Для финального обоснования
        # передаём только зарегистрированный выбор, особенно если актёр выбрал себя.
        pledge = ballot_phrase(story, actor_id, move.vote)
        cue = (f"Твой голос уже принят: «{pledge}». Выбранное действие: {decisions[move.vote]}. "
               "Произнеси только причину: что ты готов потерять или за что берёшь ответственность? "
               "Формулу выбора перед причиной добавит табло.")
        if story.key == "extra" and move.vote == f"select:{actor_id}":
            cue += " Если решение примут, остальные уйдут, а ты останешься в комнате. Осознай эту цену."
        binding = cue
    if move.source:
        fact = f" На твоей карточке: «{seen[move.source].text}». Используй один точный факт из неё в обосновании."
        cue += fact
        binding += fact
    prompt = (context + f"\n{cue}\nСыграй живой разговор. Сначала коротко ответь на то, что партнёр спросил "
              "или предложил. Затем, если нужно, добавь одно действие, сомнение или уступку. "
              "Когда ответа нет, признай это. Не приписывай партнёру слов, которых он не произносил. "
              "Одна законченная реплика до 30 слов. Имя собеседника можно опустить. "
              "Вслух говори только за себя. По желанию добавь отдельно одно видимое движение "
              "от третьего лица. За экран и дверь отвечает постановщик.")
    review = partial(review_line, gen, editor, actor, context, binding,
                     addressed_line=anchor.speech.line if anchor else "", final=final) if editor else None
    clean = (context + f"\n{binding}\nСыграй новый короткий ответ. Факты твоей жизни: {actor.stake} "
             "Если этих фактов недостаточно, скажи партнёру, почему доказательства пока нет. "
             "Вслух говори только за себя. Жест можно опустить.")
    if citation:
        cast_text = " ".join(a.name + " " + a.stake for a in story.cast.values())
        previous_line = anchor.speech.line if anchor else ""
        ordinary_review = review
        def book_review(words):
            verdict = check_comment(words, citation, cast_text, previous_line)
            return ordinary_review(words) if not verdict.issue and ordinary_review else verdict
        review = book_review
        role = f"Ты — {actor.name}, {actor.agreement}. {actor.voice} Твоя личная история: {actor.stake}"
        last = f"Перед этим партнёр сказал: «{anchor.speech.line}»." if anchor else ""
        prompt = (f"Вы спорите вчетвером. {story.notice} {last}\n{acting_note(citation)}\n"
                  f"Детали для разговора: {citation['motif']}. "
                  "Книжный текст уже прочитан. Теперь скажи, что ТЫ предлагаешь сделать здесь, "
                  "в связи с одной услышанной деталью. Один короткий ответ или вопрос нынешнему партнёру. "
                  "Только своими словами, без кавычек, до 25 слов. Закулисные метки не произносить. "
                  "Жест можно опустить.")
        clean = prompt
    words, edited, issues = deliver(gen, model, role, prompt, actor, prior,
                                    final and story.key == "extra" and move.vote == f"select:{actor_id}",
                                    clock_discussed, record, review=review, repair_prompt=clean,
                                    partners=[a for key, a in story.cast.items() if key != actor_id],
                                    addressee=story.cast.get(move.to or (anchor.actor if anchor else "")),
                                    addressed_line=anchor.speech.line if anchor else "",
                                    shorten=partial(shorten_comment, citation=citation, previous=previous_line) if citation else None)
    line = words.line
    if final and not line.startswith(pledge):
        line = f"{pledge} {line}"
    partners = [key for key in eligible if key != actor_id]
    addressed = named_addressee(words.line, story, partners) or move.to
    gesture = stage_direction(words.stage, actor) if words.stage != words.line and len(words.stage.split()) <= 16 else ""
    return Speech(line=line, stage=gesture, to=addressed, source=move.source, vote=move.vote,
                  edited=edited, quality_issues=issues)


def perform(story: Scenario, gen: Generator, models: dict[str, str], planner: str,
            turns_per_scene: int, emit: Callable[[dict], None], *, plan: Plan | None = None,
            editor: str | None = None, library: Library | None = None) -> None:
    """Опубликованные события образуют общий журнал. Готовый план позволяет повторить опыт."""
    if not 4 <= turns_per_scene <= 10:
        raise ValueError("для первых трёх сцен допустимо 4–10 ходов на сцену")
    if set(models) != set(story.cast) or len(set(models.values())) < 2:
        raise ValueError("назначьте каждому актёру модель, всего хотя бы две разные")
    cast = list(story.cast)
    archive = Archive(story.archive)
    notebooks: dict[str, dict[str, Hit]] = {key: {} for key in cast}
    history: list[Turn] = []
    public_readings: list[dict] = []
    votes: dict[str, str] = {}
    scene_openers = {1: cast[0], 2: story.researchers[-1], 3: "chandler"}

    reused = plan is not None
    if plan is None:
        plan = make_plan(story, gen, planner)
    emit({"event": "plan", "model": planner, "reused": reused, **plan.model_dump()})
    for scene, (beat, outline) in enumerate(zip(story.arc, plan.scenes, strict=True), 1):
        final = scene == len(plan.scenes)
        action = story.blocking[scene - 1]
        emit({"event": "scene", "scene": scene, "title": outline.title,
              "opening": outline.opening, "action": action, "notice": story.notice if scene == 2 else ""})
        checked: set[str] = set()
        public_citation = None
        previous_arguments: dict[str, Turn] = {}
        if scene >= 3:
            for turn in history:
                if turn.scene == scene - 1:
                    previous_arguments[turn.actor] = turn
        for index in range(len(cast) if final else turns_per_scene):
            eligible = [key for key in cast if key not in votes]
            candidates = eligible
            if scene == 1:
                unheard = [key for key in cast if all(t.actor != key for t in history)]
                if turns_per_scene - index <= len(unheard):
                    candidates = unheard
            # Во второй сцене табло даёт повод исследователю первым проверить условие.
            opener = scene_openers.get(scene)
            actor_id, reason = choose_speaker(cast, history, candidates, opener if index == 0 else None,
                                              new_scene=index == 0)
            actor = story.cast[actor_id]
            model = models[actor_id]
            context = (f"Ты — {actor.name}. {story.setting}\nСейчас сцена {scene}: {outline.title}. "
                       f"{beat}\nНа сцене сейчас: {outline.opening}\nУже произошло при всех: {action}\n"
                       f"Задача сцены: {outline.objective}\n")
            if scene >= 2:
                context += f"Сообщение на табло: {story.notice}\n"
            in_scene = [turn for turn in history if turn.scene == scene]
            if previous_arguments:
                context += ("Что участники успели сказать в прошлом действии:\n"
                            + history_text(story, list(previous_arguments.values())) + "\n")
            # Новое объявление меняет разговор. При переходе сохраняем одну последнюю
            # реплику, дальше актёр слышит партнёров текущей сцены.
            heard = in_scene or ([] if previous_arguments else history[-1:])
            context += "Последние слова на сцене:\n" + (history_text(story, heard) or "(тишина)") + "\n"
            if public_readings:
                context += "Уже прочитали вслух при всех:\n" + "\n".join(
                    f"{e['name']} открывает «{e['card']['title']}», в книге {e['card']['speaker']}: "
                    f"«{e['card']['quote']}»." for e in public_readings[-2:]) + "\n"
            prior = [turn.speech.line for turn in history if turn.actor == actor_id]
            question = pending_question(actor_id, in_scene)
            cues = story.cues[scene - 1]
            cue = cues[min(index * len(cues) // turns_per_scene, len(cues) - 1)]
            context += (f"Направление действия: {cue}\nДверь пока закрыта. Гипотезу или предложение "
                        "формулируй как гипотезу. Действие или результат измерения можно считать "
                        "состоявшимся лишь при наличии публичной ремарки или записи.\n")
            if scene == 3 and index >= turns_per_scene - 2:
                context += ("До голосования осталось время договориться. Если конкретного предложения "
                            "ещё нет, вынеси своё. Если партнёр его уже сделал, скажи, поддерживаешь ли "
                            "ты его и на каких условиях. Обещай только за себя.\n")
            # При смене действия новое объявление имеет приоритет. Внутри действия
            # прерванный прямой вопрос важнее последней произнесённой фразы.
            anchor = question or (in_scene[-1] if in_scene else None)
            if anchor:
                context += (f"Сейчас ты отвечаешь на слова {story.cast[anchor.actor].name} "
                            f"({story.cast[anchor.actor].agreement}): «{anchor.speech.line}». "
                            "Ответь по существу, затем развивай задачу сцены. Если собеседник ошибся, "
                            "можно прямо исправить его или предложить проверку.\n")
            if final and votes:
                context += "Уже произнесённые решения:\n" + "\n".join(
                    f"{story.cast[a].name} — {ballot_phrase(story, a, vote)}" for a, vote in votes.items()) + "\n"
            seen = notebooks[actor_id]
            result = use_skill(gen, model, story, actor_id, scene, context, history, archive, checked, seen, emit)
            if result:
                context += "Что удалось проверить перед этой репликой: " + result + "\n"
            citation = None
            if library and scene in (2, 3):
                event = {"scene": scene, "index": len(history) + 1, "actor": actor_id,
                         "name": actor.name, "model": model}
                if index == 0:
                    citation = find_citation(library, gen, model, actor, story.library_tasks[scene-2],
                                             history_text(story, history[-2:]), event, emit)
                elif index == 1 and public_citation and actor_id != public_citation["actor"]:
                    citation = check_citation(library, gen, model, actor, public_citation, event, emit)
            def record(draft: dict) -> None:
                emit({"event": "book_trim" if "shortened_to" in draft else "retake", "scene": scene, "index": len(history) + 1,
                      "actor": actor_id, "model": model, **draft})

            clock_discussed = (story.key == "extra" and scene in (2, 3)
                               and len({t.actor for t in heard[-3:] if mentions_clock(t.speech.line)}) >= 2)
            play = partial(play_turn, gen, model, story, actor_id, seen=seen, eligible=eligible, final=final,
                           prior=prior, clock_discussed=clock_discussed, record=record, editor=editor, anchor=anchor)
            if citation:
                try:
                    speech = play(context + acting_note(citation), citation=citation)
                except BookCommentRejected as exc:
                    emit({**event, "event": "book_withdrawn", "ref": citation["card"]["ref"], "reason": str(exc)})
                    citation = None
                    speech = play(context)
            else:
                speech = play(context)
            if citation:
                emit(citation)
                public_citation = citation
                public_readings.append(citation)
            history.append(Turn(actor_id, scene, speech, len(history) + 1))
            if final:
                votes[actor_id] = speech.vote
            emit({"event": "turn", "scene": scene, "index": len(history), "actor": actor_id,
                  "name": actor.name, "model": model, "reason": reason, "reply_to": anchor.index if anchor else None,
                  **speech.model_dump()})
    outcome, curtain = resolve(story, votes)
    emit({"event": "ending", "outcome": outcome, "votes": votes, "curtain": curtain})
