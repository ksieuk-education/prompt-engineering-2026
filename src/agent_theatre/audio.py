"""Озвучка готового журнала. Для чтения и генерации пьесы PyTorch не требуется."""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
from html import escape
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import wave

from .reading import read_performance
from .book_scene import citation_intro, printed_quote
from .language import stage_direction
from .scenario import NARRATOR, VoiceRole, load_cast, load_scenario


MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_6_ru.pt"
MODEL_SHA256 = "04cc9b1d08ce38bcaa9d02c488d80d3e6806d17f7eaa7c95982f36a3cbf50b2e"
LEGACY_SHA256 = "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00"
VOICES = {key: actor.speaker for key, actor in load_cast().items()}
RATE = 48_000


@dataclass(frozen=True)
class Cue:
    speaker: str
    name: str
    text: str
    pause_ms: int


def casting(events: list[dict]) -> dict[str, VoiceRole]:
    start = next((e for e in events if e["event"] == "start"), {})
    if "cast" in start:
        roles = {key: VoiceRole(**card) for key, card in start["cast"].items()}
        voices = [NARRATOR, *(role.speaker for role in roles.values())]
        if len(voices) != len(set(voices)):
            raise ValueError("в журнале один голос назначен нескольким участникам")
        return roles
    return load_scenario(start["scenario"]).cast if "scenario" in start else load_cast()


def cues_for(events: list[dict]) -> Iterator[Cue]:
    """Первая реплика знакомит слушателя с голосом, ремарки звучат у рассказчицы."""
    if not events or events[-1].get("event") != "ending":
        raise ValueError("для озвучки нужен завершённый спектакль с развязкой")
    roles = casting(events)
    introduced = set()
    ordinals = ("первая", "вторая", "третья", "четвёртая")
    for e in events:
        if e["event"] == "scene":
            yield Cue(NARRATOR, "Рассказчица", f"Сцена {ordinals[e['scene'] - 1]}. {e['title']}.", 450)
            yield Cue(NARRATOR, "Рассказчица", e["opening"], 380)
            if e.get("action"):
                yield Cue(NARRATOR, "Рассказчица", e["action"], 380)
            if e["notice"]:
                yield Cue(NARRATOR, "Табло", "На табло. " + e["notice"], 550)
        elif e["event"] == "citation":
            role = roles[e["actor"]]
            yield Cue(NARRATOR, "Рассказчица", citation_intro(e), 180)
            yield Cue(role.speaker, role.name + " читает", printed_quote(e["card"]), 320)
        elif e["event"] == "turn":
            role = roles[e["actor"]]
            if e["actor"] not in introduced:
                yield Cue(NARRATOR, "Рассказчица", role.name + ".", 180)
                introduced.add(e["actor"])
            if e["stage"]:
                yield Cue(NARRATOR, "Рассказчица", stage_direction(e["stage"], role), 180)
            yield Cue(role.speaker, role.name, e["line"], 220 if e["line"].rstrip().endswith("?") else 330)
        elif e["event"] == "ending":
            yield Cue(NARRATOR, "Рассказчица", e["curtain"], 550)


def score(events: list[dict]) -> Iterator[tuple[str, str]]:
    for cue in cues_for(events):
        yield cue.speaker, cue.text


def spoken_text(text: str) -> str:
    text = re.sub(r"[«»_]", "", text.replace("**", "*"))
    for name, stressed in (("Гермион", "Герми+он"), ("Чендлер", "Ч+ендлер"), ("Барби", "Б+арби")):
        text = re.sub(r"\b" + name, stressed, text, flags=re.I)
    return text.replace("№ 404", "номер четыреста четыре").strip()


def master(clip, narrator: bool):
    import numpy as np

    sound = clip.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1)
    if not len(sound) or not np.isfinite(sound).all():
        raise ValueError("синтезатор вернул пустой или повреждённый звук")
    peak = float(np.max(np.abs(sound)))
    voiced = np.flatnonzero(np.abs(sound) > max(0.001, peak * 0.008))
    if not len(voiced):
        raise ValueError("синтезатор вернул тишину вместо реплики")
    margin = RATE // 30
    sound = sound[max(0, voiced[0]-margin):voiced[-1]+margin+1]
    active = sound[np.abs(sound) > peak * 0.06]
    target = 0.13 if narrator else 0.15
    gain = min(0.90 / peak, float(np.clip(target / np.sqrt(np.mean(active**2)), 0.65, 1.65)))
    sound *= gain
    fade = min(RATE // 200, len(sound) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        sound[:fade] *= ramp
        sound[-fade:] *= ramp[::-1]
    return (sound * 32767).round().astype("<i2")


def timestamp(frame: int) -> str:
    total_ms = round(frame * 1000 / RATE)
    seconds, millis = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def load_voice_model(path: Path | None):
    import torch

    if path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "silero-v5-6-ru.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(MODEL_URL, str(path), hash_prefix=MODEL_SHA256)
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() not in {MODEL_SHA256, LEGACY_SHA256}:
            raise ValueError(f"контрольная сумма модели Silero не совпала: {path}")
        source.seek(0)
        model = torch.package.PackageImporter(source).load_pickle("tts_models", "model")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model.to(torch.device("cpu"))
    return model


def record(events: list[dict], out: Path, model_path: Path | None = None, *, captions: Path | None = None) -> None:
    # Ни прерванный синтез, ни параллельный запуск не заменяют готовую фонограмму.
    cues = list(cues_for(events))
    if out.exists():
        raise FileExistsError(f"Фонограмма уже существует: {out}")
    if captions is not None and (captions.exists() or captions.resolve() == out.resolve()):
        raise FileExistsError(f"Для субтитров нужен отдельный новый файл: {captions}")
    import torch

    model = load_voice_model(model_path)
    subtitles = ["WEBVTT", ""]
    frame = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".wav") as partial:
        with wave.open(partial, "wb") as wav, torch.inference_mode():
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            for cue in cues:
                clip = model.apply_tts(text=spoken_text(cue.text), speaker=cue.speaker, sample_rate=RATE)
                pcm = master(clip, cue.speaker == NARRATOR)
                wav.writeframes(pcm.tobytes())
                if captions is not None:
                    subtitles.extend((f"{timestamp(frame)} --> {timestamp(frame + len(pcm))}",
                                      f"<v {escape(cue.name)}>{escape(cue.text)}</v>", ""))
                pause = round(RATE * cue.pause_ms / 1000)
                wav.writeframes(bytes(pause * 2))
                frame += len(pcm) + pause
        partial.flush()
        if captions is None:
            os.link(partial.name, out)
        else:
            captions.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=captions.parent, mode="w", encoding="utf-8") as vtt:
                vtt.write("\n".join(subtitles))
                vtt.flush()
                os.link(vtt.name, captions)
                try:
                    os.link(partial.name, out)
                except OSError:
                    captions.unlink()
                    raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Озвучить завершённую пьесу голосами Silero v5.6")
    parser.add_argument("run", type=Path, help="папка с trace.jsonl")
    parser.add_argument("--out", type=Path, help="новый WAV-файл, по умолчанию play.wav в папке спектакля")
    parser.add_argument("--model", type=Path, help="скачанный пакет Silero v5_6_ru.pt или прежний v4_ru.pt")
    parser.add_argument("--captions", type=Path, help="новый WebVTT с именами и точными границами реплик")
    args = parser.parse_args(argv)
    out = args.out or args.run / "play.wav"
    try:
        record(read_performance(args.run), out, args.model, captions=args.captions)
    except ModuleNotFoundError as exc:
        parser.exit(1, f"Нужны аудиозависимости: poetry install --with audio ({exc.name})\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Не удалось записать фонограмму: {exc}\n")
    print(f"Фонограмма готова: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
