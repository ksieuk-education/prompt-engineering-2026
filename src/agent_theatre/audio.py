"""Озвучка готового журнала. Для чтения и генерации пьесы PyTorch не требуется."""

import argparse
from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import wave

from .analysis import read_trace


MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
MODEL_SHA256 = "896ab96347d5bd781ab97959d4fd6885620e5aab52405d3445626eb7c1414b00"
VOICES = {"tony": "eugene", "professor": "eugene", "chandler": "aidar",
          "hermione": "kseniya", "barbie": "xenia"}
NARRATOR = "baya"
RATE = 24_000


def score(events: list[dict]) -> Iterator[tuple[str, str]]:
    """Первая реплика знакомит слушателя с голосом, ремарки звучат у рассказчицы."""
    if not events or events[-1].get("event") != "ending":
        raise ValueError("для озвучки нужен завершённый спектакль с развязкой")
    introduced = set()
    ordinals = ("первая", "вторая", "третья", "четвёртая")
    for e in events:
        if e["event"] == "scene":
            yield NARRATOR, f"Сцена {ordinals[e['scene'] - 1]}. {e['title']}."
            yield NARRATOR, e["opening"]
            if e["notice"]:
                yield NARRATOR, "На табло. " + e["notice"]
        elif e["event"] == "turn":
            if e["actor"] not in introduced:
                yield NARRATOR, e["name"] + "."
                introduced.add(e["actor"])
            if e["stage"]:
                yield NARRATOR, e["stage"]
            yield VOICES[e["actor"]], e["line"]
        elif e["event"] == "ending":
            yield NARRATOR, e["curtain"]


def load_voice_model(path: Path | None):
    import torch

    if path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "silero-v4-ru.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(MODEL_URL, str(path), hash_prefix=MODEL_SHA256)
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError(f"контрольная сумма модели Silero v4_ru не совпала: {path}")
        source.seek(0)
        model = torch.package.PackageImporter(source).load_pickle("tts_models", "model")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model.to(torch.device("cpu"))
    return model


def record(events: list[dict], out: Path, model_path: Path | None = None) -> None:
    # Ни прерванный синтез, ни параллельный запуск не заменяют готовую фонограмму.
    cues = list(score(events))
    if out.exists():
        raise FileExistsError(f"Фонограмма уже существует: {out}")
    import torch

    model = load_voice_model(model_path)
    pause = bytes(2 * RATE // 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".wav") as partial:
        with wave.open(partial, "wb") as wav, torch.inference_mode():
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            for voice, text in cues:
                spoken = re.sub(r"[*_«»]", "", text).strip()
                clip = model.apply_tts(text=spoken, speaker=voice, sample_rate=RATE)
                pcm = (clip.clamp(-1, 1) * 32767).round().to(torch.int16)
                wav.writeframes(pcm.numpy().astype("<i2", copy=False).tobytes())
                wav.writeframes(pause)
        partial.flush()
        os.link(partial.name, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Озвучить завершённую пьесу голосами Silero v4_ru")
    parser.add_argument("run", type=Path, help="папка с trace.jsonl")
    parser.add_argument("--out", type=Path, help="новый WAV-файл, по умолчанию play.wav в папке спектакля")
    parser.add_argument("--model", type=Path, help="скачанный пакет Silero v4_ru.pt для автономного запуска")
    args = parser.parse_args(argv)
    out = args.out or args.run / "play.wav"
    try:
        record(read_trace(args.run), out, args.model)
    except ModuleNotFoundError as exc:
        parser.exit(1, f"Нужны аудиозависимости: poetry install --with audio ({exc.name})\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Не удалось записать фонограмму: {exc}\n")
    print(f"Фонограмма готова: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
