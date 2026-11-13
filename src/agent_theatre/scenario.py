from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).with_name("data")
SCENARIOS = ("extra", "lab404")
SPEAKER_GENDER = {"aidar": "male", "eugene": "male", "baya": "female",
                  "kseniya": "female", "xenia": "female"}
NARRATOR = "baya"


@dataclass(frozen=True)
class VoiceRole:
    name: str
    gender: str
    speaker: str

    def __post_init__(self) -> None:
        if self.gender not in {"male", "female"} or SPEAKER_GENDER.get(self.speaker) != self.gender:
            raise ValueError(f"{self.name}: проверьте пол роли и голос {self.speaker!r} в cast.toml")

    @property
    def agreement(self) -> str:
        return "женский род" if self.gender == "female" else "мужской род"


@dataclass(frozen=True)
class Actor(VoiceRole):
    voice: str
    stake: str
    sample: str
    family: list[str]
    tactic: str


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    source: str
    cast: dict[str, Actor]
    researchers: tuple[str, ...]
    setting: str
    question: str
    notice: str
    arc: tuple[str, ...]
    blocking: list[str]
    cues: list[list[str]]
    library_tasks: list[dict[str, str]]
    challenge: str
    unanimous: str
    nomination: str
    unresolved: str
    archive: Path


def load_cast() -> dict[str, Actor]:
    with (DATA / "cast.toml").open("rb") as file:
        return {key: Actor(**card) for key, card in tomllib.load(file).items()}


def load_scenario(key: str) -> Scenario:
    if key not in SCENARIOS:
        raise ValueError(f"Неизвестная постановка: {key}")
    path = DATA / key
    cast = load_cast()
    with (path / "scenario.toml").open("rb") as file:
        story = tomllib.load(file)
    story["cast"] = {name: cast[name] for name in story["cast"]}
    voices = [NARRATOR, *(actor.speaker for actor in story["cast"].values())]
    if len(voices) != len(set(voices)):
        raise ValueError(f"{key}: назначьте каждой роли и рассказчице отдельный голос")
    story["arc"] = tuple(story["arc"])
    story["researchers"] = tuple(story["researchers"])
    return Scenario(key=key, archive=path / "archive", **story)
