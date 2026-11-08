from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).with_name("data")
SCENARIOS = ("extra", "lab404")


@dataclass(frozen=True)
class Actor:
    name: str
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
    cues: list[list[str]]
    challenge: str
    unanimous: str
    nomination: str
    unresolved: str
    archive: Path


def load_scenario(key: str) -> Scenario:
    if key not in SCENARIOS:
        raise ValueError(f"Неизвестная постановка: {key}")
    path = DATA / key
    with (DATA / "cast.toml").open("rb") as file:
        cast = tomllib.load(file)
    with (path / "scenario.toml").open("rb") as file:
        story = tomllib.load(file)
    story["cast"] = {name: Actor(**cast[name]) for name in story["cast"]}
    story["arc"] = tuple(story["arc"])
    story["researchers"] = tuple(story["researchers"])
    return Scenario(key=key, archive=path / "archive", **story)
