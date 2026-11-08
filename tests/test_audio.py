from pathlib import Path

import pytest

from agent_theatre.analysis import read_trace
from agent_theatre.audio import NARRATOR, VOICES, record, score


def test_audio_tracks_the_accepted_dialogue_and_gives_each_actor_a_voice():
    events = read_trace(Path(__file__).resolve().parents[1] / "examples/extra")
    events.insert(4, {"event": "retake", "line": "Это отброшенный дубль."})
    cues = list(score(events))
    turns = [e for e in events if e["event"] == "turn"]
    dialogue = [(voice, words) for voice, words in cues if voice != NARRATOR]
    assert dialogue == [(VOICES[t["actor"]], t["line"]) for t in turns]
    assert len({v for v, _ in dialogue}) == 4
    assert cues[-1] == (NARRATOR, events[-1]["curtain"])
    assert (NARRATOR, "Тони Старк.") in cues


def test_partial_run_cannot_create_a_soundtrack(tmp_path):
    target = tmp_path / "play.wav"
    with pytest.raises(ValueError, match="завершённый"):
        record([{"event": "error"}], target)
    assert not target.exists()
