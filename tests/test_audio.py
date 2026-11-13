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


def test_mastered_cues_and_vtt_share_the_same_timeline(tmp_path, monkeypatch):
    import wave
    from agent_theatre import audio

    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    speakers = []
    class Synth:
        def apply_tts(self, *, text, speaker, sample_rate):
            assert sample_rate == 48000
            speakers.append(speaker)
            return torch.sin(torch.linspace(0, 20, 4800))

    monkeypatch.setattr(audio, "load_voice_model", lambda _: Synth())
    events = [{"event": "start", "scenario": "extra"},
              {"event": "turn", "actor": "barbie", "stage": "", "line": "Я согласна?"},
              {"event": "ending", "curtain": "Свет погас."}]
    wav, vtt = tmp_path / "play.wav", tmp_path / "play.vtt"
    record(events, wav, captions=vtt)
    with wave.open(str(wav)) as stream:
        assert stream.getframerate() == 48000 and stream.getnchannels() == 1
        count = stream.getnframes()
        pcm = np.frombuffer(stream.readframes(count), dtype="<i2")
    assert 0 < np.abs(pcm).max() <= 29500
    assert speakers == ["baya", "xenia", "baya"]
    assert count == 48000 * (100 + 180 + 100 + 220 + 100 + 550) // 1000
    subtitles = vtt.read_text()
    assert "00:00:00.280 --> 00:00:00.380\n<v Барби>Я согласна?" in subtitles
    assert "00:00:00.600 --> 00:00:00.700" in subtitles
