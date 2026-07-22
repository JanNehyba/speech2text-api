from types import SimpleNamespace
from unittest.mock import MagicMock

from utils import test_record_paths

from speech2text_api.whisper import Whisper, _parse_temperature_ladder


# ---------------------------------------------------------------------------
# Anti-hallucination params + hotwords (2026-07-22) — mock-based, no model load
# ---------------------------------------------------------------------------


def _mock_wrapper(monkeypatch):
    """A Whisper wrapper whose model + audio decode are mocked, so we can
    assert the kwargs passed to model.transcribe() without a real model."""
    w = Whisper.__new__(Whisper)  # skip __init__ (which loads the model)
    w.beam_size = 3
    w.vad_filter = False
    w.temperature = (0.0, 0.2, 0.4)
    w.condition_on_previous_text = False
    w.hallucination_silence_threshold = 2.0
    seg = SimpleNamespace(start=0.0, end=1.0, text="ahoj", words=[])
    w.model = MagicMock()
    w.model.transcribe.return_value = (iter([seg]), SimpleNamespace())
    import speech2text_api.whisper as mod
    monkeypatch.setattr(mod.librosa, "load", lambda *a, **k: ([0.0], 16000))
    return w


def test_transcribe_passes_anti_hallucination_params(monkeypatch, tmp_path):
    w = _mock_wrapper(monkeypatch)
    w.transcribe_file(str(tmp_path / "x.wav"), lang="cs")
    _, kwargs = w.model.transcribe.call_args
    assert kwargs["temperature"] == (0.0, 0.2, 0.4)
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["hallucination_silence_threshold"] == 2.0


def test_transcribe_passes_hotwords(monkeypatch, tmp_path):
    w = _mock_wrapper(monkeypatch)
    w.transcribe_file(str(tmp_path / "x.wav"), lang="cs", hotwords="RVP ČŠI Cermat")
    _, kwargs = w.model.transcribe.call_args
    assert kwargs["hotwords"] == "RVP ČŠI Cermat"


def test_transcribe_empty_hotwords_becomes_none(monkeypatch, tmp_path):
    w = _mock_wrapper(monkeypatch)
    w.transcribe_file(str(tmp_path / "x.wav"), lang="cs", hotwords="   ")
    _, kwargs = w.model.transcribe.call_args
    assert kwargs["hotwords"] is None


def test_temperature_ladder_parsing():
    assert _parse_temperature_ladder("0.0,0.2,0.4") == (0.0, 0.2, 0.4)
    assert _parse_temperature_ladder("0.0") == 0.0
    assert _parse_temperature_ladder("xyz") == (0.0, 0.2, 0.4)
    assert _parse_temperature_ladder("") == (0.0, 0.2, 0.4)


def test_whisper_cs() -> None:
    wrapper = Whisper()
    result = wrapper.transcribe_file(test_record_paths["cs"], lang="cs")

    assert isinstance(result, dict)
    assert "testovací" in result["transcript"]
    assert result["segments"] is None  # not requested


def test_whisper_en() -> None:
    wrapper = Whisper()
    result = wrapper.transcribe_file(test_record_paths["en"], lang="en")

    assert isinstance(result, dict)
    assert "evacuation" in result["transcript"].lower()


def test_whisper_with_timestamps() -> None:
    wrapper = Whisper()
    result = wrapper.transcribe_file(
        test_record_paths["cs"], lang="cs", return_timestamps=True,
    )

    assert "testovací" in result["transcript"]
    # faster-whisper always emits segments internally; with return_timestamps
    # we expose them. Tiny model on a short clip should produce at least one.
    assert isinstance(result["segments"], list)
    assert all("start" in s and "end" in s and "text" in s for s in result["segments"])
    assert all(s["end"] >= s["start"] for s in result["segments"])
    # word_timestamps not requested → words omitted.
    assert result["words"] is None


def test_whisper_with_word_timestamps() -> None:
    wrapper = Whisper()
    result = wrapper.transcribe_file(
        test_record_paths["cs"], lang="cs", word_timestamps=True,
    )

    assert "testovací" in result["transcript"]
    # Flat top-level list of {word, start, end} the diarization client uses
    # to split a segment at a speaker boundary.
    assert isinstance(result["words"], list)
    assert len(result["words"]) >= 1
    assert all("word" in w and "start" in w and "end" in w for w in result["words"])
    assert all(w["end"] >= w["start"] for w in result["words"])
