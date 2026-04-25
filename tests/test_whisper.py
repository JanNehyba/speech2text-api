from utils import test_record_paths

from speech2text_api.whisper import Whisper


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
    # The tiny test model + short clip should still produce at least one
    # usable timestamped chunk.
    assert isinstance(result["segments"], list)
    assert all("start" in s and "end" in s and "text" in s for s in result["segments"])
