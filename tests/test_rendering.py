import csv
import pytest
from main import main


@pytest.mark.parametrize("seconds,expected", [(0.12, .12), (.025, .02)])
def test_actual_pygame_entrypoint_renders_and_writes_results(tmp_path, monkeypatch, seconds, expected):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    screenshot = tmp_path / "frame.png"
    results = tmp_path / "episodes.csv"
    assert main(["--headless", "--seconds", str(seconds), "--results", str(results),
                 "--screenshot", str(screenshot), "--debug"]) == 0
    assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "session_end" in results.read_text()
    with results.open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["survival_duration_s"]) == expected
