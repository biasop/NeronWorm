"""CSV episode aggregates, written once per terminal event or session close."""
from dataclasses import asdict, dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class EpisodeStats:
    seed: int
    episode: int
    reason: str
    survival_duration_s: float
    food_eaten: int
    predator_hits: int
    total_spikes: int
    average_firing_rate_hz: float


class EpisodeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, stats: EpisodeStats) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(stats)))
            if header:
                writer.writeheader()
            writer.writerow(asdict(stats))
