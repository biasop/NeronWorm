"""Local concentration sampling; no object coordinates enter the brain."""
from dataclasses import dataclass
from math import cos, sin, exp, hypot
from typing import Iterable, Protocol

from config import Config


class Located(Protocol):
    x: float
    y: float


@dataclass(frozen=True)
class SensorReading:
    food_left: float
    food_right: float
    danger_left: float
    danger_right: float
    touch_left: float = 0.0
    touch_right: float = 0.0


class Sensors:
    """Bilateral head probes sample exp(-alpha*d) fields in screen coordinates."""
    def __init__(self, config: Config) -> None:
        self.config = config

    def positions(self, x: float, y: float, angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
        c, s = cos(angle), sin(angle)
        head = (x + c * self.config.sensor_forward_offset,
                y + s * self.config.sensor_forward_offset)
        lateral = self.config.sensor_lateral_offset
        # Negative y is left when facing right; positive angles turn clockwise.
        return ((head[0] + s * lateral, head[1] - c * lateral),
                (head[0] - s * lateral, head[1] + c * lateral))

    @staticmethod
    def concentration(position: tuple[float, float], sources: Iterable[Located], alpha: float) -> float:
        # Bounded sum: multiple food sources can contribute without unbounded drive.
        total = sum(exp(-alpha * hypot(source.x - position[0], source.y - position[1]))
                    for source in sources)
        return min(1.0, total)

    def sample(self, x: float, y: float, angle: float, foods: Iterable[Located],
               predator: Located | None) -> SensorReading:
        left, right = self.positions(x, y, angle)
        foods = tuple(foods)
        dangers = () if predator is None else (predator,)
        def touching(p: tuple[float, float]) -> float:
            return float(not (0 < p[0] < self.config.world_width and
                              0 < p[1] < self.config.world_height))
        return SensorReading(
            self.concentration(left, foods, self.config.food_alpha),
            self.concentration(right, foods, self.config.food_alpha),
            self.concentration(left, dangers, self.config.danger_alpha),
            self.concentration(right, dangers, self.config.danger_alpha),
            touching(left), touching(right),
        )

    def currents(self, reading: SensorReading) -> dict[str, float]:
        c = self.config
        return {
            "food_left": reading.food_left * c.food_current,
            "food_right": reading.food_right * c.food_current,
            "danger_left": reading.danger_left * c.danger_current,
            "danger_right": reading.danger_right * c.danger_current,
            "touch_left": reading.touch_left * c.touch_current,
            "touch_right": reading.touch_right * c.touch_current,
            "drive_left": c.tonic_current,
            "drive_right": c.tonic_current,
        }
