from dataclasses import dataclass
from math import hypot


@dataclass
class Predator:
    """A simple pursuer with capped speed, independent of the neural engine."""
    x: float
    y: float
    speed: float = 29.0
    radius: float = 15.0

    def step(self, target_x: float, target_y: float, dt_s: float) -> None:
        dx, dy = target_x - self.x, target_y - self.y
        distance = hypot(dx, dy)
        if distance:
            fraction = min(1.0, self.speed * dt_s / distance)
            self.x += dx * fraction
            self.y += dy * fraction
