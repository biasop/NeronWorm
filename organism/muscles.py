"""Differential motor mapping with no access to environmental or sensory data."""
from dataclasses import dataclass
from math import exp
from typing import Mapping


@dataclass
class Muscles:
    """Low-pass filtered bilateral motor activation."""
    time_constant_s: float = 0.08
    left: float = 0.0
    right: float = 0.0

    def step(self, motor_activity: Mapping[str, float], dt_s: float) -> None:
        blend = 1.0 - exp(-dt_s / self.time_constant_s)
        self.left += blend * (min(1.0, max(0.0, motor_activity["motor_left"])) - self.left)
        self.right += blend * (min(1.0, max(0.0, motor_activity["motor_right"])) - self.right)

    @property
    def turn_signal(self) -> float:
        return self.right - self.left

    @property
    def forward_signal(self) -> float:
        return self.left + self.right
