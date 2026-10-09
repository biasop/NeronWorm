"""Segmented kinematic worm; neural motor activity is its only locomotor drive."""
from dataclasses import dataclass, field
from math import cos, sin, hypot, pi
from typing import Mapping
import numpy as np

from config import Config
from .muscles import Muscles


@dataclass
class Worm:
    x: float
    y: float
    angle: float
    config: Config
    velocity: float = field(init=False)
    energy: float = field(init=False)
    health: float = field(init=False)
    body_length: float = field(init=False)
    muscles: Muscles = field(init=False)
    segments: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.velocity = 0.0
        self.energy = self.config.initial_energy
        self.health = self.config.initial_health
        self.body_length = self.config.body_length
        self.muscles = Muscles(self.config.muscle_time_constant_s)
        spacing = self.body_length / (self.config.body_segments - 1)
        self.segments = np.array([
            (self.x - i * spacing * cos(self.angle), self.y - i * spacing * sin(self.angle))
            for i in range(self.config.body_segments)
        ], dtype=float)

    def step(self, motor_activity: Mapping[str, float], dt_s: float) -> None:
        self.muscles.step(motor_activity, dt_s)
        self.angle = (self.angle + self.config.turn_rate * self.muscles.turn_signal * dt_s + pi) % (2 * pi) - pi
        self.velocity = self.config.max_speed * self.muscles.forward_signal / 2.0
        radius = self.config.body_radius
        self.x = min(self.config.world_width - radius, max(radius, self.x + cos(self.angle) * self.velocity * dt_s))
        self.y = min(self.config.world_height - radius, max(radius, self.y + sin(self.angle) * self.velocity * dt_s))
        self.segments[0] = (self.x, self.y)
        spacing = self.body_length / (len(self.segments) - 1)
        for i in range(1, len(self.segments)):
            delta = self.segments[i] - self.segments[i - 1]
            distance = hypot(float(delta[0]), float(delta[1]))
            if distance > 1e-12:
                self.segments[i] = self.segments[i - 1] + delta * spacing / distance
            self.segments[i, 0] = np.clip(self.segments[i, 0], radius, self.config.world_width - radius)
            self.segments[i, 1] = np.clip(self.segments[i, 1], radius, self.config.world_height - radius)
        drain = self.config.energy_drain_per_s + self.config.movement_energy_per_s * self.velocity / self.config.max_speed
        self.energy = max(0.0, self.energy - drain * dt_s)

    def draw(self, surface: object) -> None:
        import pygame
        for i in reversed(range(len(self.segments))):
            point = tuple(int(v) for v in self.segments[i])
            radius = max(3, int(self.config.body_radius * (1.0 - 0.45 * i / len(self.segments))))
            pygame.draw.circle(surface, (47, 156, 146), point, radius + 2)
            pygame.draw.circle(surface, (113, 231, 191) if i == 0 else (75, 201, 172), point, radius)
        for side in (-1, 1):
            eye = (int(self.x + 5 * cos(self.angle) - side * 3 * sin(self.angle)),
                   int(self.y + 5 * sin(self.angle) + side * 3 * cos(self.angle)))
            pygame.draw.circle(surface, (12, 32, 38), eye, 2)
