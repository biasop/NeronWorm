"""Seeded environment; food placement and collisions do not control locomotion."""
from math import hypot
import numpy as np

from config import Config
from organism.worm import Worm
from .food import Food
from .predator import Predator


class World:
    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.foods = [self.make_food() for _ in range(config.food_count)]
        self.predator: Predator | None = Predator(
            config.world_width * 0.12, config.world_height * 0.16,
            config.predator_speed, config.predator_radius,
        )
        self._last_hit_s = float("-inf")

    def make_food(self) -> Food:
        c = self.config
        return Food(float(self.rng.uniform(25, c.world_width - 25)),
                    float(self.rng.uniform(25, c.world_height - 25)), c.food_radius, c.food_energy)

    def step(self, worm: Worm, dt_s: float, time_s: float) -> tuple[int, int]:
        eaten = 0
        for index, food in enumerate(self.foods):
            if hypot(worm.x - food.x, worm.y - food.y) <= self.config.body_radius + food.radius:
                worm.energy = min(self.config.initial_energy, worm.energy + food.energy)
                self.foods[index] = self.make_food()
                eaten += 1
        hits = 0
        if self.predator is not None:
            self.predator.step(worm.x, worm.y, dt_s)
            distance = min(hypot(float(x) - self.predator.x, float(y) - self.predator.y)
                           for x, y in worm.segments)
            if (distance <= self.predator.radius + self.config.body_radius and
                    time_s - self._last_hit_s >= self.config.predator_hit_cooldown_s):
                worm.health = max(0.0, worm.health - self.config.predator_damage)
                self._last_hit_s = time_s
                hits = 1
        return eaten, hits

    def draw(self, surface: object, debug: bool = False) -> None:
        import pygame
        surface.fill((16, 28, 38))
        for x in range(0, self.config.world_width, 50):
            pygame.draw.line(surface, (22, 38, 48), (x, 0), (x, self.config.world_height))
        for y in range(0, self.config.world_height, 50):
            pygame.draw.line(surface, (22, 38, 48), (0, y), (self.config.world_width, y))
        for food in self.foods:
            p = (round(food.x), round(food.y))
            if debug:
                pygame.draw.circle(surface, (36, 62, 48), p, 90, 1)
            pygame.draw.circle(surface, (63, 86, 41), p, round(food.radius) + 4)
            pygame.draw.circle(surface, (185, 226, 102), p, round(food.radius))
        if self.predator is not None:
            p = (round(self.predator.x), round(self.predator.y))
            if debug:
                pygame.draw.circle(surface, (80, 38, 48), p, 130, 1)
            pygame.draw.circle(surface, (230, 89, 107), p, round(self.predator.radius))
            pygame.draw.circle(surface, (255, 151, 127), p, 7, 2)
        pygame.draw.rect(surface, (74, 100, 112), surface.get_rect(), 2)
