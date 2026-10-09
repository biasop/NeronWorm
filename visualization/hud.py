"""Compact survival and neural activity readouts."""
import pygame

from simulation import Simulation


class HUD:
    def __init__(self) -> None:
        self.title = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 18)

    def draw(self, surface: pygame.Surface, sim: Simulation, paused: bool, speed: float,
             debug: bool, fps: float) -> None:
        width, height = surface.get_size()
        overlay = pygame.Surface((width, 113), pygame.SRCALPHA)
        overlay.fill((12, 23, 34, 224))
        surface.blit(overlay, (0, 0))
        surface.blit(self.title.render("NEUROWORM", True, (131, 236, 207)), (20, 15))
        state = "PAUSED" if paused else f"{speed:g}x speed"
        surface.blit(self.font.render(f"{state}  ·  Episode {sim.episode}", True, (219, 230, 240)), (235, 19))
        rows = (
            f"Time {sim.total_time_s:6.1f}s   Survival {sim.episode_time_s:5.1f}s   Food {sim.food_eaten}   Hits {sim.predator_hits}",
            f"Energy {sim.worm.energy:5.1f}   Health {sim.worm.health:5.1f}   Spikes/s {sim.brain.spikes_per_second:6.0f}   "
            f"Motor L {sim.worm.muscles.left:.2f} / R {sim.worm.muscles.right:.2f}",
        )
        for y, row in zip((53, 81), rows):
            surface.blit(self.font.render(row, True, (195, 212, 225)), (20, y))
        footer = pygame.Surface((width, 49), pygame.SRCALPHA)
        footer.fill((12, 23, 34, 224))
        surface.blit(footer, (0, height - 49))
        text = "SPACE pause   R reset   B brain   D debug   UP/DOWN speed   ESC quit"
        surface.blit(self.small.render(text, True, (179, 202, 217)), (20, height - 31))
        if debug:
            r = sim.last_reading
            left, right = sim.sensors.positions(sim.worm.x, sim.worm.y, sim.worm.angle)
            for p in (left, right):
                pygame.draw.circle(surface, (246, 220, 143), (int(p[0]), int(p[1])), 4, 1)
            message = (f"Food L/R {r.food_left:.3f}/{r.food_right:.3f}   Danger L/R {r.danger_left:.3f}/{r.danger_right:.3f}   "
                       f"Queue {len(sim.brain.events)}   FPS {fps:.0f}")
            surface.blit(self.small.render(message, True, (249, 216, 143)), (20, 124))
