"""Fixed-step orchestration, usable with or without a display."""
from dataclasses import asdict
import logging
from math import isfinite
import numpy as np

from brain import Brain, Connectome
from config import Config
from environment.world import World
from organism.sensors import Sensors
from organism.worm import Worm
from episode_stats import EpisodeLogger, EpisodeStats

log = logging.getLogger(__name__)


class Simulation:
    """10 ms physics ticks contain ten 1 ms neural ticks by default."""
    def __init__(self, config: Config, log_episodes: bool = True) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.connectome = Connectome.load_from_json(config.connectome_path)
        # Body mappings are explicit metadata, to support arbitrary future IDs.
        self.sensor_mapping = self.connectome.metadata.get("sensor_mapping", {})
        self.motor_mapping = self.connectome.metadata.get("motor_mapping", {})
        for key in ("food_left", "food_right", "danger_left", "danger_right",
                    "touch_left", "touch_right", "drive_left", "drive_right"):
            nid = self.sensor_mapping.get(key, key)
            if nid not in self.connectome.neurons or self.connectome.neurons[nid]["role"] != "sensory":
                raise ValueError(f"Missing sensory mapping for {key}")
        for key in ("motor_left", "motor_right"):
            nid = self.motor_mapping.get(key, key)
            if nid not in self.connectome.neurons or self.connectome.neurons[nid]["role"] != "motor":
                raise ValueError(f"Missing motor mapping for {key}")
        self.brain = Brain(self.connectome, config.motor_time_constant_ms, config.motor_reference_rate_hz)
        self.sensors = Sensors(config)
        self.episode_logger = EpisodeLogger(config.results_path) if log_episodes else None
        self.episode = 1
        self.total_time_s = 0.0
        self.completed_episodes: list[EpisodeStats] = []
        self._accumulator_s = 0.0
        self._closed = False
        self._start_episode()

    def _start_episode(self) -> None:
        self.brain.reset()
        self.worm = Worm(self.config.world_width * 0.52, self.config.world_height * 0.52, 0.0, self.config)
        self.world = World(self.config, self.rng)
        self.episode_time_s = 0.0
        self.food_eaten = 0
        self.predator_hits = 0
        self.last_reading = self.sensors.sample(self.worm.x, self.worm.y, self.worm.angle,
                                               self.world.foods, self.world.predator)

    def _record_episode(self, reason: str) -> None:
        if self.episode_time_s <= 0:
            return
        stats = EpisodeStats(
            self.config.seed, self.episode, reason, round(self.episode_time_s, 6),
            self.food_eaten, self.predator_hits, self.brain.total_spikes,
            self.brain.total_spikes / (len(self.brain.neurons) * self.episode_time_s),
        )
        self.completed_episodes.append(stats)
        if self.episode_logger is not None:
            self.episode_logger.append(stats)
        log.info("Episode result: %s", asdict(stats))

    def reset_episode(self, reason: str = "manual_reset") -> None:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        self._record_episode(reason)
        self.episode += 1
        self._start_episode()

    def motor_activity(self) -> dict[str, float]:
        output = self.brain.get_motor_activity()
        return {key: output[self.motor_mapping.get(key, key)] for key in ("motor_left", "motor_right")}

    def tick(self) -> None:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        c = self.config
        self.last_reading = self.sensors.sample(self.worm.x, self.worm.y, self.worm.angle,
                                               self.world.foods, self.world.predator)
        currents = self.sensors.currents(self.last_reading)
        for _ in range(round(c.physics_dt_s * 1000 / c.brain_dt_ms)):
            for channel, current in currents.items():
                self.brain.stimulate(self.sensor_mapping.get(channel, channel), current)
            self.brain.step(c.brain_dt_ms)
        self.worm.step(self.motor_activity(), c.physics_dt_s)
        self.episode_time_s += c.physics_dt_s
        self.total_time_s += c.physics_dt_s
        eaten, hits = self.world.step(self.worm, c.physics_dt_s, self.episode_time_s)
        self.food_eaten += eaten
        self.predator_hits += hits
        if self.worm.energy <= 0 or self.worm.health <= 0:
            self.reset_episode("starvation" if self.worm.energy <= 0 else "predator")

    def advance(self, elapsed_s: float) -> int:
        """Accumulate elapsed simulation time independently of render frequency."""
        if not isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        self._accumulator_s += elapsed_s
        ticks = 0
        while self._accumulator_s + 1e-12 >= self.config.physics_dt_s:
            self.tick()
            self._accumulator_s -= self.config.physics_dt_s
            ticks += 1
        self._accumulator_s = max(0.0, self._accumulator_s)
        return ticks

    def close(self) -> None:
        if not self._closed:
            self._record_episode("session_end")
            self._closed = True
