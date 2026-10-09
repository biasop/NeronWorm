"""Central configuration. Neural time is ms; body time is seconds."""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    seed: int = 7
    world_width: int = 900
    world_height: int = 700
    brain_panel_width: int = 460
    render_fps: int = 60
    brain_dt_ms: float = 1.0
    physics_dt_s: float = 0.01
    food_count: int = 12
    food_radius: float = 7.0
    food_energy: float = 24.0
    food_alpha: float = 0.02
    danger_alpha: float = 0.018
    food_current: float = 65.0
    danger_current: float = 105.0
    touch_current: float = 100.0
    tonic_current: float = 24.0
    sensor_forward_offset: float = 12.0
    sensor_lateral_offset: float = 13.0
    motor_time_constant_ms: float = 100.0
    motor_reference_rate_hz: float = 180.0
    muscle_time_constant_s: float = 0.08
    max_speed: float = 90.0
    turn_rate: float = 7.0
    body_length: float = 52.0
    body_segments: int = 12
    body_radius: float = 8.0
    initial_energy: float = 100.0
    initial_health: float = 100.0
    energy_drain_per_s: float = 0.8
    movement_energy_per_s: float = 0.7
    predator_speed: float = 18.0
    predator_radius: float = 15.0
    predator_damage: float = 20.0
    predator_hit_cooldown_s: float = 1.0
    connectome_path: Path = ROOT / "data" / "demo_connectome.json"
    results_path: Path = ROOT / "results" / "episodes.csv"

    def __post_init__(self) -> None:
        for name in ("world_width", "world_height", "brain_panel_width", "render_fps",
                     "brain_dt_ms", "physics_dt_s", "food_alpha", "danger_alpha",
                     "motor_time_constant_ms", "motor_reference_rate_hz",
                     "muscle_time_constant_s", "max_speed", "body_length", "body_radius",
                     "initial_energy", "initial_health", "predator_hit_cooldown_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.brain_dt_ms > 1.0:
            raise ValueError("brain_dt_ms must be at most 1 ms for Euler stability")
        ratio = self.physics_dt_s * 1000 / self.brain_dt_ms
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("physics_dt_s must contain a whole number of brain steps")
        if self.food_count < 0 or self.body_segments < 2:
            raise ValueError("food_count must be nonnegative and body_segments >= 2")
        if min(self.world_width, self.world_height) < 150:
            raise ValueError("world dimensions must be at least 150 pixels")
