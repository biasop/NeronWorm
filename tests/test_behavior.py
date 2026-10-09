from math import hypot
import pytest
from brain import Brain
from environment.food import Food
from environment.predator import Predator
from organism.sensors import Sensors, SensorReading
from simulation import Simulation


@pytest.mark.parametrize("cue,left,right,turn_sign", [
    ("food", .8, .4, -1), ("food", .4, .8, 1),
    ("danger", .8, .4, 1), ("danger", .4, .8, -1),
])
def test_sensor_drive_produces_correct_neural_motor_asymmetry(config, demo, cue, left, right, turn_sign):
    brain = Brain(demo, config.motor_time_constant_ms, config.motor_reference_rate_hz)
    r = SensorReading(left, right, 0, 0) if cue == "food" else SensorReading(0, 0, left, right)
    currents = Sensors(config).currents(r)
    for _ in range(1000):
        for nid, current in currents.items():
            brain.stimulate(nid, current)
        brain.step(1)
    outputs = brain.get_motor_activity()
    assert turn_sign * (outputs["motor_right"] - outputs["motor_left"]) > .05


@pytest.mark.parametrize("cue,side", [("food", -1), ("food", 1), ("danger", -1), ("danger", 1)])
def test_local_fields_make_body_turn_toward_food_and_away_from_danger(config, cue, side):
    sim = Simulation(config, log_episodes=False)
    sim.worm.x, sim.worm.y = 400, 350
    sim.worm.angle = 0
    sim.world.foods = []
    sim.world.predator = None
    if cue == "food":
        sim.world.foods = [Food(450, 350 + side * 45)]
    else:
        sim.world.predator = Predator(450, 350 + side * 45, speed=0)
    sim.advance(.8)
    expected_sign = side if cue == "food" else -side
    assert expected_sign * sim.worm.angle > .02
    assert sim.brain.total_spikes > 0
    assert hypot(sim.worm.x - 400, sim.worm.y - 350) > 1


def test_safe_tonic_drive_produces_symmetric_forward_motion(config):
    sim = Simulation(config, log_episodes=False)
    sim.world.foods = []
    sim.world.predator = None
    x = sim.worm.x
    sim.advance(1)
    assert sim.worm.x > x + 10
    assert sim.worm.angle == pytest.approx(0)
    assert sim.worm.muscles.left == pytest.approx(sim.worm.muscles.right)


def test_head_on_wall_reflex_breaks_symmetry_and_escapes(config):
    sim = Simulation(config, log_episodes=False)
    sim.world.foods = []
    sim.world.predator = None
    sim.worm.x = config.world_width - config.body_radius
    sim.worm.y = config.world_height / 2
    sim.worm.segments[:, 0] = sim.worm.x
    sim.worm.segments[:, 1] = sim.worm.y
    sim.advance(3)
    assert abs(sim.worm.angle) > .2
    assert sim.worm.x < config.world_width - config.body_radius - 2


def test_zero_motor_activity_cannot_move_body(config):
    sim = Simulation(config, log_episodes=False)
    before = (sim.worm.x, sim.worm.y, sim.worm.angle)
    for _ in range(100):
        sim.worm.step({"motor_left": 0, "motor_right": 0}, .01)
    assert (sim.worm.x, sim.worm.y, sim.worm.angle) == before


def test_seeded_demo_collects_food_in_survival_environment(config):
    sim = Simulation(config, log_episodes=False)
    sim.advance(30)
    assert sim.food_eaten >= 1
    assert sim.worm.health > 0
