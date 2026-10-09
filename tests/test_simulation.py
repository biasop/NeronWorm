from dataclasses import replace
import csv
import numpy as np
import pytest
from environment.food import Food
from environment.predator import Predator
from organism.sensors import Sensors
from simulation import Simulation


def test_render_batching_does_not_change_seeded_state(config):
    a, b = Simulation(config, False), Simulation(config, False)
    a.advance(1)
    for _ in range(60):
        b.advance(1 / 60)
    assert a.total_time_s == pytest.approx(b.total_time_s)
    assert (a.worm.x, a.worm.y, a.worm.angle) == (b.worm.x, b.worm.y, b.worm.angle)
    assert a.brain.spike_counts == b.brain.spike_counts
    np.testing.assert_array_equal(a.brain.activity, b.brain.activity)
    assert [(f.x, f.y) for f in a.world.foods] == [(f.x, f.y) for f in b.world.foods]


def test_eating_restores_energy_respawns_food(config):
    sim = Simulation(config, False)
    sim.world.predator = None
    sim.worm.energy = 50
    food = Food(sim.worm.x, sim.worm.y)
    sim.world.foods = [food]
    sim.tick()
    assert sim.food_eaten == 1
    assert sim.worm.energy > 70
    assert sim.world.foods[0] is not food


def test_predator_hits_have_cooldown(config):
    sim = Simulation(config, False)
    sim.world.predator = Predator(sim.worm.x, sim.worm.y, 0)
    _, hits = sim.world.step(sim.worm, .01, 0)
    assert hits == 1
    assert sim.worm.health == 80
    assert sim.world.step(sim.worm, .01, .5)[1] == 0
    assert sim.world.step(sim.worm, .01, 1)[1] == 1


def test_starvation_records_episode_and_resets_neural_state(config):
    sim = Simulation(config)
    sim.world.foods = []
    sim.worm.energy = .0001
    sim.tick()
    assert sim.episode == 2
    assert sim.worm.energy == config.initial_energy
    assert sim.brain.time_ms == 0
    assert sim.completed_episodes[0].reason == "starvation"
    with config.results_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["survival_duration_s"]) == .01


def test_session_close_records_once_and_preserves_csv_header(config):
    sim = Simulation(config)
    sim.advance(.1)
    sim.reset_episode()
    sim.advance(.1)
    sim.close()
    sim.close()
    with config.results_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [row["reason"] for row in rows] == ["manual_reset", "session_end"]
    for row in rows:
        expected = int(row["total_spikes"]) / (.1 * len(sim.brain.neurons))
        assert float(row["average_firing_rate_hz"]) == pytest.approx(expected)


def test_concentration_is_local_distance_based_and_bounded(config):
    sources = [Food(0, 0)]
    assert Sensors.concentration((0, 0), sources, .01) == 1
    assert 0 < Sensors.concentration((100, 0), sources, .01) < 1
    assert Sensors.concentration((0, 0), sources * 10, .01) == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_elapsed_time(config, value):
    with pytest.raises(ValueError):
        Simulation(config, False).advance(value)


def test_nonintegral_neural_body_time_ratio_rejected(config):
    with pytest.raises(ValueError):
        replace(config, brain_dt_ms=.7)
