import numpy as np
import pytest
from brain import Brain, Connectome


def test_sensory_spike_propagates_via_interneuron_to_motor():
    net = Connectome()
    for nid, role in (("sense", "sensory"), ("relay", "interneuron"), ("motor", "motor")):
        net.add_neuron(nid, role=role)
    net.add_chemical_synapse("sense", "relay", 340, 2)
    net.add_chemical_synapse("relay", "motor", 340, 3)
    brain = Brain(net)
    brain.stimulate("sense", 400)
    for _ in range(9):
        brain.step(1)
    assert brain.neurons["sense"].last_spike_time == 1
    assert brain.neurons["relay"].last_spike_time == 4
    assert brain.neurons["motor"].last_spike_time == 8
    assert brain.get_motor_activity()["motor"] > 0
    assert brain.total_spikes == 3


def test_no_spontaneous_movement_drive_without_stimulation(demo):
    brain = Brain(demo)
    for _ in range(200):
        brain.step(1)
    assert brain.total_spikes == 0
    assert all(value == 0 for value in brain.get_motor_activity().values())


def test_reset_clears_pending_events_activity_and_clock(demo):
    brain = Brain(demo)
    brain.stimulate("food_left", 400)
    brain.step(1)
    assert len(brain.events) == 2
    brain.reset()
    assert len(brain.events) == 0
    assert brain.time_ms == 0
    assert brain.total_spikes == 0
    assert not np.any(brain.activity)


@pytest.mark.parametrize("dt", [0, 2, float("nan")])
def test_invalid_neural_dt_rejected(demo, dt):
    with pytest.raises(ValueError):
        Brain(demo).step(dt)


def test_smaller_timestep_preserves_event_impulse_area():
    net = Connectome()
    net.add_neuron("a")
    net.add_neuron("b")
    net.add_chemical_synapse("a", "b", 100, 0)
    results = []
    for dt in (1, .5):
        brain = Brain(net)
        brain.events.schedule(net.chemical_synapses[0], 0)
        brain.step(dt)
        results.append(brain.neurons["b"].membrane_potential)
    assert results == [-65, -65]
