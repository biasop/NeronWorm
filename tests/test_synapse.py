import pytest
from brain import Brain, Connectome
from brain.synapse import ChemicalSynapse, EventQueue, GapJunction


def test_event_not_delivered_before_delay_and_once_at_arrival():
    queue = EventQueue()
    queue.schedule(ChemicalSynapse("a", "b", 10, 5), 3)
    assert queue.pop_due(7.999) == []
    assert queue.pop_due(8) == [("b", 10)]
    assert queue.pop_due(100) == []


def test_heap_handles_out_of_order_and_simultaneous_events():
    queue = EventQueue()
    queue.schedule(ChemicalSynapse("a", "z", 1, 8), 0)
    queue.schedule(ChemicalSynapse("a", "c", 2, 2), 0)
    queue.schedule(ChemicalSynapse("a", "b", -3, 2), 0)
    assert queue.pop_due(2) == [("c", 2), ("b", -3)]
    assert len(queue) == 1


def test_negative_weight_inhibits_target_after_delay():
    net = Connectome()
    net.add_neuron("a")
    net.add_neuron("b")
    net.add_chemical_synapse("a", "b", -100, 2)
    brain = Brain(net)
    brain.stimulate("a", 400)
    brain.step(1)  # source fires at 1, event arrives at 3
    brain.step(1)
    brain.step(1)
    assert brain.neurons["b"].membrane_potential == -70
    brain.step(1)  # interval begins at 3: event integrates into target
    assert brain.neurons["b"].membrane_potential < -70


def test_gap_currents_are_equal_opposite_and_reduce_voltage_difference():
    gap = GapJunction("a", "b", .5)
    assert gap.currents(-50, -70) == (-10, 10)
    net = Connectome()
    net.add_neuron("a")
    net.add_neuron("b")
    net.add_gap_junction("a", "b", .5)
    brain = Brain(net)
    brain.neurons["a"].membrane_potential = -60
    brain.step(1)
    assert brain.neurons["a"].membrane_potential < -60
    assert brain.neurons["b"].membrane_potential > -70


@pytest.mark.parametrize("delay", [-1, float("inf")])
def test_invalid_chemical_delay(delay):
    with pytest.raises(ValueError):
        ChemicalSynapse("a", "b", 1, delay)


def test_invalid_gap_conductance():
    with pytest.raises(ValueError):
        GapJunction("a", "b", -1)
