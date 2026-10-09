import pytest
from brain import Neuron


def test_positive_current_depolarizes_and_is_cleared():
    neuron = Neuron("n")
    neuron.receive(40)
    assert not neuron.step(1, 0)
    assert neuron.membrane_potential == pytest.approx(-68)
    assert neuron.input_current == 0


def test_spike_resets_and_has_end_of_step_timestamp():
    neuron = Neuron("n")
    neuron.receive(400)
    assert neuron.step(1, 0)
    assert neuron.spiked
    assert neuron.membrane_potential == neuron.reset_potential
    assert neuron.last_spike_time == 1


def test_refractory_period_discards_input_then_recovers():
    neuron = Neuron("n")
    neuron.fire(5)
    for time in (5, 6):
        neuron.receive(1000)
        assert not neuron.step(1, time)
        assert neuron.input_current == 0
        assert neuron.membrane_potential == -70
    neuron.receive(1000)
    assert neuron.step(1, 7)


def test_unstimulated_voltage_leaks_toward_rest():
    neuron = Neuron("n")
    neuron.membrane_potential = -60
    neuron.step(1, 0)
    assert -70 < neuron.membrane_potential < -60


def test_currents_sum_and_negative_input_inhibits():
    neuron = Neuron("n")
    neuron.receive(20)
    neuron.receive(-60)
    neuron.step(1, 0)
    assert neuron.membrane_potential < neuron.resting_potential


@pytest.mark.parametrize("dt", [0, -1, 21, float("nan")])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError):
        Neuron("n").step(dt, 0)


def test_reset_clears_dynamic_state():
    neuron = Neuron("n")
    neuron.fire(20)
    neuron.receive(100)
    neuron.reset()
    assert not neuron.spiked
    assert neuron.input_current == 0
    assert not neuron.is_refractory(0)
