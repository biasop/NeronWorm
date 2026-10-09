import json
import pytest
from brain import Connectome


def test_demo_json_counts_and_sensor_motor_roles(demo):
    assert len(demo.neurons) == 24
    assert len(demo.chemical_synapses) == 40
    assert len(demo.gap_junctions) == 5
    for nid in ("food_left", "food_right", "danger_left", "danger_right"):
        assert demo.neurons[nid]["role"] == "sensory"
    assert demo.neurons["motor_left"]["role"] == "motor"


def test_json_round_trip_preserves_structure_and_extra_metadata(demo, tmp_path):
    demo.neurons["food_left"]["neurotransmitter"] = "unknown"
    path = tmp_path / "nested" / "network.json"
    demo.save_to_json(path)
    restored = Connectome.load_from_json(path)
    assert restored == demo


def test_duplicate_and_dangling_connections_rejected():
    net = Connectome()
    net.add_neuron("a")
    with pytest.raises(ValueError, match="Duplicate"):
        net.add_neuron("a")
    with pytest.raises(ValueError, match="Unknown"):
        net.add_chemical_synapse("a", "missing", 1)
    with pytest.raises(ValueError, match="Unknown"):
        net.add_gap_junction("a", "missing", .1)


def test_unknown_json_schema_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "neurons": []}))
    with pytest.raises(ValueError, match="schema"):
        Connectome.load_from_json(path)
