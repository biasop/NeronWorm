"""Rebuild the hand-designed 24-neuron fixture; no external dataset needed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.connectome import Connectome
from config import ROOT


def build_demo() -> Connectome:
    network = Connectome(metadata={
        "name": "NeuroWorm bilateral reflex demo",
        "description": "Artificial wiring, not an anatomical C. elegans dataset.",
        "sensor_mapping": {key: key for key in (
            "food_left", "food_right", "danger_left", "danger_right",
            "touch_left", "touch_right", "drive_left", "drive_right")},
        "motor_mapping": {key: key for key in ("motor_left", "motor_right")},
    })
    for side, letter in (("left", "L"), ("right", "R")):
        for sense in ("food", "danger", "touch", "drive"):
            network.add_neuron(f"{sense}_{side}", f"{sense.upper()} {letter}", "sensory")
        for pathway in ("food_1", "food_2", "avoid_1", "avoid_2", "forward", "touch_inter"):
            network.add_neuron(f"{pathway}_{side}", f"{pathway.upper()} {letter}", "interneuron")
        network.add_neuron(f"motor_{side}", f"MOTOR {letter}", "motor")
        network.add_neuron(f"tail_{side}", f"TAIL {letter}", "motor")

    for side, opposite in (("left", "right"), ("right", "left")):
        for relay in (1, 2):
            food = f"food_{relay}_{side}"
            avoid = f"avoid_{relay}_{side}"
            network.add_chemical_synapse(f"food_{side}", food, 340, 2)
            network.add_chemical_synapse(food, f"motor_{side}", 180, 1)
            network.add_chemical_synapse(food, f"motor_{opposite}", -45, 1)
            network.add_chemical_synapse(f"danger_{side}", avoid, 340, 1)
            network.add_chemical_synapse(avoid, f"motor_{opposite}", 270, 1)
            network.add_chemical_synapse(avoid, f"motor_{side}", -210, 1)
            network.add_chemical_synapse(avoid, food, -160, 1)
        network.add_chemical_synapse(f"drive_{side}", f"forward_{side}", 340, 1)
        network.add_chemical_synapse(f"forward_{side}", f"motor_{side}", 340, 1)
        network.add_chemical_synapse(f"touch_{side}", f"touch_inter_{side}", 340, 1)
        # A handed reflex breaks symmetry at a head-on wall encounter.
        network.add_chemical_synapse(f"touch_inter_{side}", f"motor_{opposite}", 540 if side == "left" else 360, 1)
        network.add_chemical_synapse(f"touch_inter_{side}", f"motor_{side}", -300 if side == "left" else -160, 1)
        network.add_chemical_synapse(f"motor_{side}", f"tail_{side}", 330, 2)
        network.add_gap_junction(f"motor_{side}", f"tail_{side}", 0.015)
    network.add_gap_junction("forward_left", "forward_right", 0.02)
    network.add_gap_junction("food_1_left", "food_2_left", 0.01)
    network.add_gap_junction("food_1_right", "food_2_right", 0.01)
    return network


if __name__ == "__main__":
    connectome = build_demo()
    path = ROOT / "data" / "demo_connectome.json"
    connectome.save_to_json(path)
    print(f"Saved {len(connectome.neurons)} neurons to {path}")
