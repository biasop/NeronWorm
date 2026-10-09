"""Measure isolated neural cue responses and actual field-driven body turns."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain import Brain, Connectome
from config import Config
from environment.food import Food
from environment.predator import Predator
from organism.sensors import Sensors, SensorReading
from simulation import Simulation


def verify() -> list[dict[str, object]]:
    config = Config()
    network = Connectome.load_from_json(config.connectome_path)
    results = []
    cues = [("safe", SensorReading(0, 0, 0, 0), 0),
            ("food_left", SensorReading(.8, .4, 0, 0), -1),
            ("food_right", SensorReading(.4, .8, 0, 0), 1),
            ("danger_left", SensorReading(0, 0, .8, .4), 1),
            ("danger_right", SensorReading(0, 0, .4, .8), -1)]
    for name, reading, sign in cues:
        brain = Brain(network, config.motor_time_constant_ms, config.motor_reference_rate_hz)
        currents = Sensors(config).currents(reading)
        for _ in range(1200):
            for nid, current in currents.items():
                brain.stimulate(nid, current)
            brain.step(1)
        outputs = brain.get_motor_activity()
        left, right = outputs["motor_left"], outputs["motor_right"]
        delta = right - left
        passed = sign * delta > .05 if sign else abs(delta) < 1e-6 and left > .1
        result = {"scenario": name, "left_motor": round(left, 4), "right_motor": round(right, 4),
                  "turn_signal": round(delta, 4), "spikes": brain.total_spikes, "passed": passed}
        if sign:
            sim = Simulation(config, log_episodes=False)
            sim.worm.x, sim.worm.y = 400, 350
            sim.world.foods = []
            sim.world.predator = None
            side = -1 if name.endswith("left") else 1
            if name.startswith("food"):
                sim.world.foods = [Food(450, 350 + side * 45)]
            else:
                sim.world.predator = Predator(450, 350 + side * 45, speed=0)
            sim.advance(.8)
            result["body_angle_radians"] = round(sim.worm.angle, 4)
            result["passed"] = passed and sign * sim.worm.angle > .02
        results.append(result)
    if not all(row["passed"] for row in results):
        raise RuntimeError(f"Behavior verification failed: {results}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(verify(), indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
