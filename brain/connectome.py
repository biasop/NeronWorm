"""Anatomical structure and JSON persistence, with no simulation state."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .neuron import Neuron
from .synapse import ChemicalSynapse, GapJunction


@dataclass
class Connectome:
    """Metadata and connections use stable IDs, never live neuron references."""
    neurons: dict[str, dict[str, Any]] = field(default_factory=dict)
    chemical_synapses: list[ChemicalSynapse] = field(default_factory=list)
    gap_junctions: list[GapJunction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_neuron(self, neuron_id: str, name: str = "", role: str = "interneuron",
                   parameters: dict[str, float] | None = None, **metadata: Any) -> None:
        if neuron_id in self.neurons:
            raise ValueError(f"Duplicate neuron ID: {neuron_id}")
        parameters = dict(parameters or {})
        # Validation only: the temporary neuron is not stored or simulated.
        Neuron(neuron_id=neuron_id, name=name, role=role, **parameters)
        self.neurons[neuron_id] = {
            **metadata, "neuron_id": neuron_id, "name": name or neuron_id,
            "role": role, "parameters": parameters,
        }

    def _require_ids(self, *neuron_ids: str) -> None:
        for neuron_id in neuron_ids:
            if neuron_id not in self.neurons:
                raise ValueError(f"Unknown neuron ID: {neuron_id}")

    def add_chemical_synapse(self, source_id: str, target_id: str,
                             weight: float, delay_ms: float = 1.0) -> None:
        self._require_ids(source_id, target_id)
        self.chemical_synapses.append(ChemicalSynapse(source_id, target_id, weight, delay_ms))

    def add_gap_junction(self, neuron_a_id: str, neuron_b_id: str,
                         conductance: float) -> None:
        self._require_ids(neuron_a_id, neuron_b_id)
        if neuron_a_id == neuron_b_id:
            raise ValueError("A gap junction requires two different neurons")
        self.gap_junctions.append(GapJunction(neuron_a_id, neuron_b_id, conductance))

    @classmethod
    def load_from_json(cls, path: str | Path) -> "Connectome":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version", 1) != 1:
            raise ValueError("Unsupported connectome schema_version")
        connectome = cls(metadata=payload.get("metadata", {}))
        for neuron in payload["neurons"]:
            connectome.add_neuron(**neuron)
        for synapse in payload.get("chemical_synapses", []):
            connectome.add_chemical_synapse(**synapse)
        for junction in payload.get("gap_junctions", []):
            connectome.add_gap_junction(**junction)
        return connectome

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "metadata": self.metadata,
            "neurons": list(self.neurons.values()),
            "chemical_synapses": [asdict(s) for s in self.chemical_synapses],
            "gap_junctions": [asdict(g) for g in self.gap_junctions],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
