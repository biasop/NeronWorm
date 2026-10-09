"""A single-clock spiking neural engine, independent of Pygame and the world."""
from collections import deque
from math import exp, isfinite
import numpy as np

from .connectome import Connectome
from .neuron import Neuron
from .synapse import EventQueue


class Brain:
    """Own neural state; structure is provided by a replaceable Connectome.

    Activity is an exponential spike-rate estimate normalized by a reference
    rate. Zero-delay events arrive at the next central step boundary.
    """
    def __init__(self, connectome: Connectome, activity_tau_ms: float = 100.0,
                 reference_rate_hz: float = 80.0) -> None:
        if not all(isfinite(v) and v > 0 for v in (activity_tau_ms, reference_rate_hz)):
            raise ValueError("Activity tau and reference firing rate must be positive")
        self.connectome = connectome
        self.neurons = {
            nid: Neuron(neuron_id=nid, name=meta["name"], role=meta["role"],
                        **meta.get("parameters", {}))
            for nid, meta in connectome.neurons.items()
        }
        if not self.neurons:
            raise ValueError("Brain needs at least one neuron")
        self.neuron_ids = tuple(self.neurons)
        self._indices = {nid: i for i, nid in enumerate(self.neuron_ids)}
        self._outgoing = {nid: [] for nid in self.neurons}
        for synapse in connectome.chemical_synapses:
            connectome._require_ids(synapse.source_id, synapse.target_id)
            self._outgoing[synapse.source_id].append(synapse)
        for gap in connectome.gap_junctions:
            connectome._require_ids(gap.neuron_a_id, gap.neuron_b_id)
        self.events = EventQueue()
        self.activity_tau_ms = activity_tau_ms
        self.reference_rate_hz = reference_rate_hz
        self.activity = np.zeros(len(self.neurons), dtype=float)
        self.spike_counts = {nid: 0 for nid in self.neurons}
        self._recent_spikes: deque[tuple[float, int]] = deque()
        self.time_ms = 0.0
        self.total_spikes = 0

    def stimulate(self, neuron_id: str, current: float) -> None:
        """Queue current for one step. Reapply sustained stimulation each step."""
        self.neurons[neuron_id].receive(current)

    def step(self, dt_ms: float) -> None:
        max_dt = min(1.0, *(n.membrane_time_constant for n in self.neurons.values()))
        if not isfinite(dt_ms) or not 0 < dt_ms <= max_dt:
            raise ValueError("Brain steps must be positive and at most 1 ms (and tau)")
        for target_id, weight in self.events.pop_due(self.time_ms):
            # Weight is mV-equivalent current for a 1 ms impulse. Scaling for
            # smaller dt preserves each event's integrated current area.
            self.neurons[target_id].receive(weight / dt_ms)
        gap_currents = np.zeros(len(self.neurons), dtype=float)
        for gap in self.connectome.gap_junctions:
            ia, ib = gap.currents(self.neurons[gap.neuron_a_id].membrane_potential,
                                 self.neurons[gap.neuron_b_id].membrane_potential)
            gap_currents[self._indices[gap.neuron_a_id]] += ia
            gap_currents[self._indices[gap.neuron_b_id]] += ib
        for nid, current in zip(self.neuron_ids, gap_currents):
            self.neurons[nid].receive(float(current))
        self.activity *= exp(-dt_ms / self.activity_tau_ms)
        spike_increment = 1000.0 / (self.activity_tau_ms * self.reference_rate_hz)
        spiking = []
        for nid, neuron in self.neurons.items():
            if neuron.step(dt_ms, self.time_ms):
                spiking.append(nid)
                self.spike_counts[nid] += 1
                self.activity[self._indices[nid]] += spike_increment
        self.time_ms += dt_ms
        for nid in spiking:
            for synapse in self._outgoing[nid]:
                self.events.schedule(synapse, self.time_ms)
        self.total_spikes += len(spiking)
        self._recent_spikes.append((self.time_ms, len(spiking)))
        while self._recent_spikes and self._recent_spikes[0][0] <= self.time_ms - 1000:
            self._recent_spikes.popleft()

    def get_activity(self, neuron_id: str) -> float:
        return float(self.activity[self._indices[neuron_id]])

    def get_motor_activity(self) -> dict[str, float]:
        return {nid: self.get_activity(nid) for nid, n in self.neurons.items() if n.role == "motor"}

    @property
    def spikes_per_second(self) -> float:
        duration = min(self.time_ms, 1000.0) / 1000.0
        return sum(count for _, count in self._recent_spikes) / duration if duration else 0.0

    def reset(self) -> None:
        for neuron in self.neurons.values():
            neuron.reset()
        self.events.clear()
        self.activity.fill(0)
        self.spike_counts = {nid: 0 for nid in self.neurons}
        self._recent_spikes.clear()
        self.time_ms = 0.0
        self.total_spikes = 0
