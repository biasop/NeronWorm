"""ID-based chemical connections, electrical coupling, and delayed events."""
from dataclasses import dataclass
import heapq
from itertools import count
from math import isfinite


@dataclass(frozen=True)
class ChemicalSynapse:
    """A spike transmits a signed current impulse after delay_ms."""
    source_id: str
    target_id: str
    weight: float
    delay_ms: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.weight) or not isfinite(self.delay_ms) or self.delay_ms < 0:
            raise ValueError("Synaptic weight must be finite and delay nonnegative")


@dataclass(frozen=True)
class GapJunction:
    """Symmetric electrical coupling, evaluated from pre-step voltages."""
    neuron_a_id: str
    neuron_b_id: str
    conductance: float

    def __post_init__(self) -> None:
        if not isfinite(self.conductance) or self.conductance < 0:
            raise ValueError("Gap conductance must be finite and nonnegative")

    def currents(self, voltage_a: float, voltage_b: float) -> tuple[float, float]:
        current = self.conductance * (voltage_a - voltage_b)
        return -current, current


class EventQueue:
    """One heap for chemical events; stable ordering for simultaneous arrivals."""
    def __init__(self) -> None:
        self._events: list[tuple[float, int, str, float]] = []
        self._sequence = count()

    def schedule(self, synapse: ChemicalSynapse, spike_time: float) -> None:
        heapq.heappush(self._events, (
            spike_time + synapse.delay_ms, next(self._sequence),
            synapse.target_id, synapse.weight,
        ))

    def pop_due(self, time_ms: float) -> list[tuple[str, float]]:
        events = []
        while self._events and self._events[0][0] <= time_ms + 1e-9:
            _, _, target_id, weight = heapq.heappop(self._events)
            events.append((target_id, weight))
        return events

    def clear(self) -> None:
        self._events.clear()
        self._sequence = count()

    def __len__(self) -> int:
        return len(self._events)
