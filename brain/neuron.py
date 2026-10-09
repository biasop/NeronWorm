"""A small, explicit Leaky Integrate-and-Fire model."""
from dataclasses import dataclass, field
from math import isfinite


@dataclass
class Neuron:
    """Euler LIF neuron with mV potentials and ms time units.

    Current is mV-equivalent drive (membrane resistance is implicit).
    Inputs accumulate for one step, then clear even during refractoriness.
    """
    neuron_id: str
    name: str = ""
    role: str = "interneuron"
    resting_potential: float = -70.0
    threshold: float = -55.0
    reset_potential: float = -70.0
    membrane_time_constant: float = 20.0
    refractory_period: float = 2.0
    membrane_potential: float = field(init=False)
    last_spike_time: float = field(default=float("-inf"), init=False)
    input_current: float = field(default=0.0, init=False)
    spiked: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.neuron_id:
            raise ValueError("neuron_id cannot be empty")
        if self.role not in {"sensory", "interneuron", "motor"}:
            raise ValueError(f"Unknown neuron role: {self.role}")
        values = (self.resting_potential, self.threshold, self.reset_potential,
                  self.membrane_time_constant, self.refractory_period)
        if not all(isfinite(v) for v in values):
            raise ValueError("Neuron parameters must be finite")
        if self.membrane_time_constant <= 0 or self.refractory_period < 0:
            raise ValueError("Invalid time constant or refractory period")
        if max(self.resting_potential, self.reset_potential) >= self.threshold:
            raise ValueError("Rest and reset potentials must be below threshold")
        self.name = self.name or self.neuron_id
        self.membrane_potential = self.resting_potential

    def receive(self, current: float) -> None:
        if not isfinite(current):
            raise ValueError("Input current must be finite")
        self.input_current += current

    def integrate(self, dt: float) -> None:
        if not isfinite(dt) or not 0 < dt <= self.membrane_time_constant:
            raise ValueError("dt must be positive and no greater than tau")
        self.membrane_potential += (
            -(self.membrane_potential - self.resting_potential) + self.input_current
        ) * dt / self.membrane_time_constant

    def is_refractory(self, time: float) -> bool:
        return time - self.last_spike_time < self.refractory_period - 1e-9

    def fire(self, time: float) -> None:
        self.spiked = True
        self.last_spike_time = time
        self.membrane_potential = self.reset_potential

    def step(self, dt: float, time: float) -> bool:
        """Advance [time, time + dt); stamp spikes at the interval end."""
        if not isfinite(time) or not isfinite(dt) or not 0 < dt <= self.membrane_time_constant:
            raise ValueError("Invalid step time or dt")
        self.spiked = False
        if not self.is_refractory(time):
            self.integrate(dt)
            if self.membrane_potential >= self.threshold:
                self.fire(time + dt)
        self.input_current = 0.0
        return self.spiked

    def reset(self) -> None:
        self.membrane_potential = self.resting_potential
        self.last_spike_time = float("-inf")
        self.input_current = 0.0
        self.spiked = False
