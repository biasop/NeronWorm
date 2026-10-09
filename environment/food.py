from dataclasses import dataclass


@dataclass
class Food:
    """Stationary chemical source, replaced after consumption."""
    x: float
    y: float
    radius: float = 7.0
    energy: float = 24.0
