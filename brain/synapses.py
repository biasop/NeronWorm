"""Compatibility imports for the original plural module name."""
from .synapse import ChemicalSynapse, EventQueue, GapJunction

Synapse = ChemicalSynapse
__all__ = ["ChemicalSynapse", "GapJunction", "EventQueue", "Synapse"]
