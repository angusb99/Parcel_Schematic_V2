"""Parcel Address Associator.

Takes addresses already linked to cadastre parcels (one DataFrame, INPUT_COLUMNS),
profiles the address *styles* on each parcel (the schematics), learns how those
styles map onto each other, and associates potential duplicate addresses.
Rule-based and statistical only: no LLMs, no libpostal, no I/O.

    from parcel_address_associator import AssociatorSession
    s = AssociatorSession(df).run()
    s.duplicate_groups()
"""

from .config import Config
from .engine import analyse, prepare_input, run
from .model import INPUT_COLUMNS
from .session import AssociatorSession
from .synthetic import make_synthetic

__version__ = "0.2.0"
__all__ = ["AssociatorSession", "Config", "INPUT_COLUMNS", "analyse", "make_synthetic",
           "prepare_input", "run"]
