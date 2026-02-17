from quasim.gates import (
    IGate,
    T, S, H, CX
)
from typing import List, Type


CLIFFORD_PLUS_T: List[IGate] = [H, S, T, CX]


def gateset_to_string(GateSet: List[Type]) -> str:
    gate_type_names = [
        GateType.__name__ for GateType in GateSet
    ]

    repr = "[" + ", ".join(gate_type_names) + "]"
    return repr
