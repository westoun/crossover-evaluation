import numpy as np
from quasim.gates import (
    IGate,
    T, S, H, CX, Gate
)
from typing import List, Type


class Identity(Gate):
    """Identity gate."""

    matrix: np.ndarray = np.eye(2)


CLIFFORD_PLUS_T: List[IGate] = [H, S, T, CX]
CLIFFORD_PLUS_T_PLUS_I: List[IGate] = [H, S, T, CX, Identity]


def gateset_to_string(GateSet: List[Type]) -> str:
    gate_type_names = [
        GateType.__name__ for GateType in GateSet
    ]

    repr = "[" + ", ".join(gate_type_names) + "]"
    return repr
