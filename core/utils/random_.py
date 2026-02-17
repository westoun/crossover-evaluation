
from quasim.gates import (
    Gate,
    CGate,
    CCGate,
    Swap,
    IGate,
)
from quasim import Circuit
from random import randint, sample, choices
from typing import List, Type


def random_circuit(qubit_num: int, gate_count: int, gate_set: List[Type[IGate]]) -> Circuit:
    circuit = Circuit(qubit_num)

    for _ in range(gate_count):

        gate = random_gate(
            qubit_num=qubit_num, GateSet=gate_set)
        circuit.apply(gate)

    return circuit


def random_gate(qubit_num: int, GateSet: List[Type]) -> IGate:
    # assume each gate config to be equally likely

    weights = []
    for GateType in GateSet:
        if issubclass(GateType, CGate) or GateType == Swap:
            weights.append(qubit_num * (qubit_num - 1))
        elif issubclass(GateType, CCGate):
            weights.append(qubit_num * (qubit_num - 1) * (qubit_num - 2))
        else:
            weights.append(qubit_num)

    GateType = choices(GateSet, weights=weights, k=1)[0]

    if "theta" in vars(GateType)["__annotations__"]:
        raise NotImplementedError(
            "Handling of parameterized gates has not been implemented yet.")

    if issubclass(GateType, Gate):
        target_qubit = randint(0, qubit_num - 1)
        return GateType(target_qubit)

    elif issubclass(GateType, CGate):
        target_qubit, control_qubit = sample(range(0, qubit_num), 2)
        return GateType(control_qubit, target_qubit)

    elif issubclass(GateType, CCGate):
        target_qubit, control_qubit1, control_qubit2 = sample(
            range(0, qubit_num), 3)
        return GateType(control_qubit1, control_qubit2, target_qubit)

    elif GateType == Swap:
        qubit1, qubit2 = sample(range(0, qubit_num), 2)
        return Swap(qubit1, qubit2)

    else:
        raise NotImplementedError(
            f"Could not assign {GateType} to a group of gate types.")
