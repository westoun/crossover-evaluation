import hashlib
from typing import List, Dict, Any

from quasim import Circuit


class Cache:

    _entries: Dict

    def __init__(self):
        self._entries = {}

    def add(self, circuit: Circuit, fitness: Any) -> None:
        circuit_hash = self.hash(circuit)
        self._entries[circuit_hash] = fitness

    def __contains__(self, circuit: Circuit) -> bool:
        circuit_hash = self.hash(circuit)
        return circuit_hash in self._entries

    def get(self, circuit: Circuit) -> List[float]:
        circuit_hash = self.hash(circuit)
        return self._entries[circuit_hash]

    def __len__(self) -> int:
        return len(self._entries.keys())

    def hash(self, circuit: Circuit):
        return hash(circuit.__repr__())
