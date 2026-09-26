"""V5.2 unified signal registry."""
from dataclasses import dataclass, asdict
from typing import Any, Callable, Mapping

SignalFn = Callable[..., Mapping[Any, float]]

@dataclass(frozen=True)
class SignalSpec:
    name: str
    version: str
    family: str
    description: str
    fn: SignalFn
    enabled: bool = True
    oos_replayable: bool = True

    def metadata(self):
        d = asdict(self)
        d.pop("fn", None)
        return d

class SignalRegistry:
    def __init__(self):
        self._signals = {}
    def register(self, spec):
        if spec.name in self._signals:
            raise ValueError(f"signal already registered: {spec.name}")
        self._signals[spec.name] = spec
        return spec
    def upsert(self, spec):
        self._signals[spec.name] = spec
        return spec
    def get(self, name):
        return self._signals[name]
    def names(self, enabled_only=True):
        return [n for n,s in self._signals.items() if not enabled_only or s.enabled]
    def specs(self, enabled_only=True):
        return [s for s in self._signals.values() if not enabled_only or s.enabled]
    def metadata(self):
        return [s.metadata() for s in self._signals.values()]

GLOBAL_SIGNAL_REGISTRY = SignalRegistry()

def register_signal(name, version, family, description, fn, *, enabled=True, oos_replayable=True):
    return GLOBAL_SIGNAL_REGISTRY.upsert(SignalSpec(name, version, family, description, fn, enabled, oos_replayable))
