from typing import Callable, Dict, List, Set, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Measurement:
    """A result of calling a benchmark function."""
    attrs: Dict[str, Any]
    value: float
    name: Optional[str] = None

@dataclass
class Benchmark:
    """A benchmark function and its associated parameter configurations."""
    func: Callable
    param_configs: List[Dict[str, Any]]
    groups: Set[str]
    label: Optional[str] = None

    @property
    def fn_name(self) -> str:
        """Name of the benchmark function."""
        return self.func.__name__

    @property
    def name(self) -> str:
        """Full name of the benchmark, including lablel as suffix if present."""
        if self.suffix:
            return f"{self.fn_name}_{self.label}"
        return self.fn_name

    def __call__(self) -> List[Measurement]:
        """Run the benchmark function with all parameter configurations."""
        measurements = []
        for params in self.param_configs:
            result = self.func(**params)
            
            # Handle different return types
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                # List of results
                for r in result:
                    assert isinstance(r, Measurement)
                    # Create new Measurement with name and merged attrs
                    measurements.append(Measurement(
                        attrs={**params, **r.attrs},
                        value=r.value,
                        name=self.fn_name  # Use base_name for measurements
                    ))
            elif isinstance(result, Measurement):
                # Single Measurement
                measurements.append(Measurement(
                    attrs={**params, **result.attrs},
                    value=result.value,
                    name=self.fn_name  # Use base_name for measurements
                ))
            else:
                # Single value - must be float
                if not isinstance(result, float):
                    raise TypeError(f"Benchmark {self.name} returned {type(result)}, expected float, Measurement, or iterable of Measurements")
                measurements.append(Measurement(
                    attrs=params,
                    value=result,
                    name=self.fn_name  # Use base_name for measurements
                ))
        return measurements
