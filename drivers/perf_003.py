"""Benchmark: assigning to model attributes in a tight loop.

Prints elapsed milliseconds for the timed section; lower is better.
Covers plain fields, nested models that validate on assignment, private
attributes and cached properties.
"""

import time
from functools import cached_property

from pydantic import BaseModel, ConfigDict


class Inner(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    inner_field1: str
    inner_field2: int


class Model(BaseModel):
    field1: str
    field2: int
    field3: float
    inner1: Inner
    inner2: Inner

    _private_field1: str
    _private_field2: int

    @cached_property
    def prop_cached1(self) -> str:
        return self.field1 + self._private_field1


def make() -> Model:
    return Model(
        field1='a', field2=1, field3=1.0,
        inner1=Inner(inner_field1='a', inner_field2=1),
        inner2=Inner(inner_field1='b', inner_field2=2),
    )


def work(m: Model) -> None:
    m.field1 = 'test1'
    m.field2 = 43
    m.field3 = 4.0
    m.inner1.inner_field1 = 'test inner1'
    m.inner1.inner_field2 = 421
    m.inner2.inner_field1 = 'test inner2'
    m.inner2.inner_field2 = 422
    m._private_field1 = 'test2'
    m._private_field2 = 44


ITERS, REPS = 20000, 5
m = make()
work(m)  # warm up
best = float('inf')
for _ in range(REPS):
    start = time.perf_counter()
    for _ in range(ITERS):
        work(m)
    best = min(best, (time.perf_counter() - start) * 1000)

# Check assignment still validates and is still visible.
assert m.field2 == 43 and m.inner1.inner_field2 == 421 and m._private_field2 == 44
try:
    m.inner1.inner_field2 = 'not an int'
except Exception:
    pass
else:
    raise AssertionError('assignment validation was lost')

print(f'{best:.4f}')
