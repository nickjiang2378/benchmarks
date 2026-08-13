"""Benchmark: preparing internal schemas for models built out of deeply
parameterised standard-library generics.

Prints elapsed milliseconds for the timed section; lower is better.
The annotations nest unions inside containers inside optionals.
"""

import time
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

REPS = 5
ROUNDS = 25


class Deferred(BaseModel):
    model_config = ConfigDict(defer_build=True)


class Complex(Deferred):
    field1: Union[str, int, float]
    field2: list[dict[str, Union[int, float]]]
    field3: Optional[list[Union[str, int]]]


class Nested(Deferred):
    field1: str
    field2: list[int]
    field3: dict[str, float]


class Outer(Deferred):
    nested: Nested
    optional_nested: Optional[Nested]
    many: list[Nested]
    mapped: dict[str, list[Nested]]
    mixed: Union[Nested, Complex, None]


MODELS = [Complex, Nested, Outer]

for _m in MODELS:  # warm up
    _m.model_rebuild(force=True)

best = float('inf')
for _ in range(REPS):
    start = time.perf_counter()
    for _ in range(ROUNDS):
        for m in MODELS:
            m.model_rebuild(force=True)
    best = min(best, (time.perf_counter() - start) * 1000)

assert Complex(field1='a', field2=[{'k': 1}], field3=['a', 1]).field2[0]['k'] == 1
assert Outer(nested={'field1': 'a', 'field2': [1], 'field3': {}}, optional_nested=None,
             many=[], mapped={}, mixed=None).nested.field1 == 'a'

print(f'{best:.4f}')
