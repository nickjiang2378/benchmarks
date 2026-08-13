"""Benchmark: preparing internal schemas for nested, recursive and
discriminated-union models.

Prints elapsed milliseconds for the timed section; lower is better.
Each model is defined once with schema preparation deferred, then its schema is
prepared repeatedly, so the measurement covers preparation rather than class
definition.
"""

import time
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

REPS = 5
ROUNDS = 30


class Deferred(BaseModel):
    model_config = ConfigDict(defer_build=True)


class Leaf(Deferred):
    x: int
    y: str


class Nested(Deferred):
    field1: str
    field2: list[int]
    field3: dict[str, float]
    leaf: Leaf
    optional_leaf: Optional[Leaf] = None


class Cat(Deferred):
    kind: Literal['cat']
    meows: int


class Dog(Deferred):
    kind: Literal['dog']
    barks: int


class Tagged(Deferred):
    pet: Annotated[Union[Cat, Dog], Field(discriminator='kind')]
    pets: list[Annotated[Union[Cat, Dog], Field(discriminator='kind')]] = Field(default_factory=list)


class Recursive(Deferred):
    name: str
    nested: Optional[Nested] = None
    children: list['Recursive'] = Field(default_factory=list)


MODELS = [Leaf, Nested, Tagged, Recursive]

for _m in MODELS:  # warm up
    _m.model_rebuild(force=True)

best = float('inf')
for _ in range(REPS):
    start = time.perf_counter()
    for _ in range(ROUNDS):
        for m in MODELS:
            m.model_rebuild(force=True)
    best = min(best, (time.perf_counter() - start) * 1000)

# Check the prepared schemas behave correctly.
assert Nested(field1='a', field2=[1], field3={'k': 1.0}, leaf={'x': 1, 'y': 'b'}).leaf.x == 1
assert Tagged(pet={'kind': 'dog', 'barks': 2}).pet.barks == 2
assert Recursive(name='a', children=[{'name': 'b'}]).children[0].name == 'b'

print(f'{best:.4f}')
