"""Benchmark: preparing the internal schema for a model with many fields.

Prints elapsed milliseconds for the timed section; lower is better.
Every field carries the same annotation.
"""

import time
from typing import Union

from pydantic import create_model

REPS = 5
ROUNDS = 10
N_FIELDS = 100

IntStr = Union[int, str]
Model = create_model(
    'WideModel',
    __config__={'defer_build': True},
    **{f'f{i}': (IntStr, ...) for i in range(N_FIELDS)},
)

Model.model_rebuild(force=True)  # warm up

best = float('inf')
for _ in range(REPS):
    start = time.perf_counter()
    for _ in range(ROUNDS):
        Model.model_rebuild(force=True)
    best = min(best, (time.perf_counter() - start) * 1000)

assert len(Model.model_fields) == N_FIELDS
assert Model(**{f'f{i}': i for i in range(N_FIELDS)}).f0 == 0

print(f'{best:.4f}')
