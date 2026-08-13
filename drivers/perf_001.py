"""Benchmark: creating a group of models that reference one another.

Prints elapsed milliseconds for the timed section; lower is better.
Model names are unique per repetition so that repeated runs do not reuse
previously computed results.
"""

import time
from typing import Any, Optional

from pydantic import BaseModel, create_model

N = 22          # number of models in the chain
REPS = 1        # one pass is enough at this size


def build(tag: int) -> type[BaseModel]:
    classes: list[type[BaseModel]] = []
    for i in range(N):
        fields: dict[str, Any] = {'value': (int, ...)}
        for j, prev in enumerate(classes[-3:]):
            fields[f'ref{j}'] = (Optional[prev], None)
        classes.append(create_model(f'N{tag}_{i}', **fields))
    return classes[-1]


best = float('inf')
for rep in range(REPS):
    start = time.perf_counter()
    last = build(rep)
    best = min(best, (time.perf_counter() - start) * 1000)

# Check the models behave correctly, so a run that builds something
# degenerate fails loudly instead of reporting a time.
inst = last.model_validate({'value': 1, 'ref2': {'value': 2, 'ref2': {'value': 3}}})
assert inst.ref2.ref2.value == 3
assert inst.model_dump(exclude_none=True) == {'value': 1, 'ref2': {'value': 2, 'ref2': {'value': 3}}}

print(f'{best:.4f}')
