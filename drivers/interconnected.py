"""Driver for the exponential-traversal candidate (PR13523).

Called as `python interconnected.py <N>`. Builds a chain of N models where each
references the previous three, which is the shape that makes
`gather_schemas_for_cleaning()` blow up exponentially before the fix.

Prints `import=<s> build=<s>` so the build cost is separable from import cost.
Usable both as a wall-clock driver and (with small N) as a callgrind driver.

Measured on this box, `python interconnected.py`:

    N     base (b75fadba)   gold (c333d775)
    15    0.098s            0.051s
    20    0.648s            0.041s
    25   16.322s            0.041s
    30   >100s (timeout)    0.043s
    60    -                 0.069s
   120    -                 0.139s
"""

import sys
import time
from typing import Any, Optional

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

_t = time.perf_counter()
from pydantic import BaseModel, create_model  # noqa: E402

t_import = time.perf_counter() - _t

_t = time.perf_counter()
classes: list[type[BaseModel]] = []
for i in range(N):
    fields: dict[str, Any] = {'value': (int, ...)}
    for j, prev in enumerate(classes[-3:]):
        fields[f'ref{j}'] = (Optional[prev], None)
    classes.append(create_model(f'Node{i}', **fields))
t_build = time.perf_counter() - _t

# Prove the models actually work, so a "fast" but broken patch cannot pass by
# building nothing useful.
last = classes[-1]
instance = last.model_validate({'value': 1, 'ref2': {'value': 2, 'ref2': {'value': 3}}})
assert instance.ref2.ref2.value == 3

print(f'import={t_import:.4f} build={t_build:.4f}')
