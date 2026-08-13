"""Callgrind driver: model class building + core schema generation.

Called as `python build_models.py <REPS>`. Instruction counts are compared
between two REPS values so the fixed `import pydantic` cost cancels out:

    Ir(reps=N) - Ir(reps=0)  =  N x cost of one `work()` call

Every model gets a unique name so pydantic's schema caches cannot short-circuit
repeated iterations. Keep this file dependency-free and deterministic -- no
randomness, no clock reads -- or the differencing stops being exact.
"""

import sys
from typing import Optional

REPS = int(sys.argv[1])

from pydantic import create_model  # noqa: E402  (after argv read, before timing)


def work(i: int) -> None:
    inner = create_model(f'Inner{i}', a=(int, ...), b=(str, 'x'), c=(float, 0.0))
    create_model(
        f'Outer{i}',
        **{
            f'f{j}': field
            for j, field in enumerate(
                [
                    (int, ...),
                    (str, 's'),
                    (float, 1.0),
                    (bool, False),
                    (Optional[int], None),
                    (list[int], []),
                    (dict[str, int], {}),
                    (inner, ...),
                    (Optional[inner], None),
                    (list[inner], []),
                ]
            )
        },
    )


for _i in range(REPS):
    work(_i)
