"""Benchmark: cost of importing the library.

Prints elapsed milliseconds for the timed section; lower is better.
The import is timed in-process, so run the whole script again for another
sample rather than looping inside it.
"""

import time

start = time.perf_counter()
import pydantic  # noqa: E402
elapsed = (time.perf_counter() - start) * 1000

# Check the imported library is usable.
class _M(pydantic.BaseModel):
    a: int
    b: str = 'x'

assert _M(a=1).b == 'x'
assert _M.model_json_schema()['properties']['a']['type'] == 'integer'
assert pydantic.TypeAdapter(list[int]).validate_python([1, 2]) == [1, 2]

print(f'{elapsed:.4f}')
