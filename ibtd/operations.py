from  typing import Callable, Tuple, Union, Optional, Dict
import numpy as np
from inspect import signature


class Op:
    def __init__(self, name: str, description: str, op: Callable, partial_difs: Tuple[Callable]):
        assert len(signature(op).parameters) == len(partial_difs)
        self._name = name
        self._desc = description
        self._op = op
        self._partials = partial_difs

    def __call__(self, *args):
        return self._op.__call__(*args)

    def __str__(self):
        return f"{self._name}: {self._desc}"

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._desc

    def partial(self, i):
        return self._partials[i]


class OpTable:
    def __init__(self, *ops: Op):
        self._ops = { op.name: op for op in ops }

    def __getitem__(self, op_name: str) -> Op:
        return self._ops[op_name]

    def __len__(self):
        return len(self._ops)
    
    def op_descriptions(self) -> Dict[str, str]:
        return {op.name: op.description for op in self._ops.values()}
    

add = Op(
    "add",
    "Scalar or vecrtor addition. If one arg is a matrix then the other arg "
    "must be a matrix of the same shape",
    lambda x, y: x + y, 
    (
        lambda x, y, c: c, 
        lambda x, y, c: c
    )
)


smul = Op(
    "smul",
    "Scalar multiplication. The first arg must be a scalar, the second arg "
    "may be a scalar or a matrix",
    lambda x, y: x * y,
    (
        lambda x, y, c: (c * y).sum(), 
        lambda x, y, c: c * x * np.ones_like(y),
    )
)


mmul = Op(
    "mmul",
    "Matrix multiplication. Both args must be matrices and have compatible "
    "shapes.",
    lambda x, y: x @ y,
    (
        lambda x, y, c: c @ y.T, 
        lambda x, y, c: x.T @ c,
    )
)


relu = Op(
    "relu",
    "For each elament x in a matrix set x = max(x, 0)", 
    lambda x: np.maximum(x, 0.0),
    (
        lambda x, c: np.where(x > 0, 1.0, 0.0),
    ),
)


loss = Op(
    "loss",
    "Calculate the RMS loss between a target and observed values",
    lambda target, actual: np.sqrt(np.mean(np.square(target - actual))),
    (
        lambda t, a, c: c * 0.5 * (t - a) * t.size, 
        lambda t, a, c: c * 0.5 * (a - t) * a.size,
    )
)


default_op_table = OpTable(add, smul,  mmul, relu, loss)

