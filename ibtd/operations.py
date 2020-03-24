from  typing import Callable, Tuple, Union, Optional, Any, Dict
import numpy as np
from inspect import signature
from uuid import uuid4
from collections import ChainMap


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
    
    def op_descriptions(self) -> Dict[str, str]:
        return {op.name: op.description for op in self._ops}
    

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


class Graph:
    def __init__(self, op_table: OpTable=default_op_table):
        self._op_table = op_table
        self._steps = dict()
        self._step_values = dict()
        self._variables = dict()
        self._constants = set()
        self._values = ChainMap(self._step_values, self._variables)

    def _new_name(self, name):
        if name == None:
            name = str(uuid4())
        if name in self._values:
            raise Exception(f"allready have a value named {name}")
        return name

    def new_variable(self, initial_value: Any, name: Optional[str]=None):
        name = self._new_name(name)
        self._variables[name] = initial_value
        return name

    def new_placeholder(self, name: Optional[str]=None):
        name = self._new_name(name)
        self._constants.add(name)
        return name

    def add_step(self, op_name: str, op_args: Tuple[str], name=None) -> str:
        step_name = self._new_name(name)
        assert step_name not in self._steps
        valid_ids = self._constants.union(self._values.keys())
        for arg in op_args:
            assert arg in valid_ids
        self._steps[step_name] = op_name, op_args
        self._step_values[step_name] = None
        return step_name

    def evaluate(self, step_name, **kwargs):
        all_values = ChainMap(self._values, kwargs)
        for z, (op, xs) in self._steps.items():
            arg_values = [all_values[x] for x in xs]
            self._step_values[z] = self._op_table[op](*arg_values)
        return self._values[step_name]

    def gradients(self, step_name: str, **kwargs):
        result = self.evaluate(step_name, **kwargs)
        deltas = { step_name: np.ones_like(result) }
        all_values = ChainMap(self._values, kwargs)
        for  z, (op, xs) in reversed(list(self._steps.items())):
            z_value = deltas[z]
            for i, x in enumerate(xs):
                diff_fn = self._op_table[op].partial(i)
                x_values = [all_values[x] for x in xs]
                delta = diff_fn(*x_values, z_value)
                deltas[x] = deltas[x] + delta if x in deltas else delta
        return {k: v for k, v in deltas.items() if k in self._variables}


if __name__ == "__main__":
    g = Graph()
    W = g.new_variable(np.eye(2), "W")
    x = g.new_placeholder("x")
    b = g.new_variable(np.ones(2).reshape(2,1), "b")
    z = g.add_step("add", 
        (
            g.add_step("mmul", (W, x)), 
            b
        )
    )
    x= 2 * np.ones(2).reshape(2, 1)
    print(f"At {x}\nz = {g.evaluate(z, x=x)}")
    grads = g.gradients(z, x=x)
    for var, val in grads.items():
        print(f"{var} has gradient:\n{val}")
