from typing import Any, Optional, Tuple
from collections import ChainMap
from uuid import uuid4
import numpy as np
from .operations import OpTable, default_op_table


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