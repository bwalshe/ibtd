import numpy as np
from ibtd import Graph

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