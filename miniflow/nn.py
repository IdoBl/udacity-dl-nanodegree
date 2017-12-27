"""
This script builds and runs a graph with miniflow.
"""
from miniflow import *

# Testing Add

# w, x, y, z = Input(), Input(), Input(), Input()
#
# f = Add(w, x, y, z)
#
# feed_dict = {w: 2, x: 4, y: 5, z: 10}
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)

# # should output 19
# print("{} + {} + {} + {} = {} (according to miniflow)".format(feed_dict[w], feed_dict[x], feed_dict[y], feed_dict[z], output))

# ******************************************

# Testing Linear

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output) # should be 12.7 with this example

# ******************************************

# Testing LinearVec

import numpy as np

X, W, b = Input(), Input(), Input()

f = LinearVec(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

"""
Output should be:
[[-9., 4.],
"""