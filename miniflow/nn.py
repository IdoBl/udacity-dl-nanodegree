"""
This script builds and runs a graph with miniflow.
"""
from miniflow import *

w, x, y, z = Input(), Input(), Input(), Input()

f = Add(w, x, y, z)

feed_dict = {w: 2, x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} + {} = {} (according to miniflow)".format(feed_dict[w], feed_dict[x], feed_dict[y], feed_dict[z], output))
