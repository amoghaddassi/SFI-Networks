import numpy as np
import random

def random_edge(graph):
	"""Given a graph, returns a random edge (i, j) that is in the graph."""
	i = random.sample(range(len(graph.nodes)), 1)[0]
	j = random.sample(range(len(graph.nodes)), 1)[0]

	if graph.adj_matrix[i][j] == 0:
		#means we found a null edge, so we repeat
		return random_edge(graph)
	return i, j

def random_null_edge(graph):
	"""Returns a random tuple (i, j) whose edge is not in the graph."""
	i = random.sample(range(len(graph.nodes)), 1)[0]
	j = random.sample(range(len(graph.nodes)), 1)[0]

	if graph.adj_matrix[i][j] != 0:
		#means we found a null edge, so we repeat
		return random_edge(graph)
	return i, j

def remove_edge(i, j, hopfield_graph):
	"""Returns the abs of the original weight to set the value of the new edge."""
	weight = hopfield_graph.weights[i][j]
	hopfield_graph.adj_matrix[i][j] = 0
	hopfield_graph.adj_matrix[j][i] = 0
	hopfield_graph.weights[i][j] = 0
	hopfield_graph.weights[j][i] = 0

def add_edge(i, j, hopfield_graph, w = None):
	"""Makes the edge (i, j) with weight w."""
	hopfield_graph.adj_matrix[i][j] = 1
	hopfield_graph.adj_matrix[j][i] = 1
	#trains the edge to get the sign
	hopfield_graph.train_edge(i, j)
	if w != None:
		sign = np.sign(hopfield_graph.weights[i][j])
		hopfield_graph.weights[i][j] = sign * w
		hopfield_graph.weights[j][i] = sign * w

def zero_edge(i, j, hopfield_graph):
	"""Returns true if the trained edge (i, j) has weight 0."""
	#makes the edge.
	hopfield_graph.adj_matrix[i][j] = 1
	hopfield_graph.adj_matrix[j][i] = 1
	#trains the edge and checks if 0
	hopfield_graph.train_edge(i, j)
	zero = hopfield_graph.weights[i][j] == 0
	#removes the edge
	hopfield_graph.adj_matrix[i][j] = 0
	hopfield_graph.adj_matrix[j][i] = 0
	#resets edge
	hopfield_graph.train_edge(i, j)
	return zero

def rewire(i, j, hopfield_graph):
	"""Changes edge (i, j) to edge (i, k) st i != k, the edge doesn't already exist, and that
	the edge has the same signed edge weight as (i, j) (doesn't make edges that have 0 weight)."""
	remove_edge(i, j, hopfield_graph)
	remove_edge(j, i, hopfield_graph)
	k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
	#commented code lines correspond to using the fancy weighting model or not.
	while i == k or hopfield_graph.adj_matrix[i][k] != 0 or zero_edge(i, k, hopfield_graph):
	#while i == k or hopfield_graph.adj_matrix[i][k] != 0:
		k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
	add_edge(i, k, hopfield_graph)
	add_edge(k, i, hopfield_graph)

def edge_overlap(g1, g2):
	"""Given two graphs, returns the number of common edges."""
	common = 0
	for i in range(len(g1.nodes)):
		for j in range(i):
			g1_edge = g1.adj_matrix[i][j] == 1
			g2_edge = g2.adj_matrix[i][j] == 1
			if g1_edge and g2_edge:
				common += 1
	return common

