from Graph import *

import math

class KuramotoGraph(Graph):
	"""Given an existing graph topology, initializes all nodes to have values
	in [0, 2pi) which we call the node's natural frequency."""
	def __init__(self, graph):
		self.nodes = graph.nodes
		self.natural_phases = {}
		for node in self.nodes:
			freq = np.random.uniform(0, 2*math.pi)
			node.val = freq
			self.natural_phases[node] = freq

	def dynamic(self, node):
		"""Generic dynamics function for the Kuramoto model. Can be passed into
		any of the update functions in dynamics.py."""
		summation = sum([math.sin(v.val - node.val) for v in node.in_edges])
		change = self.natural_phases[node] + summation
		node.val += change
		self.renormalize()

	def renormalize(self):
		"""Adjusts all node vals to be in the range [0, 2pi)."""
		for node in self.nodes:
			while node.val < 0:
				node.val += 2 * math.pi
			while node.val > 2 * math.pi:
				node.val -= 2 * math.pi

	def erdos_reyni(n, p):
		graph = Graph.erdos_reyni(n, p)
		return KuramotoGraph(graph)

	def stochastic_block_model(community_alloc, edge_matrix):
		graph = Graph.stochastic_block_model(community_alloc, edge_matrix)
		return KuramotoGraph(graph)