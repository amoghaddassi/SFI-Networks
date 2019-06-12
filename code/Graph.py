import numpy as np
import math

class Graph:
	
	class Node:
		def __init__(self, val, in_edges):
			"""val: node's value, 1 - on, 0 - off
			in_edges: list of node objects going into this node."""
			self.val = val
			self.in_edges = in_edges
	
	def __init__(self, nodes):
		"""nodes is a list of node objects."""
		self.nodes = nodes

	def graph_of(node_list, edge_list):
		"""Returns a graph with nodes that have values from node_list.
		edge_list specifies edges as a list of lists, where the ith
		entry contains the nodes going into the ith node. Nodes in the lists
		are specified by the index in node_list."""
		nodes = []
		#creates all node objects with no connections
		for val in node_list:
			nodes.append(Graph.Node(val, None))

		#specifies the connections
		for i in range(len(nodes)):
			in_edges = edge_list[i]
			in_nodes = []
			for edge in in_edges:
				in_nodes.append(nodes[edge])
			nodes[i].in_edges = in_nodes

		return Graph(nodes)

	def state(self):
		"""Returns the current state of the graph as a list of node states."""
		return [node.val for node in self.nodes]

	def __repr__(self):
		return self.state().__str__()

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
