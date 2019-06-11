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

	def erdos_reyni(n, p, prob_on = .5):
		"""Creates an Erdos Reyni random graph with parameters n, p.
		Each node has prob_on chance of being a 1."""
		nodes = []
		for _ in range(n):
			val = np.random.binomial(1, prob_on)
			nodes.append(Graph.Node(val, None))

		for node_index in range(n):
			edge_list = []
			for edge_index in range(n):
				if node_index == edge_index:
					continue #don't want self loops.
				p_coin_flip = np.random.binomial(1, p)
				if p_coin_flip:
					#means we're adding an edge
					edge_list.append(nodes[edge_index])
			nodes[node_index].in_edges = edge_list

		return Graph(nodes)

	def stochastic_block_model(community_alloc, edge_matrix, prob_on = .5):
		"""
		community_alloc: list of r numbers indicating how many nodes should
		be in each community. Only need numbers since actual node assignments
		are arbitrary.
		edge_matrix: rxr matrix (list of lists) that gives the prob of edges between
		groups i and j.
		"""
		cache = {}
		def community_number(node):
			if node in cache: return cache[node]
			for community, nodes in communities.items():
				if node in nodes:
					cache[node] = community
					return community

		nodes, communities = [], {}
		#adds nodes to communities based on the given sizes.
		for i in range(len(community_alloc)):
			communities[i] = []
			for _ in range(community_alloc[i]):
				val = np.random.binomial(1, prob_on)
				node = Graph.Node(val, None)
				communities[i].append(node)
				nodes.append(node)

		for node1 in nodes:
			edge_list = []
			for node2 in nodes:
				if node1 == node2: continue
				comm1, comm2 = community_number(node1), community_number(node2)
				p_coin_flip = np.random.binomial(1, edge_matrix[comm1][comm2])
				if p_coin_flip:
					edge_list.append(node2)
			node1.in_edges = edge_list

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
