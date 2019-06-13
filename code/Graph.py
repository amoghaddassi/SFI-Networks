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

class HopfieldGraph(Graph):
	"""Given an existing graph topology with N nodes, and some set of orientations
	(assuming binary valued nodes), trains the network using the Hopfield learning rule."""
	def __init__(self, graph, states, thresholds = None):
		self.nodes = graph.nodes
		self.stored_states = states
		if not thresholds:
			self.thresholds = [0 for _ in range(len(self.nodes))]
		else:
			self.thresholds = thresholds
		self.set_adj_matrix()
		self.set_weight_matrix()

	def set_adj_matrix(self):
		"""Makes a symmetric adjacency matrix for the given graph."""
		self.adj_matrix = []
		for i in range(len(self.nodes)):
			self.adj_matrix.append([])
			for j in range(len(self.nodes)):
				if self.nodes[j] in self.nodes[i].in_edges:
					self.adj_matrix[i].append(1)
				else:
					self.adj_matrix[i].append(0)

	def set_weight_matrix(self):
		"""Makes an edge weight matrix with all values initialized to 0."""
		row = [0 for _ in range(len(self.nodes))]
		self.weights = [row for _ in range(len(self.nodes))]

	def train(self):
		"""Trains the edge weights of the network on self.states using
		the Hopfield training rule."""
		for i in range(len(self.nodes)):
			for j in range(i): # only train the lower triangle of the weights matrix
				if self.adj_matrix[i][j] == 0:
					#means there's no edge so we don't train anything.
					continue
				weight = 0
				for state in self.stored_states:
					weight += (2*state[i] - 1) * (2*state[j] - 1)
				self.weights[i][j] = weight
				self.weights[j][i] = weight

	def set_node_vals(self, state):
		"""Given some orientation for all the node values, sets
		all the nodes to the appropriate states."""
		for i in range(len(state)):
			self.nodes[i].val = state[i]

	def dynamic(self, node):
		"""Given some node, updates its value according to the Hopfield rule."""
		node_index = 0
		for n in self.nodes:
			if n == node: break
			else: node_index += 1
		total = 0
		for j in range(len(self.nodes)):
			total += self.weights[node_index][j] * self.nodes[j].val

		if total > self.thresholds[node_index]:
			self.nodes[node_index].val = 1
		else:
			self.nodes[node_index].val = 0


