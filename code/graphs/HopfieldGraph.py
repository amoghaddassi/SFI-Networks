from graphs.Graph import *

import numpy as np

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
		self.weights = [[0 for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]
		self.full_weights = [[0 for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]

	def train(self):
		"""Trains the edge weights of the network on self.states using
		the Hopfield training rule."""
		for i in range(len(self.nodes)):
			for j in range(i): # only train the lower triangle of the weights matrix
				
				weight = 0
				for state in self.stored_states:
					weight += (2*state[i] - 1) * (2*state[j] - 1)
				if self.adj_matrix[i][j] == 0:
					#means there's no edge so we don't train anything.
					self.weights[i][j] = 0
					self.weights[j][i] = 0
				else:
					self.weights[i][j] = weight
					self.weights[j][i] = weight
				self.full_weights[i][j] = weight
				self.full_weights[j][i] = weight

	def set_node_vals(self, state):
		"""Given some orientation for all the node values, sets
		all the nodes to the appropriate states."""
		for i in range(len(state)):
			self.nodes[i].val = state[i]

	def random_config(self, p = .5):
		"""Sets the states of nodes randomly, p is prob of a node being on."""
		state = [np.random.binomial(1, p) for _ in range(len(self.nodes))]
		self.set_node_vals(state)

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
