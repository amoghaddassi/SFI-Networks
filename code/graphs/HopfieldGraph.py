from graphs.Graph import *

import numpy as np
import networkx as nx
import copy
import json

class HopfieldGraph(Graph):
	"""Given an existing graph topology with N nodes, and some set of orientations
	(assuming binary valued nodes), trains the network using the Hopfield learning rule."""
	def __init__(self, graph = None, states = None, thresholds = None, nodes = None, adj_mat = None,
		filename = None):
		if filename != None:
			#if we are given a filename, we set all atrributes based on the file
			self.read(filename)
			self.nodes = [Graph.Node(0, []) for _ in range(len(self.stored_states[0]))]
			self.set_node_attributes()
			self.thresholds = [0 for _ in range(len(self.nodes))]
			return
		if graph == None:
			self.nodes = nodes
		else:
			self.nodes = graph.nodes
		self.stored_states = states
		if not thresholds:
			self.thresholds = [0 for _ in range(len(self.nodes))]
		else:
			self.thresholds = thresholds
		if adj_mat == None:
			self.set_adj_matrix()
		else:
			self.adj_matrix = adj_mat
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
				self.train_edge(i, j)
		self.set_node_attributes() #adjusts all node attributes to be same with training vals 

	def train_edge(self, i, j):
		"""Trains the edge (i, j)."""
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

	def set_node_attributes(self):
		"""Call this method after training to set node attributes (in_edges and weights) to
		be consistent with the trained weight matrix."""
		for i in range(len(self.nodes)):
			neighbors, weights = [], {}
			for j in range(len(self.nodes)):
				if self.adj_matrix[i][j] == 0: continue #means there's no edge to consider
				#adds j to i's neighbors and sets the correct weight
				neighbors.append(self.nodes[j])
				weights[self.nodes[j]] = self.weights[i][j]
			#sets nodes i's attributes to neighbors and weights
			self.nodes[i].in_edges = neighbors
			self.nodes[i].weights = weights

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
		for v in node.in_edges:
			total += node.weights[v] * v.val

		if total > self.thresholds[node_index]:
			self.nodes[node_index].val = 1
		else:
			self.nodes[node_index].val = 0
			
	def to_networkx(self):
		"""Returns an unweighted networkx graph with the same topology as self."""
		N = len(self.nodes)
		g = nx.Graph()
		g.add_nodes_from(range(N))
		for i in range(N):
			for j in range(i):
				if self.adj_matrix[i][j]:
					g.add_edge(i, j)
		return g

	def binarize_weights(self, val = 1):
		"""For all non-zero edges, changes weight to be sign(trained edge) * val."""
		for i in range(len(self.nodes)):
			for j in range(i):
				if self.weights[i][j]:
					weight = np.sign(self.weights[i][j]) * val
					self.weights[i][j] = weight
					self.weights[j][i] = weight
		self.set_node_attributes()


	def num_edges(self):
		"""Returns the number of edges in the network."""
		count = 0
		for i in range(len(self.nodes)):
			for j in range(i):
				if self.adj_matrix[i][j]:
					count += 1
		return count

	def copy(self):
		"""Returns a copy of the graph."""
		nodes = [Graph.Node(n.val, None) for n in self.nodes]
		new_graph = HopfieldGraph(None, self.stored_states, self.thresholds, 
			nodes = nodes, adj_mat = copy.deepcopy(self.adj_matrix))
		new_graph.train()
		return new_graph

	def degree_dist(self):
		"""Returns a list of the degrees of all nodes."""
		return [len(node.in_edges) for node in self.nodes]

	def energy(self, state = None):
		"""Returns the energy of state for this network. If no state is provided, uses the current state."""
		if state is None:
			state = [node.val for node in self.nodes]
		tot = 0
		for i in range(len(state)):
			for j in range(len(state)):
				tot += self.weights[i][j] * (2*state[i] - 1) * (2*state[j] - 1)
		return -1/2 * tot

	def save(self, filename):
		"""Saves a json file containing self's weights matrix and stored states to filename."""
		outdict = dict()
		outdict["states"] = self.stored_states
		outdict["weights"] = self.weights
		outdict["adj_matrix"] = self.adj_matrix
		with open(filename, 'w') as file:
			json.dump(outdict, file)

	def read(self, filename):
		"""Given a json text file of the type saved by filename, makes self that graph by setting the
		stored states and weight matrix appropriately."""
		with open(filename) as jsonFile:
			data = json.load(jsonFile)
			self.weights = data["weights"]
			self.adj_matrix = data["adj_matrix"]
			self.stored_states = data["states"]
		