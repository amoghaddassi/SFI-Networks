"""File to contain functions that returns Hopfield Graphs built using certain models."""
from graphs.Graph import *
from graphs.HopfieldGraph import *

from bit_string_helpers import *
from random_graphs import *

import numpy as np
import random
import heapq
from scipy.misc import comb

def random_state(N, p = .5):
	"""Returns a random binary list of len N, where each bit has p chance of being a 1."""
	return [np.random.binomial(1, p) for _ in range(N)]

def related_states(N, M, p):
	"""Generates a set of related binary patterns (as per variation of information). First
	produces a random seed state of length N, and then produces all M-1 other states by
	flipping p portion of bits in the seed state."""
	seed = random_state(N)
	states = [seed]
	for _ in range(M - 1):
		states.append(flip_porition(seed, p))
	return states
	
def states_from_dist(group_dist, N):
	"""Given some distribution of groups (must be a multiple of 2) and some number of nodes N,
	returns a set of patterns st the number of nodes in each cluster (based on orientations) follows
	the given group_dist."""
	M = int(np.log2(len(group_dist))) + 1 #since num groups is 2^M-1
	#get all possible node patterns and consolidate based on which have some orientation pattern
	bit_strings = all_bit_strings(M)
	states = []
	for bs in bit_strings:
		if bs not in states and flip_bits(bs) not in states:
			states.append(bs)
	#now create all the patterns by choosing a sequence for each node.
	patterns = [[] for _ in range(M)]
	for _ in range(N):
		#selects base pattern for node then randomly flips bits
		pattern = np.random.choice(states, p = group_dist)
		if np.random.binomial(1, .5):
			pattern = flip_bits(pattern)
		for i in range(M):
			patterns[i].append(int(pattern[i]))
	return patterns

def random_hopfield(N, M, graph = None):
	"""Returns a trained Hopfield network of N nodes where the M
	stored states are chosen randomly. Can provide a graph architecture
	if want something diff from fully connected network."""
	if not graph:
		graph = fully_connected(N)
	states = [random_state(N) for _ in range(M)]
	hopfield_graph = HopfieldGraph(graph, states)
	hopfield_graph.train()
	return hopfield_graph

def pruned_hopfield(patterns, edges):
	"""Returns a trained Hopfield network that only has the edges highest weight
	edges (by absolute value) after the network is trained on the patterns."""
	class Entry:
		"""Class to hold items in the priority queue. Each object has an item
		which will be some tuple (representing an edge) and a priority (the absolute
		value of the edge weight after training). Compares the items solely on priority."""
		def __init__(self, item, priority):
			self.item = item
			self.priority = priority
		def __lt__(self, other):
			return self.priority < other.priority

	#first train the fully connect hopfield net on the patterns
	N = len(patterns[0])
	graph = fully_connected(N)
	hop_net = HopfieldGraph(graph, patterns)
	hop_net.train()
	#add all the edges to a priority queue (only looking at bottom porition of adj. mat.)
	pq = []
	for i in range(N):
		for j in range(i):
			weight = hop_net.weights[i][j]
			heapq.heappush(pq, Entry((i, j), abs(weight)))
	#remove the N - e lowest priority edges and set their weights to be 0 in hop_net
	for _ in range(int(comb(N, 2) - edges)):
		min_edge = heapq.heappop(pq)
		edge = min_edge.item
		hop_net.weights[edge[0]][edge[1]] = 0
		hop_net.weights[edge[1]][edge[0]] = 0
		hop_net.adj_matrix[edge[0]][edge[1]] = 0
		hop_net.adj_matrix[edge[1]][edge[0]] = 0
	hop_net.set_node_attributes() #adjusts all the node attributes appropriately
	return hop_net

def rewire(i, j, hopfield_graph):
	"""Changes edge (i, j) to edge (i, k) st i != k, the edge doesn't already exist, and that
	the edge has the same signed edge weight as (i, j) (doesn't make edges that have 0 weight)."""
	def remove_edge(i, j):
		"""Returns the abs of the original weight to set the value of the new edge."""
		weight = hopfield_graph.weights[i][j]
		hopfield_graph.adj_matrix[i][j] = 0
		hopfield_graph.adj_matrix[j][i] = 0
		hopfield_graph.weights[i][j] = 0
		hopfield_graph.weights[j][i] = 0
		return abs(weight)

	def add_edge(i, j, w = None):
		"""Makes the edge (i, j) with weight w."""
		hopfield_graph.adj_matrix[i][j] = 1
		hopfield_graph.adj_matrix[j][i] = 1
		#trains the edge to get the sign
		hopfield_graph.train_edge(i, j)
		if w != None:
			sign = np.sign(hopfield_graph.weights[i][j])
			hopfield_graph.weights[i][j] = sign * w
			hopfield_graph.weights[j][i] = sign * w

	def zero_edge(i, j):
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

	weight = remove_edge(i, j)
	remove_edge(j, i)
	k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
	#commented code lines correspond to using the fancy weighting model or not.
	#while i == k or hopfield_graph.adj_matrix[i][k] != 0 or zero_edge(i, k):
	while i == k or hopfield_graph.adj_matrix[i][k] != 0:
		k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
	#add_edge(i, k, weight)
	#add_edge(k, i, weight)
	add_edge(i, k)
	add_edge(k, i)

def rewired_hopfield(hopfield_graph, rewire_prob):
	"""Given some hopfield_graph, rewires each edge (same protocol as in Watts - 
	Strogatz Model) with prob rewire_prob."""
	for i in range(len(hopfield_graph.nodes)):
		for j in range(i):
			if not hopfield_graph.adj_matrix[i][j]:
				continue
			if np.random.binomial(1, rewire_prob):
				rewire(i, j, hopfield_graph)
	hopfield_graph.set_node_attributes() #adjusts the graph's nodes after the rewiring


