"""File for functions that make random/synthetic graphs using the Graph class and subclasses."""
from graphs.Graph import *
from graphs.HopfieldGraph import *
from graphs.KuramotoGraph import *

import numpy as np
import random
import heapq
from scipy.misc import comb

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

def ring_lattice(N, k):
	"""Construcuts a regular ring lattice with N nodes where each
	node is connected to each of its k/2 neighbors on both sides."""
	#makes the nodes 
	nodes = [Graph.Node(0, None) for _ in range(N)]

	for i in range(N):
		edge_list = []
		for j in range(N):
			#makes edge (i, j) if j is within k/2 spots from i circularly
			dist_ij = abs(i - j) % (N - k/2)
			if 0 < dist_ij <= k / 2:
				edge_list.append(nodes[j])
		nodes[i].in_edges = edge_list

	return Graph(nodes)

def small_world(N, k, beta):
	"""Constructs a small world random graph using the Watts-Strogatz model."""
	def rewire(graph, i, j):
		"""Removes edge (i, j) and adds some edge (i, k) st
		i != k and (i, k) is not already present in the graph."""
		def remove_edge(m, n):
			"""Removes directed edge (m, n) from graph."""
			for node in graph.nodes[m].in_edges:
				if graph.nodes[n] == node:
					graph.nodes[m].in_edges.remove(node)

		def add_edge(m, n):
			"""Adds directed edge (m, n) from graph."""
			n_node = graph.nodes[n]
			graph.nodes[m].in_edges.append(n_node)

		def edge_exists(m, n):
			"""Checks if undirected edge (m, n) is in graph."""
			for node in graph.nodes[m].in_edges:
				if graph.nodes[n] == node:
					return True
			for node in graph.nodes[n].in_edges:
				if graph.nodes[m] == node:
					return True

		remove_edge(i, j)
		remove_edge(j, i)

		k = int(np.random.uniform(0, N))
		while i == k or edge_exists(i, k):
			k = int(np.random.uniform(0, N))

		add_edge(i, k)
		add_edge(k, i)

	base_graph = ring_lattice(N, k) #base graph that we rewire
	
	for i in range(N):
		for j in range(i + 1, int(i + k/2 + 1)):
			#loops over all of i's k/2 right neighbors
			neighbor_index = j % N #normalizes to account for circular structure
			if np.random.binomial(1, beta):
				#means we rewire edge (i, j)
				rewire(base_graph, i, neighbor_index)
	
	return base_graph

def fully_connected(N):
	"""Returns a fully connected graph with N nodes."""
	nodes = [Graph.Node(0, None) for _ in range(N)]
	for node in nodes:
		node.in_edges = [n for n in nodes if n != node]
	return Graph(nodes)

def random_edges(N, e):
	"""Returns a graph with N nodes and e edges where the
	edges are placed randomly. The edges are undirected and are
	sampled uniformly from the lower triangle of the adj. matrix."""
	rows = list(range(N)) #potential values for i in (i, j)
	#normalize rows to get the probs of selecting each i
	row_probs = [rows[i] / sum(rows) for i in range(len(rows))]
	edges = set()
	while len(edges) < e:
		#samples row according to row probs
		row = np.random.choice(rows, p = row_probs)
		col = np.random.choice(range(row))
		edges.add((row, col))
	#now builds the graph with N nodes and given edges
	nodes = [Graph.Node(0, []) for _ in range(N)]
	for edge in edges:
		row, col = edge[0], edge[1]
		nodes[row].in_edges.append(nodes[col])
		nodes[col].in_edges.append(nodes[row])
	return Graph(nodes)

def random_state(N, p = .5):
	"""Returns a random binary list of len N, where each bit has p chance of being a 1."""
	return [np.random.binomial(1, p) for _ in range(N)]

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
	return hop_net
