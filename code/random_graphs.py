"""File for functions that make random graphs using the Graph.py class."""
from Graph import *

import numpy as np
import random

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
	encoded in base N during the implemenation."""
	nodes = [Graph.Node(0, []) for _ in range(N)]
	#min (0, 1) --> 0 + 1*N
	#max (N-1, N) --> N-1 + N*N
	edges = random.sample(range(N, N**2 + N), e)
	for edge in edges:
		#converts edge from base 10 to a 2 dig number (i, j) in base N
		i = edge % N
		j = edge // N
		nodes[i].in_edges.append(nodes[j])
		nodes[j].in_edges.append(nodes[i])
	return Graph(nodes)
