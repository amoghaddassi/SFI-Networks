"""File to contain functions that returns Hopfield Graphs built using certain models."""
from graphs.Graph import *
from graphs.HopfieldGraph import *

from bit_string_helpers import *
from random_graphs import *
from hopfield_evaluation import *

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

def random_dist(M):
	"""Given a number of patterns, returns a random distribution to use to generate states."""
	dist = []
	for _ in range(int(2 ** (M - 1))):
		dist.append(random.sample(range(100), 1)[0])
	#normalize dist
	dist = [d / sum(dist) for d in dist]
	return dist

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

def pruned_hopfield(patterns, edges, graph = None, shuffle = True):
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
	if graph == None:
		N = len(patterns[0])
		graph = fully_connected(N)
		hop_net = HopfieldGraph(graph, patterns)
		hop_net.train()
	else:
		hop_net = graph
	#add all the edges to a priority queue (only looking at bottom porition of adj. mat.)
	pq = []
	for i in range(N):
		for j in range(i):
			weight = hop_net.weights[i][j]
			if hop_net.adj_matrix[i][j] == 0: #only adds edges that exist to pq
				continue
			pq.append(Entry((i, j), abs(weight)))
	if shuffle:
		random.shuffle(pq) #randomizes order before heapification
	heapq.heapify(pq) #heapifies for popping smallest edges
	#remove the N - e lowest priority edges and set their weights to be 0 in hop_net
	for _ in range(hop_net.num_edges() - edges):
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

	remove_edge(i, j)
	remove_edge(j, i)
	k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
	#commented code lines correspond to using the fancy weighting model or not.
	while i == k or hopfield_graph.adj_matrix[i][k] != 0 or zero_edge(i, k):
	#while i == k or hopfield_graph.adj_matrix[i][k] != 0:
		k = random.sample(range(len(hopfield_graph.nodes)), 1)[0]
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

def node_groups(hopfield_graph):
	"""Given a graph, returns a dict where the key is the pattern and the value is a list of nodes
	that belong to that pattern group."""
	groups = dict()
	#iterate over all nodes and collect the pattern group
	for i in range(len(hopfield_graph.nodes)):
		pattern = []
		for state in hopfield_graph.stored_states:
			pattern.append(state[i])
		pattern_str = bit_list_to_string(pattern)
		#if the ndoes pattern or flipped pattern is already in the dict, add to appropriate bucket.
		if pattern_str in groups:
			groups[pattern_str].append(hopfield_graph.nodes[i])
		elif flip_bits(pattern_str) in groups:
			groups[flip_bits(pattern_str)].append(hopfield_graph.nodes[i])
		#else make a new bucket
		else:
			groups[pattern_str] = [hopfield_graph.nodes[i]]
	return groups

def hopfield_sbm(edge_probs, states, num_edges):
	"""edge_probs: dict of dicts where both have keys that are the group's orientation patterns
	and values (in the inner dicts) that are the probabilities of making an edge between groups."""
	def adjust_probs():
		"""Changes edge_probs so that the expected number of edges is num_edges."""
		#calculates the expected number of edges with current edge_probs
		groups = node_groups(hop_graph)
		expected_edges = 0
		considered_pairs = set() #running collection of all bit string pairs considered
		for str1 in edge_probs.keys():
			for str2 in edge_probs.keys():
				#if we've already considered this pair, skip to avoid double counting
				if (str1, str2) in considered_pairs:
					continue
				#add the pair to considered_pairs
				considered_pairs.add((str1, str2))
				#expected edges for this group pairing is #(str1) * #(str2) * prob
				groups_str1 = str1 if str1 in groups else flip_bits(str1)
				groups_str2 = str2 if str2 in groups else flip_bits(str2)
				expected_edges += len(groups[groups_str1]) * len(groups[groups_str2]) * edge_probs[str1][str2]
		#adjusts all probs by multiplying by the ratio between num_edges and expected_edges
		adjustment_ratio = 2 * num_edges / expected_edges #times by 2 bc we have undirected edges
		#does this same style of iteration as above to adjust.
		for str1 in edge_probs.keys():
			for str2 in edge_probs.keys():
				edge_probs[str1][str2] = adjustment_ratio * edge_probs[str1][str2]

	N = len(states[0])
	#Constructs hopfield graph with states that has no edges
	nodes = [Graph.Node(0, []) for _ in range(N)]
	graph = Graph(nodes)
	hop_graph = HopfieldGraph(graph, states)
	#Renormalizes edge_probs so that the expected number of edges will be num_edges
	adjust_probs()
	#For each node pair, construct and train an edge with the appropriate probability
	patterns_cache = dict() #caches patterns of nodes to avoid n^2 runtime
	for i in range(N):
		for j in range(i):
			#gets the patterns of i and j so we know which groups they're in
			if i not in patterns_cache:
				i_pattern = bit_list_to_string([state[i] for state in states])
				i_pattern = i_pattern if i_pattern in edge_probs else flip_bits(i_pattern)
				patterns_cache[i] = i_pattern
			if j not in patterns_cache:
				j_pattern = bit_list_to_string([state[j] for state in states])
				j_pattern = j_pattern if j_pattern in edge_probs else flip_bits(j_pattern)
				patterns_cache[j] = j_pattern
			#gets prob of an edge and flips a p coin
			prob = edge_probs[patterns_cache[i]][patterns_cache[j]]
			#TODO: change to uniform (0, 1)
			p_coin_flip = 1 if np.random.uniform() < prob else 0
			if p_coin_flip: #means we're making the edge
				hop_graph.adj_matrix[i][j] = 1
				hop_graph.adj_matrix[j][i] = 1
	#trains the graph according to the updated adj matrix
	hop_graph.train()
	return hop_graph

def clustering_matrix(hopfield_graph):
	"""Given a Hopfield graph, returns the matrix (2^M-1 * 2^M-1) of probabilities of an edge
	existing between any two groups. Define the groups as usual. If norm_group_size, will normalize
	all probabilities wrt to the group size of each cluster."""
	group_cache = dict()
	def get_group(node):
		"""Returns the group of the given node assuming groups has been defined."""
		if node not in group_cache:
			for group, nodes in groups.items():
				if node in nodes:
					group_cache[node] = group
					break
		return group_cache[node]

	groups = node_groups(hopfield_graph)
	cluster_mat = dict()
	for group, nodes in groups.items(): #iterates over all groups and counts all edges to other groups.
		edge_probs = {g: 0 for g in groups.keys()}
		for node in nodes: #counts all the edges and tallys which group the other node is in.
			for adj in node.in_edges:
				edge_probs[get_group(adj)] += 1
		#makes each count a percent of edges that could possibly exist between the two groups
		row = {} 
		for other_group, count in edge_probs.items():
			other_size = len(groups[other_group])
			if other_group == group:
				possible_edges = other_size * (other_size - 1)
			else:
				possible_edges = other_size * len(nodes)
			row[other_group] = count / possible_edges
		cluster_mat[group] = row
	return cluster_mat

def hopfield_lattice(states, k, intersection = False):
	"""Returns a graph where each node is connected to its k nearest neighbors.
	A node's nearest neighbors are the nodes which have the highest magnitude edges
	between each other (same notion as in the pruning rule). If intersection is true,
	will only add edges where the nodes are mutual nearest neighbors."""
	def nearest_neighbors(hop_graph):
		"""Returns a list (using the same indicies as hop_graph.nodes) of sets where each
		set is the indicies of that nodes nearest neighbors."""
		def node_neighbors(i):
			"""Returns a set that contains the indicies of i's k nearest neighbors."""
			class Entry:
				"""Class to hold items in the priority queue. Each object has an item
				which will be some int (representing a node) and a priority (the absolute
				value of the edge weight after training). Compares the items solely on priority."""
				def __init__(self, item, priority):
					self.item = item
					self.priority = priority
				def __lt__(self, other):
					return self.priority < other.priority
			neighbors = set()
			pq = []
			for j in range(len(hop_graph.nodes)):
				ij_weight = hop_graph.full_weights[i][j] #gets weight of trained ij edge
				#adds edge entry to the pq. negative mag. since this is a min pq
				pq.append(Entry(j, -abs(ij_weight)))
			heapq.heapify(pq) #heapifies the array so we can pop the k top edges
			for _ in range(k):
				neighbor = heapq.heappop(pq)
				neighbors.add(neighbor.item)
			return neighbors

		res = [] #what we're going to return
		for i in range(len(hop_graph.nodes)):
			res.append(node_neighbors(i))
		return res

	def make_edge(i, j):
		"""Makes an edge (i, j) in the graph, and trains the edge weight."""
		hop_graph.adj_matrix[i][j] = 1
		hop_graph.adj_matrix[j][i] = 1
		hop_graph.train_edge(i, j)

	nodes = [Graph.Node(0, []) for _ in range(len(states[0]))]
	hop_graph = HopfieldGraph(Graph(nodes), states) #makes an empty hop graph to serve as the blank slate
	hop_graph.train()
	neighbors = nearest_neighbors(hop_graph)

	for i in range(len(states[0])):
		for j in neighbors[i]:
			#iterates over all pairs of nearest neighbors
			if intersection:
				if i not in neighbors[j]:
					continue
			make_edge(i, j)
	hop_graph.set_node_attributes() #sets all node attributes correctly
	return hop_graph

