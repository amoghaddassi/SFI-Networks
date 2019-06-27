import numpy as np
from scipy.misc import comb
from scipy.stats import entropy
import csv
import random
import time
import networkx as nx

from graphs.Graph import *
from graphs.HopfieldGraph import *
from graphs.KuramotoGraph import *

from random_graphs import *
from dynamics import *
from bit_string_helpers import *
from hopfield_models import *

def match_coefficient(hopfield_graph, i, j):
	"""For a given hopfield graph and nodes i and j, computes the matching coefficient
	across all stored states. Define it as the percent's of matching states absolute
	deviation from .5."""
	matches = 0
	for state in hopfield_graph.stored_states:
		if state[i] == state[j]: matches += 1
	return abs(matches / len(hopfield_graph.stored_states) - .5)

def vi_performance_metric(hopfield_graph):
	"""Initializes the hopfield graph to a random state, finds a fixed point, and returns 
	the min variation of information between the fixed point and a stored state."""
	hopfield_graph.random_config()
	steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
	if not steady_state: return vi_performance_metric(hopfield_graph) #repeats if no state is found
	steady_state_str = bit_list_to_string(steady_state)
	return min([variation_of_information(steady_state_str, bit_list_to_string(stored_state))
			for stored_state in hopfield_graph.stored_states])

def overlap_performance_metric(hopfield_graph, stability = True, nonneg = False):
	"""Measures the overlap performance metric for each of the stored states and returns the average.
	If stability is true, runs the graph from the stored state at each iteration. If false, measures 
	retrievability by running the graph from a 25% maligned state."""
	perf = 0
	for state in hopfield_graph.stored_states:
		if stability:
			hopfield_graph.set_node_vals(state)
		else:
			hopfield_graph.set_node_vals(flip_porition(state, .25))
		steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
		state_perf = 0
		for i, j in zip(steady_state, state):
			if nonneg:
				state_perf += 1 if i == j else 0
			else:
				state_perf += (2 * i - 1) * (2 * j - 1)
		perf += state_perf / len(state)
	return perf / len(hopfield_graph.stored_states)

def stability_performance_metric(hopfield_graph):
	return overlap_performance_metric(hopfield_graph)

def retrievability_performance_metric(hopfield_graph, nonneg = False):
	return overlap_performance_metric(hopfield_graph, False, nonneg)

def hopfield_performance(hopfield_graph, metric = vi_performance_metric, runs = 100, nonneg = False):
	"""Uses the metric function to get the performance of a graph on a single run, and 
	returns the average metric over all runs."""
	total_perf = 0
	for _ in range(runs):
		if metric is retrievability_performance_metric:
			total_perf += metric(hopfield_graph, nonneg = nonneg)
		else:
			total_perf += metric(hopfield_graph)
	return total_perf / runs

def runtime(hopfield_graph, runs = 100):
	"""Measures the average time needed to find a fixed point in hopfield_graph when starting
	from a random config."""
	total_time = 0
	for _ in range(runs):
		hopfield_graph.random_config()
		start = time.time()
		fixed_point(hopfield_graph, hopfield_graph.dynamic)
		total_time += time.time() - start
	return total_time / runs

def random_edges_for_sim(N, num_stored_states, edge_count):
	g = random_edges(N, edge_count)
	return random_hopfield(N, num_stored_states, g)

def pruned_edges_for_sim(N, num_stored_states, edge_count):
	patterns = [random_state(N) for _ in range(num_stored_states)]
	return pruned_hopfield(patterns, edge_count)

def random_edges_for_p_sim(patterns, edges):
	g = random_edges(len(patterns[0]), edges)
	hop_graph = HopfieldGraph(g, patterns)
	hop_graph.train()
	return hop_graph

def pruned_edges_for_p_sim(patterns, edges):
	return pruned_hopfield(patterns, edges)

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

def clustering_matrix(hopfield_graph, norm_group_size = False):
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
		total_count = sum(list(edge_probs.values()))
		if norm_group_size:
			row = {group: count / len(groups[group]) for group, count in edge_probs.items()} #norms counts by group size
			row = {group: normed / sum(row.values()) for group, normed in row.items()}
		elif total_count == 0:
			row = {group: 0 for group, count in edge_probs.items()}
		else:
			row = {group: count / total_count for group, count in edge_probs.items()} #converts counts to probs
		cluster_mat[group] = row
	return cluster_mat

def small_world_coeff(hopfield_graph):
	"""Returns the small world coefficient omega (Lrand / L - C/Clatt) of the given graph."""
	try:
		nx_graph = hopfield_graph.to_networkx()
		N = len(hopfield_graph.nodes)
		#computes path length stats
		L = nx.average_shortest_path_length(nx_graph)
		random_graph = random_edges(N, hopfield_graph.num_edges())
		random_hop_graph = HopfieldGraph(random_graph, None)
		Lrand = nx.average_shortest_path_length(random_hop_graph.to_networkx())
		#computes clustering states
		C = nx.clustering(nx_graph)
		C = np.average(list(C.values())) #since clustering returns a dict of all nodes clustering coeffs
		latt = ring_lattice(N, int(hopfield_graph.num_edges() * 2 / N))
		latt_hop = HopfieldGraph(latt, None)
		Clatt = nx.clustering(latt_hop.to_networkx())
		Clatt = np.average(list(Clatt.values()))
		#returns omega
		return (Lrand / L) - (C / Clatt)
	except nx.exception.NetworkXError:
		return -1 #for when the graph isn't connected
