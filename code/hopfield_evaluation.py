import numpy as np
from scipy.misc import comb
from scipy.stats import entropy
import csv
import random
import time
import networkx as nx
import heapq

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

def overlap_performance_metric(hopfield_graph, type = "stability", p = .2, by_node = False):
	"""Measures the overlap performance metric for each of the stored states and returns the average.
	If stability is true, runs the graph from the stored state at each iteration. If false, measures 
	retrievability by running the graph from a 25% maligned state. p is percent to flip for high_deg.
	If by_node, will return the average performance for each node individually, instead of the average."""
	perf = []
	for state in hopfield_graph.stored_states:
		#flips some of the bits in the state according to the metric.
		if type == "stability":
			hopfield_graph.set_node_vals(state)
		elif type == "retrievability":
			hopfield_graph.set_node_vals(flip_porition(state, p))
		elif type == "high_deg_errors":
			hopfield_graph.set_node_vals(flip_high_deg(hopfield_graph, state, p))
		steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
		state_perf = []
		if [1 - b for b in steady_state] == state: #means we have a flipped state so we adjust
			steady_state = [1 - b for b in steady_state]
		for i in range(len(state)):
			if state[i] == steady_state[i]:
				state_perf.append(1)
			else:
				state_perf.append(0)
		perf.append(state_perf)
	#perf is now a 2d matrix where the ith col is the matching results of the ith node
	np_arr = np.array(perf) #converts to numpy matrix to take columnwise mean
	res = np.mean(np_arr, axis = 0)
	if by_node:
		return res #returns whole array if we want by node values
	else:
		return np.average(res) #returns average across all nodes otherwise

def stability_performance_metric(hopfield_graph):
	return overlap_performance_metric(hopfield_graph)

def retrievability_performance_metric(hopfield_graph, nonneg = True, p = .25):
	return overlap_performance_metric(hopfield_graph, "retrievability", p = p)

def high_degree_errors_performance_metric(hopfield_graph, nonneg = False, p = .15):
	return overlap_performance_metric(hopfield_graph, "high_deg_errors", p = p)

def hopfield_performance(hopfield_graph, metric = vi_performance_metric, runs = 15, p = .2):
	"""Uses the metric function to get the performance of a graph on a single run, and 
	returns the average metric over all runs."""
	total_perf = 0
	for _ in range(runs):
		if metric is retrievability_performance_metric or metric is high_degree_errors_performance_metric:
			total_perf += metric(hopfield_graph, p = p)
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

def flip_high_deg(hopfield_graph, state, portion_to_flip):
	"""Returns a version of state, where the portion_to_flip highest degree
	nodes in the hopfield_graph have been flipped."""
	new_state = state.copy() #shouldn't destructively modify the state.
	#now get the positions of the bits to flip.
	class Entry:
		"""Class to hold items in the priority queue. Each object has an item
		which will be some int (representing the node's index) and a priority (the degree of the node). 
		Compares the items solely on priority."""
		def __init__(self, item, priority):
			self.item = item
			self.priority = priority
		def __lt__(self, other):
			return self.priority < other.priority
	heap = []
	deg_dist = hopfield_graph.degree_dist()
	for i in range(len(deg_dist)):
		node = Entry(i, -deg_dist[i]) #negate degree since this is a min pq
		heapq.heappush(heap, node)
	num_to_pop = int(portion_to_flip * len(deg_dist)) #number of nodes to flip
	#flips that number of bits in new_state
	for _ in range(num_to_pop):
		highest_deg = heapq.heappop(heap).item #gets the index of the next highest deg node
		new_state[highest_deg] = 1 - new_state[highest_deg] #flips that bit
	return new_state

def deg_perf_corr(nodes, edges, num_states,runs = 1000):
	"""For a pruned graph model, gets the correlation between degree of a node and 
	retrievability performance of that node over runs."""
	corrs = []
	states = [random_state(nodes) for _ in range(num_states)]
	ph = pruned_hopfield(states, edges)
	for i in range(runs):
		perf = []
		for _ in range(100):
			run_perf = overlap_performance_metric(ph, 'retrievability', True, .25, True)
			perf.append(run_perf)
		perf = np.mean(np.array(perf), axis = 0) #sums over all perf runs
		corr = np.corrcoef(perf, ph.degree_dist())[0][1]
		corrs.append(corr)
	return np.nanmean(corrs), np.nanstd(corrs), ph

def stability_metric_for_memory_load(hopfield_graph):
	"""Returns the average of percent of nodes that are not stable from stored states across
	all stored states. Used in particular as a comprable measure to the analytic stabilty result."""
	perf = []
	for state in hopfield_graph.stored_states:
		hopfield_graph.set_node_vals(state)
		asynch_update(hopfield_graph, hopfield_graph.dynamic)
		matches = 0
		for i in range(len(state)):
			if state[i] == hopfield_graph.nodes[i].val:
				matches += 1
		perf.append(matches / len(state))
	return np.average(perf)


def node_groups(hopfield_graph, size = False):
	"""Given a graph, returns a dict where the key is the pattern and the value is a list of nodes
	that belong to that pattern group. If size is True, returns the size of each group, not list of nodes."""
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
	if size:
		res = {}
		for k, v in groups.items():
			res[k] = len(v)
		return res
	return groups

def clustering_matrix(hopfield_graph, counts = False):
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
		if counts:
			#means we don't normalize and just use the edge_probs as the row
			cluster_mat[group] = edge_probs
			continue
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

def memory_load(hopfield_graph):
	"""Returns the proportionality constant between the average degree of all nodes in the
	network and number of stored patterns."""
	avg_deg = (2 * hopfield_graph.num_edges()) / len(hopfield_graph.nodes)
	return len(hopfield_graph.stored_states) / avg_deg

def prop_in_group_edges(hopfield_graph):
	"""Given a hopfield graph, proportion of possible in group edges that are in the graph."""
	clustering_mat = clustering_matrix(hopfield_graph)
	in_group_props = []
	for key, value in clustering_mat.items():
		in_group_props.append(value[key]) #adds the diagonal term of the clustering matrix
	return np.average(in_group_props)

def edge_magnitude_dist(hopfield_graph):
	"""Given a hopfield graph, returns a dictionary whose keys are all the possible edge magnitudes
	between nodes, and whose values are the proportion of edges of that magnitude over the number of 
	possible edges of that magnitude."""
	def group_diff(g1, g2):
		"""Given the bit strings of two nodes, calculates magnitude of the edge."""
		weight = 0
		for b1, b2 in zip(list(g1), list(g2)):
			#iterates over the pairwise bits of the two groups
			b1, b2 = int(b1), int(b2) #makes them ints for easy calc
			weight += (2 * b1 - 1) * (2 * b2 - 1) #adds 1 for matching bits, -1 otherwise
		return abs(weight)
	group_sizes = node_groups(hopfield_graph, size = True)
	edges = dict() #symbolic numerator for the result
	possible_edges = dict() #symbolic denominator
	edge_counts = clustering_matrix(hopfield_graph, counts = True)
	for k1, v1 in edge_counts.items():
		for k2, v2 in v1.items():
			#iterate over all the possible node group pairs
			diff = group_diff(k1, k2)
			if diff not in edges:
				edges[diff] = 0
			edges[diff] += v2 #adds number of existing edges to the diff group
			if diff not in possible_edges:
				possible_edges[diff] = 0
			if k1 == k2:
				#means we add n(n -1)possible edges (n being group size of k)
				possible_edges[diff] += group_sizes[k1] * (group_sizes[k1] - 1)
			else:
				#means we multiply group sizes to get possible edges
				possible_edges[diff] += group_sizes[k1] * group_sizes[k2]
	#Now have to do the "division" btw edges and possible_edges
	res = dict()
	for k, v in edges.items():
		res[k] = v / possible_edges[k]
	return res





