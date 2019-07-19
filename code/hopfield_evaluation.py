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
