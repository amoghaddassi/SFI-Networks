import numpy as np
from scipy.misc import comb
import csv
import random

from graphs.Graph import *
from graphs.HopfieldGraph import *
from graphs.KuramotoGraph import *

from random_graphs import *
from dynamics import *

def match_coefficient(hopfield_graph, i, j):
	"""For a given hopfield graph and nodes i and j, computes the matching coefficient
	across all stored states. Define it as the percent's of matching states absolute
	deviation from .5."""
	matches = 0
	for state in hopfield_graph.stored_states:
		if state[i] == state[j]: matches += 1
	return abs(matches / len(hopfield_graph.stored_states) - .5)

def variation_of_information(A, B):
	"""Returns the variation of information between two bit strings A and B.
	A and B should both be strings of 1's and 0's."""
	def bit_entropy(A):
		"""Returns the entropy of a single bit string."""
		counts = {0: 0, 1: 0}
		for b in A:
			counts[int(b)] += 1
		return count_entropy([val for key, val in counts.items()])

	def joint_bit_entropy(A, B):
		"""Returns the joint entropy of two bit strings."""
		counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
		for a, b in zip(A, B):
			counts[(int(a), int(b))] += 1
		return count_entropy([val for key, val in counts.items()])

	def count_entropy(counts):
		"""Returns the entropy of some counts list."""
		normed = [float(x)/float(sum(counts)) for x in counts]
		info = 0
		for p in normed:
			if p <= .0000001: continue #avoids div by 0 errors
			info -= p * np.log2(p)
		return info	

	ha = bit_entropy(A)
	hb = bit_entropy(B)
	hab = joint_bit_entropy(A, B)
	return 2*hab - ha - hb

def vi_performance_metric(hopfield_graph):
	"""Initializes the hopfield graph to a random state, finds a fixed point, and returns 
	the min variation of information between the fixed point and a stored state."""
	def bit_list_to_string(lst):
		"""Converts a list of bits to a string."""
		n = ""
		for b in lst:
			n += str(b)
		return n
	hopfield_graph.random_config()
	steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
	if not steady_state: return vi_performance_metric(hopfield_graph) #repeats if no state is found
	steady_state_str = bit_list_to_string(steady_state)
	return min([variation_of_information(steady_state_str, bit_list_to_string(stored_state))
			for stored_state in hopfield_graph.stored_states])

def overlap_performance_metric(hopfield_graph, stability = True):
	"""Measures the overlap performance metric for each of the stored states and returns the average.
	If stability is true, runs the graph from the stored state at each iteration. If false, measures 
	retrievability by running the graph from a 25% maligned state."""
	def flip_porition(state, p):
		"""Flips p portion of bits in state."""
		num_to_flip = int(p * len(state)) #rounds up how many to flip
		pos_to_flip = random.sample(range(len(state)), num_to_flip) #chooses which indicies to flip
		#flips the bits
		res = state.copy()
		for i in pos_to_flip:
			res[i] = 1 - res[i]
		return res

	perf = 0
	for state in hopfield_graph.stored_states:
		if stability:
			hopfield_graph.set_node_vals(state)
		else:
			hopfield_graph.set_node_vals(flip_porition(state, .25))
		steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
		state_perf = 0
		for i, j in zip(steady_state, state):
			state_perf += (2 * i - 1) * (2 * j - 1)
		perf += state_perf / len(state)
	return perf / len(hopfield_graph.stored_states)

def stability_performance_metric(hopfield_graph):
	return overlap_performance_metric(hopfield_graph)

def retrievability_performance_metric(hopfield_graph):
	return overlap_performance_metric(hopfield_graph, False)

def hopfield_performance(hopfield_graph, metric = vi_performance_metric, runs = 100):
	"""Uses the metric function to get the performance of a graph on a single run, and 
	returns the average metric over all runs."""
	total_perf = 0
	for _ in range(runs):
		total_perf += metric(hopfield_graph)
	return total_perf / runs

def hopfield_perf_sim(N, num_stored_states, graph_model,runs_per_edge_count = 100,
	edge_iterator=None, filepath = "data/random_edges/", show_runs = True):
	"""Outputs a csv with 2 columns: edge count, performance. Edge count refers
	to the number of edges we place randomly in an N node graph built using graph_model. 
	Performance refersto the constructed graph's performance when it is trained on 
	num_stored_states randomly generated stored states. graph_model is a function that returns
	a hopfield graph with parameters N, num_stored_states, and edge_count. edge_iterator is how we
	vary the edge counts during the simulation."""
	if not edge_iterator:
		edge_iterator = range(1, int(comb(N, 2) + 1))
	data = [["edges", "variation_of_information", "retrievability", "stability"]]
	for edge_count in edge_iterator:
		for _ in range(runs_per_edge_count):
			hop_graph = graph_model(N, num_stored_states, edge_count)
			vi = hopfield_performance(hop_graph, metric = vi_performance_metric)
			retrievability = hopfield_performance(hop_graph, metric = retrievability_performance_metric)
			stability = hopfield_performance(hop_graph, metric = stability_performance_metric)
			if show_runs:
				print([edge_count, vi, retrievability, stability])
			data.append([edge_count, vi, retrievability, stability])		
	#NAMING PROTOCOL: nodes_states_runs in folder of correct model
	filename = filepath + str(N) + "_" + str(num_stored_states) + "_" + str(runs_per_edge_count) +".csv"
	with open(filename, 'w') as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerows(data)
	csvFile.close()

def hopfield_match_coeff_sim(N, num_stored_states, num_edges, graph_model, 
	runs = 100, filepath = "data/", show_runs = True):
	"""Outputs a csv with 3 columns: in? (whether the edge is in the graph), match_coeff, edge_weight.
	The sim will run runs amount of times for a fixed graph_model and output a row for each edge and run."""
	match_coeff_data = [["in?", "match_coeff", "edge_weight"]]
	for _ in range(runs):
		hop_graph = graph_model(N, num_stored_states, num_edges)
		for i in range(N):
				for j in range(i):
					in_graph = 1 if hop_graph.adj_matrix[i][j] == 1 else 0
					match_coeff = match_coefficient(hop_graph, i, j)
					edge_weight = hop_graph.full_weights[i][j]
					if show_runs:
						print([in_graph, match_coeff, edge_weight])
					match_coeff_data.append([in_graph, match_coeff, edge_weight])
	
	filename = filepath + "random_edge_null_match_coeff_" + str(N) + "_" + str(num_stored_states) + "_" + str(runs) +".csv"
	with open(filename, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(match_coeff_data)
	csvFile.close()

#set of functions that can be passed in as graph_model to the above function
def random_edges_for_sim(N, num_stored_states, edge_count):
	g = random_edges(N, edge_count)
	return random_hopfield(N, num_stored_states, g)

def pruned_edges_for_sim(N, num_stored_states, edge_count):
	patterns = [random_state(N) for _ in range(num_stored_states)]
	return pruned_hopfield(patterns, edge_count)

