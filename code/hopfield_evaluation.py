import numpy as np

from graphs.Graph import *
from graphs.HopfieldGraph import *
from graphs.KuramotoGraph import *

from random_graphs import *
from dynamics import *

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
		normed = [x/sum(counts) for x in counts]
		info = 0
		for p in normed:
			if p <= .0001: continue #avoids div by 0 errors
			info -= p * np.log2(p)
		return info	

	ha = bit_entropy(A)
	hb = bit_entropy(B)
	hab = joint_bit_entropy(A, B)
	return 2*hab - ha - hb

def hopfield_performance(hopfield_graph, runs = 100):
	"""Define the performance of a hopfield network on a individual run as the min
	variation of information between a steady state (found by running the network from a random IC)
	and a stored state in the network. Total performance is the average of all runs."""
	def bit_list_to_string(lst):
		"""Converts a list of bits to a string."""
		n = 0
		for b in lst:
			n += b
			n *= 10
		return str(n // 10)
	
	total_perf = 0 #sum of all the variation of information
	for _ in range(runs):
		hopfield_graph.random_config()
		steady_state = fixed_point(hopfield_graph, hopfield_graph.dynamic)
		if not steady_state: continue
		steady_state_str = bit_list_to_string(steady_state)
		#TODO: make sure that this min is a fair way to consider perf on a run
		total_perf += min([variation_of_information(steady_state_str, stored_state)
			for stored_state in hopfield_graph.stored_states])
		
	return total_perf / runs
