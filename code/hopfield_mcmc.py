"""Uses Metropolis-Hastings to find an optimal graph given a certain set of states and # of edges."""
from graphs.Graph import *
from graphs.HopfieldGraph import *

from random_graphs import *
from hopfield_models import *
from hopfield_evaluation import *

import numpy as np
import random

def metropolis_hastings(patterns, num_edges, fp_threshold = 20, 
	max_iter = 1000, rewires_per_iter = 1):
	"""The main runner function of the file. Returns the result of running MCMC to find
	the optiminal graph topology given some set of patterns and number of edges."""
	def random_edge(graph):
		"""Given a graph, returns a random edge (i, j) that is in the graph."""
		i = random.sample(range(len(graph.nodes)), 1)[0]
		j = random.sample(range(len(graph.nodes)), 1)[0]

		if graph.adj_matrix[i][j] == 0:
			#means we found a null edge, so we repeat
			return random_edge(graph)
		return i, j
	
	def accept(curr, prop, beta = .3):
		"""Given an acceptance ratio, returns true if we should accept."""
		if prop > curr:
			#always accept when the proposal is better than current
			return True
		p = np.exp(-1 * beta * (prop / curr))
		return np.random.binomial(1, p) == 1
	
	#get initial random state
	graph = random_edges(len(patterns[0]), num_edges)
	hop_graph = HopfieldGraph(graph, patterns) #the running variable for the current optima
	hop_graph.train()
	curr_score = hopfield_performance(hop_graph, metric = retrievability_performance_metric, nonneg = True)
	old_score = curr_score #for fixed point checking
	
	while curr_score < .1:
		#makes sure we start at a reasonable state
		graph = random_edges(len(patterns[0]), num_edges)
		hop_graph = HopfieldGraph(graph, patterns) #the running variable for the current optima
		hop_graph.train()
		curr_score = hopfield_performance(hop_graph, metric = retrievability_performance_metric, nonneg = True)
		print(curr_score)
	
	best_graph, best_score = hop_graph, curr_score #track the running best
	run_count, fp_count, away_from_best_count = 0, 0, 0 #count state vars
	
	while run_count < max_iter:
		#generate a proposal by randomly swapping an edge in hop_graph
		prop = hop_graph.copy()
		for _ in range(rewires_per_iter):
			i, j = random_edge(hop_graph)
			rewire(i, j, prop)
		#moves to prop according to acceptance function
		prop_score = hopfield_performance(prop, metric = retrievability_performance_metric, nonneg = True)
		if accept(curr_score, prop_score) or fp_count >= fp_threshold:
			hop_graph = prop
			curr_score = prop_score
			fp_count = 0
		else:
			fp_count += 1

		#updates best vars:
		if curr_score > best_score:
			best_score = curr_score
			best_graph = hop_graph
			away_from_best_count = 0
		else:
			away_from_best_count += 1

		print(str(run_count) + ", " + str(curr_score))
		run_count += 1
	return best_graph
