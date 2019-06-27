import numpy as np
from scipy.misc import comb
from scipy.stats import entropy
import csv
import random
import time

from graphs.Graph import *
from graphs.HopfieldGraph import *
from graphs.KuramotoGraph import *

from random_graphs import *
from dynamics import *
from bit_string_helpers import *
from hopfield_models import *
from hopfield_evaluation import *

def hopfield_perf_sim(N, num_stored_states, graph_model,runs_per_edge_count = 100,
	edge_iterator=None, filepath = "data/random_edges/vary_edges/", show_runs = True,
	metrics = ["variation_of_information", "runtime"]):
	"""Outputs a csv with 2 columns: edge count, performance. Edge count refers
	to the number of edges we place randomly in an N node graph built using graph_model. 
	Performance refersto the constructed graph's performance when it is trained on 
	num_stored_states randomly generated stored states. graph_model is a function that returns
	a hopfield graph with parameters N, num_stored_states, and edge_count. edge_iterator is how we
	vary the edge counts during the simulation."""
	if not edge_iterator:
		edge_iterator = range(1, int(comb(N, 2) + 1))
	data = [["edges"] + metrics]
	for edge_count in edge_iterator:
		for _ in range(runs_per_edge_count):
			hop_graph = graph_model(N, num_stored_states, edge_count)
			stats = [edge_count]
			#computes all the metrics in metrics and adds to data
			if "variation_of_information" in metrics:
				vi = hopfield_performance(hop_graph, metric = vi_performance_metric)
				stats.append(vi)
			if "retrievability" in metrics:
				retrievability = hopfield_performance(hop_graph, metric = retrievability_performance_metric)
				stats.append(retrievability)
			if "stability" in metrics:
				stability = hopfield_performance(hop_graph, metric = stability_performance_metric)
				stats.append(stability)
			if "runtime" in metrics:
				rt = runtime(hop_graph)
				stats.append(rt)
			if show_runs:
				print(stats)
			data.append(stats)		
	#NAMING PROTOCOL: nodes_states_runs in folder of correct model
	filename = filepath + str(N) + "_" + str(num_stored_states) + "_" + str(runs_per_edge_count) +".csv"
	with open(filename, 'w') as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerows(data)
	csvFile.close()

def hopfield_vary_patterns_sim(N, num_edges, graph_model, runs_per_states = 100, num_states_iterator = None,
	filepath = "data/random_edges/vary_patterns/", show_runs = True):
	"""For each number of patterns in pattern dist, generates a random pattern dist and records the entropy (
	coarse measure of how uniform the group sizes are). For each data point, records all metrics."""
	data = [["num_patterns", "entropy_of_states", "retrievability"]]
	for num_states in num_states_iterator:
		for _ in range(runs_per_states):
			pattern_dist = random_dist(num_states)
			uniformness = entropy(pattern_dist)
			patterns = states_from_dist(pattern_dist, N)
			graph = graph_model(patterns, num_edges)
			perf = hopfield_performance(graph, metric = retrievability_performance_metric)
			stats = [num_states, uniformness, perf]
			if show_runs:
				print(stats)
			data.append(stats)
	filename = filepath + str(N) + "_" + str(num_edges) + "_" + str(runs_per_states) + ".csv"
	with open(filename, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(data)
	csvFile.close()

def hopfield_rewiring_sim(graph, runs_per_beta = 10, beta_iterator = None, 
	filepath = "data/pruned_edges/rewire/", metrics = ["retrievability"], show_runs = True):
	if not beta_iterator:
		beta_iterator = range(0, 250, 5)
	data = [["Rewire Probability"] + metrics]
	for beta in beta_iterator:
		for _ in range(runs_per_beta):
			hop_graph = graph.copy() #makes a copy since rewiring is destructive
			rewired_hopfield(hop_graph, beta / 1000)
			stats = [beta]
			#computes all the metrics in metrics and adds to data
			if "variation_of_information" in metrics:
				vi = hopfield_performance(hop_graph, metric = vi_performance_metric)
				stats.append(vi)
			if "retrievability" in metrics:
				retrievability = hopfield_performance(hop_graph, metric = retrievability_performance_metric)
				stats.append(retrievability)
			if "stability" in metrics:
				stability = hopfield_performance(hop_graph, metric = stability_performance_metric)
				stats.append(stability)
			if "runtime" in metrics:
				rt = runtime(hop_graph)
				stats.append(rt)
			if show_runs:
				print(stats)
			data.append(stats)

	#NAMING PROTOCOL: nodes_states_edges_runs in folder of correct model
	filename = filepath + str(len(graph.nodes)) + "_" + str(len(graph.stored_states)) + "_" + str(graph.num_edges()) + "_" + str(runs_per_beta) +".csv"
	with open(filename, 'w') as csvFile:
	    writer = csv.writer(csvFile)
	    writer.writerows(data)
	csvFile.close()

def hopfield_edges_states_sim(N, num_stored_states, graph_model, runs = 1000,
	filepath = "data/pruned_edges/vary_edges_states/", show_runs = True, 
	metrics = ["variation_of_information"]):
	"""For each run, randomly samples an edge count and gets perf data as we vary the relatedness
	of states from the related_states_sim. graph_model is the same as for that function."""
	data = [["edges", "Mutual Information of States"] + metrics]
	for _ in range(runs):
		edges = random.sample(range(1, int(comb(N, 2) + 1)), 1)[0]
		run_data = hopfield_related_states_sim(N, num_stored_states, graph_model, edges, runs_per_p = 10,
											metrics = metrics, save_data = False, show_runs = show_runs)
		#adds the edge value to everything in run_data
		run_data_with_edges = [[edges] + d for d in run_data]
		data.extend(run_data_with_edges[1:])
	filename = filepath + str(N) + "_" + str(num_stored_states) + "_" + str(runs) +".csv"
	with open(filename, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(data)
	csvFile.close()

def hopfield_related_states_sim(N, num_stored_states, graph_model, num_edges,runs_per_p = 100,
	p_iterator = range(0, 51, 5), filepath = "data/random_edges/vary_states/", show_runs = True,
	metrics = ["variation_of_information"], save_data = True):
	"""For a graph model with a fixed number of nodes and edges, computes each metric runs_per_p times
	for each p in p_iterator. p will be the p parameter passed into related states when we generate our patterns
	for storage. graph_model takes a set of patterns and number of edges and returns a trained hop graph"""
	data = [["Mutual Information of States"] + metrics]
	for p in p_iterator:
		for _ in range(runs_per_p):
			patterns = related_states(N, num_stored_states, p / 100) # div by 100 since iterate over percentages
			hop_graph = graph_model(patterns, num_edges)
			stats = [avg_variation_of_information(patterns)]
			#computes all the metrics in metrics and adds to data
			if "variation_of_information" in metrics:
				vi = hopfield_performance(hop_graph, metric = vi_performance_metric)
				stats.append(vi)
			if "retrievability" in metrics:
				retrievability = hopfield_performance(hop_graph, metric = retrievability_performance_metric)
				stats.append(retrievability)
			if "stability" in metrics:
				stability = hopfield_performance(hop_graph, metric = stability_performance_metric)
				stats.append(stability)
			if "runtime" in metrics:
				rt = runtime(hop_graph)
				stats.append(rt)
			if show_runs:
				print(stats)
			data.append(stats)
	#NAMING PROTOCOL: nodes_states_runs in folder of correct model
	if save_data:
		filename = filepath + str(N) + "_" + str(num_stored_states) + "_" + str(num_edges) + "_" + str(runs_per_p) +".csv"
		with open(filename, 'w') as csvFile:
		    writer = csv.writer(csvFile)
		    writer.writerows(data)
		csvFile.close()
	else: return data

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