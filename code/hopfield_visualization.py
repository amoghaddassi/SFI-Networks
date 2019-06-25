from graphs.HopfieldGraph import *
from graphs.Graph import *

from random_graphs import *
from hopfield_evaluation import *

import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.misc import comb

def plot_colored_network(hopfield_graph, state, labels = True, 
	metric = retrievability_performance_metric, save_plot = False, path = None):
	"""For a given hopfield_graph and a certain state (index of state), draws the topology
	of the graph where a node is green if it is 1 in the given state, or blue otherwise."""
	nx_graph = hopfield_graph.to_networkx()
	color_map = []
	#gets the color of each node for the given state
	for node in nx_graph:
		if hopfield_graph.stored_states[state][node] == 1:
			color_map.append('green')
		else:
			color_map.append('blue')
	#gets the title for the plot (N, num_stored_states, num_edges, perf)
	N = len(hopfield_graph.nodes)
	num_stored_states = len(hopfield_graph.stored_states)
	num_edges = hopfield_graph.num_edges()
	perf = hopfield_performance(hopfield_graph, metric = metric)
	plt.title(str(N) + " - " + str(num_stored_states) + " - " + str(num_edges) + " - " + str(perf))
	nx.draw(nx_graph, node_color = color_map, with_labels = labels)
	if save_plot:
		assert path != None, "Must specify a path if saving"
		plt.savefig(path)
	else:
		plt.show()
	plt.cla() #erases the figure 

def plot_many_sim(N, num_stored_states, graph_model, runs_per_edge = 10, 
	edge_iterator = None, filepath = "data/random_edges/topologies/"):
	"""For each edge in iterator, will make runs_per_edges hopfield_graphs using graph model (which 
	takes N, num_stored_states, and num_edges as args) and saves the topology of the graph, colored according
	to each of the stored states. Stores each run a folder of its own within filepath."""
	if not edge_iterator:
		edge_iterator = range(1, int(comb(N, 2) + 1))
	#folder for sim will be filepath/N_numstates_runs
	plots_path = filepath + str(N) + "_" + str(num_stored_states) + "_" + str(runs_per_edge)
	os.makedirs(plots_path)
	for edge in edge_iterator:
		#within plots path, makes a folder for each edge run
		edge_path = plots_path + "/" + str(edge)
		os.makedirs(edge_path)
		for i in range(runs_per_edge):
			print(str(edge) + ", " + str(i))
			#makes folder in edge_path for each graph
			run_path = edge_path + "/" + str(i)
			os.makedirs(run_path)
			hop_graph = graph_model(N, num_stored_states, edge)
			#saves fig of each stored state config
			for s in range(num_stored_states):
				save_path = run_path + "/" + str(s)
				plot_colored_network(hop_graph, s, save_plot = True, path = save_path)

def connected_component_VI(hopfield_graph):
	"""Computes the average variation of information across all stored states within 
	each connected component in hopfield_graph and returns a list of the values."""
	nx_graph = hopfield_graph.to_networkx()
	res = dict()
	for component in nx.connected_components(nx_graph):
		states = []
		for state in hopfield_graph.stored_states:
			#adds only the portion of state in the current connected component
			cc_state = [state[i] for i in range(len(state)) if i in component]
			states.append(cc_state)
		res[str(component)] = avg_variation_of_information(states)
	return res




