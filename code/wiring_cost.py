import numpy as np
from sklearn.decomposition import PCA

from graphs.Graph import *
from graphs.HopfieldGraph import *

from bit_string_helpers import *

def wiring_cost_energy(hopfield_graph):
	"""Uses the following pipeline for computing the energy associated with a given Hopfield network:
	1. Gets the node bit strings for all nodes in the network.
	2. Places the nodes on verticies of an M dimensional simplex by flipping the bits if more than half are 1.
	3. Runs PCA on the simplex points to get two dimensional coordinates for each nodes.
	4. Uses these coordinates to compute the energy function of the strees majorization method."""
	def stress_energy():
		"""Uses the Kamada and Kawai energy function for force directed graph layouts to calculate the
		energy of the hopfield_graph with coordinates given by PCA."""
		energy = 0
		#gets some nx stuff for use in the for loop
		nx_graph = hopfield_graph.to_networkx()
		network_diameter = nx.diameter(nx_graph)
		for i in range(len(hopfield_graph.nodes)):
			for j in range(i):
				p_i, p_j = coords[i], coords[j] #gets the appropriate points
				#gets individual coordinates
				x_i, y_i = p_i[0], p_i[1]
				x_j, y_j = p_j[0], p_j[1]
				#computes the different parts of the energy term
				x_sq_diff = (x_i - x_j) ** 2
				y_sq_diff = (y_i - y_j) ** 2
				#gets the path len btw i and j. If there is no path, uses diameter of network instead.
				try:
					path_len = nx.shortest_path_length(nx_graph, i, j)
				except:
					path_len = network_diameter
				#computes the full energy term and adds to total energy
				energy_term = x_sq_diff + y_sq_diff + path_len ** 2 - 2*path_len*np.sqrt(x_sq_diff + y_sq_diff)
				energy += energy_term / 2
		return energy
	#places each node on the vertex of a simplex by getting the associated bit string and flipping if needed.
	node_strings = node_bit_strings(hopfield_graph)
	flipped_strings = flip_node_strings(node_strings)
	#runs PCA on the flipped_strings to get 2d coordinates.
	pca = PCA(n_components = 2)
	coords = pca.fit_transform(flipped_strings)
	#returns the energy of the graph with the layout given by the coords.
	return stress_energy()

def node_bit_strings(hopfield_graph):
	"""Returns a list of lists, where the ith inner list is the M length bit string
	associated with the firing pattern of the ith node."""
	res = [[] for _ in range(len(hopfield_graph.nodes))]
	for state in hopfield_graph.stored_states:
		for i in range(len(state)):
			res[i].append(state[i])
	return res

def flip_node_strings(node_strings):
	"""Given a list of lists of node bit strings, returns a list of lists with the same
	dimensions, where each inner list that has more than half 1's is flipped."""
	res = []
	for node_string in node_strings:
		if np.count_nonzero(node_string) > len(node_string) / 2:
			res.append(flip_porition(node_string, 1))
		else:
			res.append(node_string)
	return res