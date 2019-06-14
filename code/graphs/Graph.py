class Graph:
	
	class Node:
		def __init__(self, val, in_edges):
			"""val: node's value, 1 - on, 0 - off
			in_edges: list of node objects going into this node."""
			self.val = val
			self.in_edges = in_edges
	
	def __init__(self, nodes):
		"""nodes is a list of node objects."""
		self.nodes = nodes

	def graph_of(node_list, edge_list):
		"""Returns a graph with nodes that have values from node_list.
		edge_list specifies edges as a list of lists, where the ith
		entry contains the nodes going into the ith node. Nodes in the lists
		are specified by the index in node_list."""
		nodes = []
		#creates all node objects with no connections
		for val in node_list:
			nodes.append(Graph.Node(val, None))

		#specifies the connections
		for i in range(len(nodes)):
			in_edges = edge_list[i]
			in_nodes = []
			for edge in in_edges:
				in_nodes.append(nodes[edge])
			nodes[i].in_edges = in_nodes

		return Graph(nodes)

	def state(self):
		"""Returns the current state of the graph as a list of node states."""
		return [node.val for node in self.nodes]

	def __repr__(self):
		return self.state().__str__()




