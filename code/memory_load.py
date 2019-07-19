from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from hopfield_models import *
from hopfield_evaluation import *
from random_graphs import *

from graphs.HopfieldGraph import *
from graphs.Graph import *

def states_vs_perf(p, N):
	"""Same idea as alpha_vs_perf, but is in terms of p instead of alpha."""
	alpha = p / N
	return alpha_vs_perf(alpha, N)

def alpha_vs_perf(alpha, N):
	"""Uses the derivation for the probability of stability given a memory load
	via the random walk noise term. Returns the standard normal cdf evaluated at
	-1 / sqrt(alpha) times N. The cdf term is from the probability that the signs of 
	the local field of node i in pattern 1 and the value of node i in pattern 1 are the
	same. Multiply by N to see if the configuration of all nodes is the same."""
	return N * np.sqrt(alpha / (2 * np.pi)) * np.exp(-1 / (2 * alpha))

def critical_alpha(N):
	"""Returns 1 / 2lnN, the max memory load for a given number of nodes."""
	return 1 / (2 * np.log(N))

def critical_p(N):
	"""Returns N * critical_alpha(N), since that is the number of memories
	we can hold given N."""
	return N * critical_alpha(N)

def expected_errors_to_stability(Nerr, N):
	"""Given an expected number of unstable nodes in a steady state and the size of the
	network, returns the stability of such a state as a percent of total nodes."""
	return 1 - Nerr / N

def alpha_vs_perf_plot(N, alpha_iterator = np.arange(0, 1, .01)):
	"""For a fixed number of nodes, varies alpha according to alpha_iterator to 
	produce the alpha vs. perf plot. Also plots a marker for the critical alpha."""
	perf = []
	for alpha in alpha_iterator:
		perf.append(expected_errors_to_stability(alpha_vs_perf(alpha, N), N))
	plt.plot(alpha_iterator, perf) #performance plot
	plt.plot(critical_alpha(N), 1, color = "red", marker = "^") #critical alpha marker
	plt.plot(.138, 1, color = "green", marker = "^") #.138 marker
	plt.xlabel("Memory load")
	plt.ylabel("stability")
	plt.show()

def alpha_vs_perf_sim(N, num_edges, model = "random"):
	"""Draws the same plot as above, but now does so using real fully connected networks for a
	given N. Varies the number of stored states from 1 to N to vary alpha, and collects stability
	measure for each new pattern added."""
	alphas, perfs = [], [] #metrics to track for plot
	states = [random_state(N)] #running list of stored states
	alpha = 0
	while len(states) <= N / 3 and alpha <= 2:
		print(len(states))
		alpha = len(states) / (2 * num_edges / N) #computes appropriate alpha for this run
		if model == "pruned":
			hop_graph = pruned_edges_for_p_sim(states, num_edges)
		else:
			hop_graph = random_edges_for_p_sim(states, num_edges)
		alphas.append(alpha)
		perf = hopfield_performance(hop_graph, stability_metric_for_memory_load, runs = 5)
		perfs.append(perf)
		print(str(alpha) + ", " + str(perf))
		states.append(random_state(N))
	#plotting
	plt.plot(alphas, perfs) #performance plot
	plt.plot(critical_alpha(N), 1, color = "red", marker = "^") #critical alpha marker
	plt.plot(.138, 1, color = "green", marker = "^") #.138 marker
	plt.xlabel("Memory load")
	plt.ylabel("stability")
	plt.show()
	return alphas, perfs
