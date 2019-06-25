import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Set of functions for dealing with performance simulations when we vary # of edges
def perf_summary_stats(df, group_by = "edges"):
	"""Given a data frame with all data from a performance simulation, returns
	2 dataframes (avg, std) of summary stats with one row for each edge count and
	1 column for each performance metric."""
	gb = df.groupby(group_by)
	avg = gb.agg(np.average)
	std = gb.agg(np.std)
	return avg, std

def fix_sw_edges(df, N):
	"""Given data from a small world simulation (N nodes), adjusts the edge count
	from the degree of each node to total number of edges."""
	df["edges"] = df["edges"] / 2 * N

def fixed_edged_perf_dist(df, edges):
	"""Given a dataframe with perf stats, returns a df of performance data for each
	trial with edges # of edges."""
	gb = df.groupby("edges")
	return gb.get_group(edges)