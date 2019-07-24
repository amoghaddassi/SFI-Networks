from hopfield_mcmc import *
from hopfield_evaluation import *
from hopfield_models import *
from wiring_cost import *

states = [random_state(300) for _ in range(5)]
perf = []
cost = []
for alpha in range(0, 100, 5):
	for _ in range(5):
		print()
	print(alpha)
	realAlpha = alpha / 100
	mcmc = metropolis_hastings(states, 3000, max_iter = 1000, alpha = realAlpha, moves_per_iter = 3)
	perf.append(hopfield_performance(mcmc, retrievability_performance_metric, runs = 10))
	cost.append(wiring_cost_energy(mcmc))