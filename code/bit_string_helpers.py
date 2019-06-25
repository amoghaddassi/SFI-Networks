import random
import numpy as np

def bit_list_to_string(lst):
	"""Converts a list of bits to a string."""
	n = ""
	for b in lst:
		n += str(b)
	return n

def flip_porition(state, p):
	"""Flips p portion of bits in state."""
	num_to_flip = int(p * len(state)) #rounds up how many to flip
	pos_to_flip = random.sample(range(len(state)), num_to_flip) #chooses which indicies to flip
	#flips the bits
	res = state.copy()
	for i in pos_to_flip:
		res[i] = 1 - res[i]
	return res

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

def avg_variation_of_information(states):
	"""For a given set of states, computes the pairwise variation of information
	for all states and returns the average."""
	tot = 0
	count = 0
	for i in range(len(states)):
		for j in range(i):
			tot += variation_of_information(states[i], states[j])
			count += 1
	return tot / count 

def all_bit_strings(N, arr = ["0", "1"]):
	"""Appends all bit strings of length N to arr using a simple recursive alg."""
	if N == 1:
		return arr
	return all_bit_strings(N - 1, [i + "0" for i in arr] + [i + "1" for i in arr])

def flip_bits(bit_string):
	"""Given a bit string, returns the string with all bits flipped."""
	res = []
	for b in bit_string:
		flipped = 1 - int(b)
		res.append(str(flipped))
	return "".join(res)
