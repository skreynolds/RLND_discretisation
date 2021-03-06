# import required libraries
import numpy as np

#################################################
# utility functions for discretisation
#################################################

def create_uniform_grid(low, high, bins=(10,10)):
	'''
	Define a uniformly-spaced grid that can be used to discretize a space.

	Parameters
	----------
	low : array_like
	    Lower bounds for each dimension of the continuous space.
	high : array_like
	    Upper bounds for each dimension of the continuous space.
	bins : tuple
	    Number of bins along each corresponding dimension.

	Returns
	-------
	grid : list of array_like
	    A list of arrays containing split points for each dimension.
	'''

	# TODO: Implement this
	return [np.linspace(low[i], high[i], bins[i] + 1) for i in range(len(bins))]


def discretise(sample, grid):
	"""Discretize a sample as per given grid.

	Parameters
	----------
	sample : array_like
	    A single sample from the (original) continuous space.
	grid : list of array_like
	    A list of arrays containing split points for each dimension.

	Returns
	-------
	discretized_sample : array_like
	    A sequence of integers with the same number of dimensions as sample.
	"""
	# TODO: Implement this
	
	digitized_values = []

	for s, g in zip(sample, grid):
		digitized_values.append(np.digitize(s,g))

	return digitized_values