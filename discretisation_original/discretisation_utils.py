#!/usr/bin/python3

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

	discretised_array = []

	for i in range(2):
		discretised_array.append(np.linspace(low[i], high[i], bins[i]+1)[1:-1])

	for l, h, b, splits in zip(low, high, bins, discretised_array):
		print("[{}, {}] / {} => {}".format(l, h, b, splits))

	return discretised_array


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
	return [int(np.digitize(s,g)) for s, g in zip(sample, grid)]