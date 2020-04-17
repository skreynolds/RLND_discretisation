#!/usr/bin/python3

'''
NOTE: ON LOCAL MACHINE EXECUTE THIS SCRIPT IN THE ROBOND ENVIRONMENT
'''

import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# import the discretisation utils library
from discretisation_utils import *


if __name__ == '__main__':
	#################################################################
	# Discretise the State Space with a Uniform Grid
	#################################################################
	
	
	# TEST 1: create_uniform_grid  
	low = [-1.0, -5.0]
	high = [1.0, 5.0]
	test = create_uniform_grid(low, high) #[test]
	
	print(test)

	
	# TEST 2: discretize
	# Test with a simple grid and some samples
	grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
	samples = np.array(
	[[-1.0 , -5.0],
	 [-0.81, -4.1],
	 [-0.8 , -4.0],
	 [-0.5 ,  0.0],
	 [ 0.2 , -1.9],
	 [ 0.8 ,  4.0],
	 [ 0.81,  4.1],
	 [ 1.0 ,  5.0]])
	discretized_samples = np.array([discretise(sample, grid) for sample in samples])
	print("\nSamples:", repr(samples), sep="\n")
	print("\nDiscretized samples:", repr(discretized_samples), sep="\n")