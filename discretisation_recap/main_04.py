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

# import the visualisation utils library
from visualisation_utils import *


if __name__ == '__main__':
	
	#################################################################
	# Required from main_02.py
	#################################################################

	env = gym.make('MountainCar-v0')
	env.seed(505);

	#################################################################
	# Required from main_03.py
	#################################################################	

	low = [-1.0, -5.0]
	high = [1.0, 5.0]

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

	#################################################################
	# Visualisation
	#################################################################

	visualize_samples(samples, discretized_samples, grid, low, high)

	# Create a grid to discretize the state space
	state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
	
	print(state_grid)

	# Obtain some samples from the space, discretize them, and then visualize them
	state_samples = np.array([env.observation_space.sample() for i in range(10)])
	discretized_state_samples = np.array([discretise(sample, state_grid) for sample in state_samples])
	visualize_samples(state_samples, discretized_state_samples, state_grid,
	                  env.observation_space.low, env.observation_space.high)
	plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space