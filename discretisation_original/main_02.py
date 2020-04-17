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

if __name__ == '__main__':
	#################################################################
	# Specify the Environment, and Explore the State Action Space
	#################################################################

	# Create an environment and set random seed
	env = gym.make('MountainCar-v0')
	env.seed(505);
	
	
	# This code will simply show a snippet of the agent
	state = env.reset()
	img = plt.imshow(env.render(mode='rgb_array'))

	for t in range(1000):
		action = env.action_space.sample()
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		state, reward, done, _ = env.step(action)
		if done:
			print('Score: ', t+1)
			break

	env.close()
	

	# Explore the state (observation space)
	print("State space:", env.observation_space, "\n")
	print("- low:", env.observation_space.low, "\n")
	print("- high:", env.observation_space.high, "\n")

	# Generate some samples from the state space
	print("State space samples:")
	print(np.array([env.observation_space.sample() for i in range(10)]), "\n")

	# Expore the action space
	print("Action space:", env.action_space, "\n")

	# Generate some samples from the action space
	print("Action space samples:")
	print(np.array([env.action_space.sample() for i in range(10)]))