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



def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
	"""Visualize original and discretized samples on a given 2-dimensional grid."""

	fig, ax = plt.subplots(figsize=(10, 10))

	# Show grid
	ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
	ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
	ax.grid(True)

	# If bounds (low, high) are specified, use them to set axis limits
	if low is not None and high is not None:
	    ax.set_xlim(low[0], high[0])
	    ax.set_ylim(low[1], high[1])
	else:
	    # Otherwise use first, last grid locations as low, high (for further mapping discretized samples)
	    low = [splits[0] for splits in grid]
	    high = [splits[-1] for splits in grid]

	# Map each discretized sample (which is really an index) to the center of corresponding grid cell
	grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T))  # add low and high ends
	grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2  # compute center of each grid cell
	locs = np.stack(grid_centers[i, discretized_samples[:, i]] for i in range(len(grid))).T  # map discretized samples

	ax.plot(samples[:, 0], samples[:, 1], 'o')  # plot original samples
	ax.plot(locs[:, 0], locs[:, 1], 's')  # plot discretized samples in mapped locations
	ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))  # add a line connecting each original-discretized sample
	ax.legend(['original', 'discretized'])

	plt.show()




class QLearningAgent:
	"""Q-Learning agent that can act on a continuous state space by discretizing it."""

	def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
				 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
		"""Initialize variables, create grid for discretization."""
		# Environment info
		self.env = env
		self.state_grid = state_grid
		self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
		self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
		self.seed = np.random.seed(seed)
		print("Environment:", self.env)
		print("State space size:", self.state_size)
		print("Action space size:", self.action_size)

		# Learning parameters
		self.alpha = alpha  # learning rate
		self.gamma = gamma  # discount factor
		self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
		self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
		self.min_epsilon = min_epsilon

		# Create Q-table
		self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
		print("Q table size:", self.q_table.shape)

	def preprocess_state(self, state):
		"""Map a continuous state to its discretized representation."""
		# TODO: Implement this
		pass

	def reset_episode(self, state):
		"""Reset variables for a new episode."""
		# Gradually decrease exploration rate
		self.epsilon *= self.epsilon_decay_rate
		self.epsilon = max(self.epsilon, self.min_epsilon)

		# Decide initial action
		self.last_state = self.preprocess_state(state)
		self.last_action = np.argmax(self.q_table[self.last_state])
		return self.last_action
    
	def reset_exploration(self, epsilon=None):
		"""Reset exploration rate used when training."""
		self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

	def act(self, state, reward=None, done=None, mode='train'):
		"""Pick next action and update internal Q table (when mode != 'test')."""
		state = self.preprocess_state(state)
		if mode == 'test':
			# Test mode: Simply produce an action
			action = np.argmax(self.q_table[state])
		else:
			# Train mode (default): Update Q table, pick next action
			# Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
			self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
				(reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

			# Exploration vs. exploitation
			do_exploration = np.random.uniform(0, 1) < self.epsilon
			if do_exploration:
				# Pick a random action
				action = np.random.randint(0, self.action_size)
			else:
				# Pick the best action from Q table
				action = np.argmax(self.q_table[state])

		# Roll over current state, action for next step
		self.last_state = state
		self.last_action = action
		return action



if __name__ == '__main__':
	
	
	#################################################################
	# Specify the Environment, and Explore the State Action Space
	#################################################################

	# Create an environment and set random seed
	env = gym.make('MountainCar-v0')
	env.seed(505);
	
	'''
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
	'''

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
	

	#################################################################
	# Discretise the State Space with a Uniform Grid
	#################################################################
	
	low = [-1.0, -5.0]
	high = [1.0, 5.0]
	create_uniform_grid(low, high) #[test]


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


	#################################################################
	# Visualisation
	#################################################################

	visualize_samples(samples, discretized_samples, grid, low, high)

	# Create a grid to discretize the state space
	state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))
	
	state_grid

	# Obtain some samples from the space, discretize them, and then visualize them
	state_samples = np.array([env.observation_space.sample() for i in range(10)])
	discretized_state_samples = np.array([discretise(sample, state_grid) for sample in state_samples])
	visualize_samples(state_samples, discretized_state_samples, state_grid,
	                  env.observation_space.low, env.observation_space.high)
	plt.xlabel('position'); plt.ylabel('velocity');  # axis labels for MountainCar-v0 state space

	#################################################################
	# Q-Learning
	#################################################################

