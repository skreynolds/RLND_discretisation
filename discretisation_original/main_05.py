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

# import the agent class
from agent import *

# import the monitoring function to run simulation
from monitoring_utils import *


if __name__ == '__main__':
	
	#################################################################
	# Required from main_02.py
	#################################################################

	env = gym.make('MountainCar-v0')
	env.seed(505);

	#################################################################
	# Required from main_04.py
	#################################################################

	state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))

	#################################################################
	# Q-Learning
	#################################################################

	# create agent
	q_agent = QLearningAgent(env, state_grid)
	
	# run simulation
	scores = run(q_agent, env)

	# Plot scores obtained per episode
	plt.plot(scores)
	plt.title("Scores")

	rolling_mean = plot_scores(scores)

	# run simulation in test mode
	test_scores = run(q_agent, env, num_episodes=100, mode='test')
	print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
	_ = plot_scores(test_scores, rolling_window=10)

	# plot q-table
	plot_q_table(q_agent.q_table)