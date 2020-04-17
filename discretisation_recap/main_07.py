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
	# Required from main_06.py
	#################################################################

	# TODO: Create a new agent with a different state space grid
	state_grid_new = create_uniform_grid(?, ?, bins=(?, ?))
	q_agent_new = QLearningAgent(env, state_grid_new)
	q_agent_new.scores = []  # initialize a list to store scores for this agent

	# Train it over a desired number of episodes and analyze scores
	# Note: This cell can be run multiple times, and scores will get accumulated
	q_agent_new.scores += run(q_agent_new, env, num_episodes=50000)  # accumulate scores
	rolling_mean_new = plot_scores(q_agent_new.scores)

	# Run in test mode and analyze scores obtained
	test_scores = run(q_agent_new, env, num_episodes=100, mode='test')
	print("[TEST] Completed {} episodes with avg. score = {}".format(len(test_scores), np.mean(test_scores)))
	_ = plot_scores(test_scores)

	# Visualize the learned Q-table
	plot_q_table(q_agent_new.q_table)

	#################################################################
	# Smart agent action rendering
	#################################################################

	state = env.reset()
	score = 0
	img = plt.imshow(env.render(mode='rgb_array'))
	for t in range(1000):
    	action = q_agent_new.act(state, mode='test')
    	img.set_data(env.render(mode='rgb_array')) 
    	plt.axis('off')
    	state, reward, done, _ = env.step(action)
    	score += reward
    	if done:
        	print('Score: ', score)
        	break
        
	env.close()