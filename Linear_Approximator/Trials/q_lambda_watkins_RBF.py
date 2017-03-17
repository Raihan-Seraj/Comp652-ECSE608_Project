import sys
sys.path.insert(0, "/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/")
import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import pandas as pd
import sys
import random

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from collections import namedtuple

import matplotlib.pyplot as plt

import pyrl.basis.fourier as fourier
import pyrl.basis.rbf as rbf
import pyrl.basis.tilecode as tilecode




"""
Environment
"""

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()

# env = gym.envs.make("MountainCar-v0")




"""
Feature Extactor
"""
observation_examples = np.array([env.observation_space.sample() for x in range(1)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
# featurizer.fit(scaler.transform(observation_examples))

featurizer.fit(observation_examples)


def featurize_state(state):
	# state = np.array([state])
	# scaled = scaler.transform([state])
	# featurized = featurizer.transform(scaled)
	featurized = featurizer.transform(state)
	return featurized[0]




def behaviour_policy_epsilon_greedy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def create_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn



from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample


"""
Agent policies
"""

def epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_lambda_watkins(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.5, lambda_param=1):

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	for i_episode in range(num_episodes):

		print "Episode Number, Watkins Q(lambda):", i_episode

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		state = env.reset()

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		

		#initialising eligibility traces
		eligibility = np.zeros(shape=(theta.shape[0],env.action_space.n))


		for t in itertools.count():


			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)
			next_q_values_state_action = next_q_values[next_action]

			best_action = np.argmax(next_q_values)

			max_q_value = next_q_values[best_action]

			td_target = reward + discount_factor * max_q_value

			Delta = td_target - q_values_state_action

			# if next_action == best_action:
			# 	eligibility[:, action] = discount_factor * lambda_param * eligibility[:, action] + features_state			
			# else:
			# 	eligibility[:, :] = 0 

			# eligibility[:, action] = eligibility[:, action] * discount_factor * lambda_param
			eligibility[:, action] = eligibility[:, action] + features_state


			# eligibility[:, action] = eligibility[:, action] + features_state

			theta[:, action] += alpha * Delta * eligibility[:, action]

			theta = theta / theta.sum(axis=1)[:, np.newaxis]
			
			# print "Theta", theta

			if next_action == best_action:
				eligibility[:, :] += discount_factor * lambda_param * eligibility
			else:
				eligibility[:, :] += 0 

			if done:
				break			

			state = next_state
			action = next_action

		print "Theta End of Episode", theta

		print "Eligibility_Traces End of Episide", eligibility
		

	return stats







def main():

	print "Watkins Q(lambda)"

	theta = np.zeros(shape=(400, env.action_space.n))
	
	num_episodes = 1000
	smoothing_window = 1

	stats_sarsa_q_lambda = q_lambda_watkins(env, theta, num_episodes)
	rewards_smoothed_stats_q_lambda = pd.Series(stats_sarsa_q_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_q_lambda
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'q_lambda_watkins_rbf' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_sarsa_q_lambda)
	env.close()



	
if __name__ == '__main__':
	main()



