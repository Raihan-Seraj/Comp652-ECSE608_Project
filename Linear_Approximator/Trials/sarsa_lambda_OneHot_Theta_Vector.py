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
# observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(observation_examples)

observation_examples = np.array([env.observation_space.sample() for x in range(1)])


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])


featurizer.fit(observation_examples)

feature_length = 1

total_states = env.observation_space.n * env.action_space.n * feature_length


def behaviour_policy_epsilon_greedy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        features_state = featurize_state(observation)
        q_values = np.dot(theta.T, features_state)

        best_action = np.argmax(q_values)

        A[best_action] += (1.0 - epsilon)
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
        features_state = featurize_state(observation)
        q_values = np.dot(theta.T, features_state)

        best_action = np.argmax(q_values)
        
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def one_hot_encoding(state, action):
    
    feature_vector=np.zeros([env.observation_space.n*env.action_space.n,1])  #creating (192 by 1) feature vector
    all_feature_vector = np.zeros([feature_vector.shape[0] * feature_length, 1])

    if action==0: 
       feature_vector[state]=1
    
    if action==1:
        feature_vector[state+1*env.observation_space.n]=1
    
    if action==2:
        feature_vector[state+2*env.observation_space.n]=1
    
    if action==3:
        feature_vector[state+3*env.observation_space.n]=1
    
    # all_feature_vector = np.concatenate((feature_vector, feature_vector, feature_vector, feature_vector, feature_vector), axis=0)

    all_feature_vector = feature_vector

    return all_feature_vector
 

def featurize_state(state):
	nA = env.action_space.n
	nS = env.observation_space.n
	feature_vector=np.zeros([nS*nA*feature_length,1])

	all_features = np.zeros(shape=(feature_vector.shape[0]))
	all_features = np.array([all_features]).T

	for a in range(nA):
		feature_vector = one_hot_encoding(state, a)
		all_features = np.append(all_features, feature_vector, axis=1)

	all_features = all_features[:, 1:]

	return all_features




# def featurize_state(state):
# 	nA = env.action_space.n

# 	for a in range(nA):
# 		features = one_hot_encoding(state, a)


#     state_features_a1 =one_hot_encoding(state,action=0)
#      state_features_a2=one_hot_encoding(state,action=1)
#             state_features_a3=one_hot_encoding(state,action=2)
#             state_features_a4=one_hot_encoding(state,action=3)
#             state_features=np.concatenate((state_features_a1,state_features_a2,state_features_a3,state_features_a4),axis=1)
# return state_features



def sarsa_lambda(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.5, lambda_param=1):

	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	for i_episode in range(num_episodes):

		print "Episode Number, SARSA(lambda):", i_episode

		policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
		
		state = env.reset()
		# next_action = None

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		

		eligibility = np.zeros(shape=(theta.shape[0]))



		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_state = featurize_state(state)
			q_values = np.dot(theta.T, features_state)
			q_values_state_action = q_values[action]

			features_state_action = features_state[:, action]

			eligibility = eligibility + features_state_action

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			next_features_state = featurize_state(next_state)
			next_q_values = np.dot(theta.T, next_features_state)			
			next_q_values_state_action = next_q_values[next_action]

			td_target = np.add(reward, discount_factor * next_q_values_state_action)

			Delta = td_target - q_values_state_action

			theta += np.multiply(alpha * Delta, eligibility)

			eligibility = eligibility * discount_factor * lambda_param



			if done:
				break			


			state = next_state
			action = next_action

	return stats







def main():

	print "SARSA(lambda)"

	theta = np.zeros(shape=(total_states))

	num_episodes = 5000
	smoothing_window = 1

	stats_sarsa_lambda = sarsa_lambda(env, theta, num_episodes)
	rewards_smoothed_stats_sarsa_lambda = pd.Series(stats_sarsa_lambda.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_sarsa_lambda
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/Eligibility_Traces/Accumulating_Traces/Cliff_Walking_Results/'  + 'sarsa_lambda_rbf' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_sarsa_lambda)
	env.close()



	
if __name__ == '__main__':
	main()



