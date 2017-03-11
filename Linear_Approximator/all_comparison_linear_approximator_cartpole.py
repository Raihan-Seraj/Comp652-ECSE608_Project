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



#with the cartpole from openAi gym
env = gym.envs.make("CartPole-v0")


observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))



class Estimator():

	"""
	Class to define the value function - Linear Function Approximator in this case
	"""
	def __init__(self):
		self.models = []

		for _ in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)

	def featurize_state(self, state):
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]



	def predict(self, s, a=None):
		features = self.featurize_state(s)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]


	def predict_s_a(self, s, a=None):
		features = self.featurize_state_action(s, a)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]


	def update(self, s, a, target):

		#updates the estimator parameters for given s,a towards target y
		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [target])




def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Q Learning:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			#Q-value TD Target
			td_target = reward + discount_factor * np.max(q_values_next)

			#update the Q values
			#not this anymore
			#Q[state][action] += alpha * td_delta
			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats



def sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

	for i_episode in range(num_episodes):
		print "Episode Number, SARSA:", i_episode
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()
		next_action = None

		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		

		# if next_action is None:
		# 	action_probs = policy(state)
		# 	action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		# else:
		# 	action = next_action

		# action_probs = policy(state)
		# action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		for t in itertools.count():

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#update Q-values for the next state, next action
			q_values_next = estimator.predict(next_state)

			q_next_state_next_action = q_values_next[next_action] 

			td_target = reward + discount_factor * q_next_state_next_action

			estimator.update(state, action, td_target)

			if done:
				break

			state = next_state
			action = next_action

	return stats


def expected_sarsa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Expected SARSA:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			V = np.sum( next_action_probs * q_values_next)


			#Q-value TD Target
			td_target = reward + discount_factor * V

			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats






def two_step_tree_backup(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Two Step Tree Backup:", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			next_state, reward, done, _ = env.step(action)
			if done:
				break
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			V = np.sum( next_action_probs * q_values_next)


			next_next_state, next_reward, done, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			q_values_next_next = estimator.predict(next_next_state)

			next_V = np.sum(next_next_action_probs * q_values_next_next)

			q_next_next_state_next_next_action = q_values_next_next[next_next_action]

			Delta = next_reward + discount_factor * next_V - q_next_next_state_next_next_action

			next_action_selection_probability = np.max(next_action_probs)

			td_target = reward + discount_factor * V + discount_factor * next_action_selection_probability * Delta


			estimator.update(state, action, td_target)

			if done:
				break
			state = next_state

	return stats





"""
On Policy Q Sigma Algorithm with Linear Function Approximator
"""
def Q_Sigma_On_Policy(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Q(sigma):", i_episode
		#agent policy based on the greedy maximisation of Q
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()
		next_action = None

		#for each one step in the environment
		for t in itertools.count():
			if next_action is None:
				action_probs = policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action


			next_state, reward, done, _ = env.step(action)
			if done:
				break
			
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)

			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			sigma = random.randint(0,1)

			V = np.sum( next_action_probs * q_values_next)

			q_next_state_next_action = q_values_next[next_action]

			Sigma_Effect = sigma * q_next_state_next_action+ (1 - sigma) * V

			td_target = reward + discount_factor * Sigma_Effect

			#Q-value TD Target
			td_target = reward + discount_factor * V

			estimator.update(state, action, td_target)
			if done:
				break
			state = next_state
	return stats









def save_cum_rwd(stats, smoothing_window=1, noshow=False):


    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    cum_rwd = rewards_smoothed

    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/gym_examples/DQN_Experiments/'  + 'linear_approx_cart_pole_v0_cumulative_reward' + '.npy', cum_rwd)

    return cum_rwd




def plot_episode_stats(stats1, stats2, stats3, stats4, stats5, smoothing_window=200, noshow=False):

	#higher the smoothing window, the better the differences can be seen

    # Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="SARSA")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Q Learning")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Expected SARSA")
    cum_rwd_4, = plt.plot(rewards_smoothed_4, label="2-Step Tree Backup")
    cum_rwd_5, = plt.plot(rewards_smoothed_5, label="Q(sigma)")


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Algorithms with Linear Function Approximator")
    plt.show()


    return fig





def main():
	estimator = Estimator()
	num_episodes = 2000

	print "SARSA"
	stats_sarsa = sarsa(env, estimator, num_episodes, epsilon=0.1)
	
	print "Q Learning"
	stats_q_learning = q_learning(env, estimator, num_episodes, epsilon=0.1)

	print "Expected SARSA"
	stats_expected_sarsa = expected_sarsa(env, estimator, num_episodes, epsilon=0.1)

	print "Two Step Tree Backup"
	stats_2_tree_backup = two_step_tree_backup(env, estimator, num_episodes, epsilon=0.1)

	print "Q Sigma"
	stats_q_sigma = Q_Sigma_On_Policy(env, estimator, num_episodes, epsilon=0.1)


	plot_episode_stats(stats_sarsa, stats_q_learning, stats_expected_sarsa, stats_2_tree_backup, stats_q_sigma)

	env.close()

if __name__ == '__main__':
	main()






