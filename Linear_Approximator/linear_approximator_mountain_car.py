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




#with the cartpole from openAi gym
env = gym.envs.make("MountainCar-v0")


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



"""
Behaviour policy for off-policy algorithms
"""
def behaviour_policy_epsilon_greedy(estimator, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon/nA
        q_values = estimator.predict(observation)       
        best_action = np.argmax(q_values)
        A[best_action] += ( 1.0 - epsilon)
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

			q_values = estimator.predict(state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)
			q_values_next_state_next_action = q_values_next[next_action]

			V = np.sum( next_action_probs * q_values_next)

			Delta = reward + discount_factor * V - q_values_state_action


			next_next_state, next_reward, done, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			q_values_next_next = estimator.predict(next_next_state)
			q_values_next_next_state_next_next_action = q_values_next_next[next_next_action]

			next_V = np.sum(next_next_action_probs * q_values_next_next)

			Delta_t_1 = next_reward + discount_factor * next_V - q_values_next_state_next_action

			next_action_selection_probability = np.max(next_action_probs)
			td_target = q_values_state_action + Delta + discount_factor * next_action_selection_probability * Delta_t_1

			estimator.update(state, action, td_target)

			if done:
				break
			state = next_state

	return stats





def three_step_tree_backup(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print "Episode Number, Three Step Tree Backup:", i_episode
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

			q_values = estimator.predict(state)
			q_values_state_action = q_values[action]


			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
			

			#update Q-values for the next state
			q_values_next = estimator.predict(next_state)
			q_values_next_state_next_action = q_values_next[next_action]

			V = np.sum( next_action_probs * q_values_next)

			Delta = reward + discount_factor * V - q_values_state_action


			next_next_state, next_reward, done, _ = env.step(next_action)
			next_next_action_probs = policy(next_next_state)
			next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

			q_values_next_next = estimator.predict(next_next_state)
			q_values_next_next_state_next_next_action = q_values_next_next[next_next_action]

			next_V = np.sum(next_next_action_probs * q_values_next_next)

			Delta_t_1 = next_reward + discount_factor * next_V - q_values_next_state_next_action


			next_next_next_state, next_next_reward, done, _ = env.step(next_next_action)
			next_next_next_action_probs = policy(next_next_next_state)
			next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

			q_values_next_next_next = estimator.predict(next_next_next_state)
			q_values_next_next_next_state_next_next_next_action = q_values_next_next_next[next_next_next_action]

			next_next_V = np.sum(next_next_next_action_probs * q_values_next_next_next)

			Delta_t_2 = next_next_reward + discount_factor * next_next_V - q_values_next_next_state_next_next_action


			next_action_selection_probability = np.max(next_action_probs)
			next_next_action_selection_probability = np.max(next_next_action_probs)


			td_target = q_values_state_action + Delta + discount_factor * next_action_selection_probability * Delta_t_1 + discount_factor * discount_factor * next_action_selection_probability * next_next_action_selection_probability * Delta_t_2


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
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if done:
				break

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






from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample

def Q_Sigma_Off_Policy(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	alpha = 0.01


	for i_episode in range(num_episodes):

		print "Episode Number, Off Policy Q(sigma):", i_episode		

		off_policy = behaviour_policy_epsilon_greedy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

		state = env.reset()

		# action_probs = off_policy(state)
		# action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
		next_action = None


		for t in itertools.count():
			if next_action is None:
				action_probs = off_policy(state)
				action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			else:
				action = next_action

			state_t_1, reward, done, _ = env.step(action)
			if done:
				break			


			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t


			q_values = estimator.predict(state)
			q_values_state_action = q_values[action]
			#evaluate Q(current state, current action

			#select sigma value
			probability = 0.5
			sigma_t_1 = binomial_sigma(probability)

			#select next action based on the behaviour policy at next state
			next_action_probs = off_policy(state_t_1)
			action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


			q_values_t_1 = estimator.predict(state_t_1)
			q_values_next_state_next_action = q_values_t_1[action_t_1]
			# features_state_1 = featurize_state(state_t_1)
			# q_values_t_1 = np.dot(theta.T, features_state_1)
			# q_values_next_state_next_action = q_values_t_1[action_t_1]


			on_policy_next_action_probs = policy(state_t_1)
			on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
			V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

			Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action


			"""
			target for one step
			1 step TD Target --- G_t(1)
			"""
			td_target = q_values_state_action + Delta_t 

			estimator.update(state, action, td_target)

			state = state_t_1
			# action = action_t_1

	return stats



def main():
	estimator = Estimator()
	num_episodes = 1000

	print "Running for Total Episodes", num_episodes

	smoothing_window = 1

	# stats_Q_sigma = Q_Sigma_On_Policy(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_Q_sigma = pd.Series(stats_Q_sigma.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_Q_sigma
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Q_Sigma' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_Q_sigma)
	# env.close()


	# stats_2_TB = two_step_tree_backup(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_2_TB = pd.Series(stats_2_TB.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_2_TB
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + '2_Step_Tree_Backup_' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_2_TB)
	# env.close()

	# stats_expected_sarsa = expected_sarsa(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_expected_sarsa = pd.Series(stats_expected_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_expected_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Expected_SARSA_' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_expected_sarsa)
	# env.close()

	# stats_q_learning = q_learning(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_stats_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_stats_q_learning
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Q_Learning_' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_q_learning)
	# env.close()


	# stats_sarsa = sarsa(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_sarsa
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'SARSA_' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_sarsa)
	# env.close()


	# stats_3_TB = three_step_tree_backup(env, estimator, num_episodes, epsilon=0.1)
	# rewards_smoothed_stats_3_TB = pd.Series(stats_3_TB.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	# cum_rwd = rewards_smoothed_stats_3_TB
	# np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + '3_Step_Tree_Backup_' + '.npy', cum_rwd)
	# plotting.plot_episode_stats(stats_3_TB)
	# env.close()



	stats_off_Q_sigma = Q_Sigma_Off_Policy(env, estimator, num_episodes, epsilon=0.1)
	rewards_smoothed_stats_off_Q_sigma = pd.Series(stats_off_Q_sigma.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()	
	cum_rwd = rewards_smoothed_stats_off_Q_sigma
	np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/raw_results/'  + 'Off_Policy_Q_sigma' + '.npy', cum_rwd)
	plotting.plot_episode_stats(stats_off_Q_sigma)
	env.close()












if __name__ == '__main__':
	main()


