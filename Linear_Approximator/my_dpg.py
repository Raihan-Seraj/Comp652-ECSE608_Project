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


from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
#env = CliffWalkingEnv()
env=WindyGridworldEnv()


# #with the mountaincar from openAi gym
# env = gym.envs.make("MountainCar-v0")


#samples from the state space to compute the features
observation_examples = np.array([env.observation_space.sample() for x in range(1)])



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



def featurize_state(state):

	state = np.array([state])

	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]




def make_epsilon_greedy_policy(w, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def behaviour_policy_epsilon_greedy(w, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def create_greedy_policy(w, epsilon, nA):
    def policy_fn(observation):
        A = np.zeros(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(w.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] = 1
        return A
    return policy_fn


def return_action_probs(theta,state):
	state_feature=featurize_state(state)
	#print (np.shape(state_feature))
	soft_max=(np.exp(np.dot(theta.T, state_feature)))/(np.sum(np.exp(np.dot(theta.T, state_feature))))

	#gaussian=(1/np.sqrt((2*pi*sigma**2)))*np.exp(-)

	return soft_max







from numpy.random import binomial
def binomial_sigma(p):
	sample = binomial(n=1, p=p)
	return sample

def behaviour_policy_Boltzmann(w, tau, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * tau / nA
        phi = featurize_state(observation)
        q_values = np.dot(w.T, phi)
        exp_tau = q_values / tau
        policy = np.exp(exp_tau) / np.sum(np.exp(exp_tau), axis=0)
        A = policy

        return A
    return policy_fn



def actor_critic(env, w,theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.99):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)) 
	cumulative_errors = np.zeros(shape=(num_episodes, 1)) 

	alpha = 0.01
	tau=1

  
	for i_episode in range(num_episodes):
		state_count=np.zeros(shape=(env.observation_space.n,1))

		print ("Epsisode Number actor_critic", i_episode)

		state = env.reset()
		
		for t in itertools.count():
			action_probs=return_action_probs(theta,state)

			#print("Action Probs", action_probs)

			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

			#print("Action", action)

			features_state=featurize_state(state)

			# action = np.random.normal(np.dot(theta.T, features_state), 0.5)

			value=np.dot(w.T,features_state)
			value_action=value[action]


			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward

			next_features_state=featurize_state(next_state)
			value_next=np.dot(w.T,next_features_state)

			td_target = reward + discount_factor * value_next[action]
			
			td_error=td_target-value[action]
			#print(np.shape(td_error))

			if done:
				break			
					
			w[:, action] += alpha * td_error * features_state

			# zeta=action_probs-np.dot(theta.T,features_state)
			# zeta=np.array([zeta])
			
			# features_state=np.array([features_state])
			# #print(np.shape(features_state))			
			# beta=np.dot(zeta.T,features_state)

			# beta=beta/(2*0.9**2)
			# #print(np.shape(beta))
			# nabla=np.dot(beta.T,value)
			# #print(nabla)
			# theta[:,action]+=alpha*#(features_state)*value[action]#(((action_probs-theta.T*features_state))*features_state.T/(2*0.1**2)).T*value
			# #print(theta)

			grad_log = ((action - np.dot(theta.T, features_state)[action]) * features_state) / 0.1**2
			grad_log= features_state-np.ones(shape=(len(features_state)))

			#print("Grad Log", grad_log)

			theta[:, action] += alpha * grad_log * td_error
			#print ("Theta", theta)

			rms_error = np.sqrt(np.sum((td_error)**2))
			cumulative_errors[i_episode, :] += rms_error

			state = next_state

	return stats,cumulative_errors



def take_average_results(experiment,num_experiments,num_episodes,env,w):
	reward_mat=np.zeros([num_episodes,num_experiments])
	error_mat=np.zeros([num_episodes,num_experiments])
	for i in range(num_experiments):
		stats,cum_error=experiment(env,w,num_episodes)
		reward_mat[:,i]=stats.episode_rewards
		error_mat[:,i]=cum_error.T
		average_reward=np.mean(reward_mat,axis=1)
		average_error=np.mean(error_mat,axis=1)
		np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /exploration_based_sigma/'  + 'Qsigma_onpolicy_reward' + '.npy',average_reward)
		np.save('/home/raihan/Desktop/Final_Project_Codes/Windy_GridWorld/Experimental_Results /exploration_based_sigma/'  + 'Qsigma_onpolicy_error' + '.npy',average_error)
		
	return(average_reward,average_error)



def main():
	w = np.random.normal(size=(400,env.action_space.n))
	theta=np.random.normal(size=(400,env.action_space.n))
	num_episodes = 1000
	#num_experiments=20
	print ("Running for Total Episodes", num_episodes)
	smoothing_window = 1
	stats,cum_reward=actor_critic(env,w,theta,num_episodes)



	#avg_cum_reward,avg_cum_error=take_average_results(Q_Sigma_On_Policy,num_experiments,num_episodes,env,w)
	
	env.close()


if __name__ == '__main__':
	main()