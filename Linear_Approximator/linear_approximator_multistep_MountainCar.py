# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:31:34 2017

@author: Raihan
"""
import gym
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from lib import plotting
from sklearn.linear_model import SGDRegressor 
from sklearn.kernel_approximation import RBFSampler as RBFSampler
#from lib import plotting
import itertools
import sklearn.pipeline
import sklearn.preprocessing
import sklearn 

env=gym.make("MountainCar-v0")

#Creates an epsilon greedy policy 
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

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        
        features=self.featurize_state(s)
        
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
                               
    
            
      
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])

def sarsa_2_step_TD(env,estimator, num_episodes, discount_factor=1.0, epsilon=0.015,epsilon_decay=1.0):
     stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))
     for i_episode in range (num_episodes):
         policy=make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        # last_reward = stats.episode_rewards[i_episode - 1]
         state=env.reset()
         action_probs=policy(state)
         action=np.random.choice(np.arange(len(action_probs)), p = action_probs)
         
         for t in itertools.count():
            next_state,reward,_,_=env.step(action)
            next_action_probs=policy(next_state)
            next_action=np.random.choice(np.arange(len(action_probs)), p = next_action_probs)
            next_2_state, reward_2_step, done, _ = env.step(next_action)
            next_2_action_probs = policy(next_2_state)
            next_2_action = np.random.choice(np.arange(len(next_2_action_probs)), p = next_2_action_probs)
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            Q_val=estimator.predict(next_2_state)
            Q_val=Q_val[next_2_action]
            td_target=reward + discount_factor * reward_2_step + discount_factor*discount_factor*(Q_val)
            estimator.update(state,action,td_target)
            if done:
                print('Episode no',i_episode)
                break
            state=next_state
            action=next_action
     return stats
def Q_learning_2_step_TD(env,estimator,num_episodes,discount_factor=1,epsilon=0.1):
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))
    policy=make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)
    for i_episode in range(num_episodes):
        state=env.reset()
        for t in itertools.count():
            
            action_probs=policy(state)
            action=np.random.choice(np.arange(len(action_probs)), p = action_probs)
            next_state,reward,done,_=env.step(action)
            next_action_probs=policy(next_state)
            next_action=np.random.choice(np.arange(len(action_probs)), p = next_action_probs)
            next_2_state,reward_2_step,done,_=env.step(next_action)
            stats.episode_rewards[i_episode] += reward_2_step
            stats.episode_lengths[i_episode] = t
            Q_val=estimator.predict(next_2_state)
            best_action=np.argmax(Q_val)
            td_target=reward+discount_factor*reward_2_step+discount_factor*discount_factor*Q_val[best_action]
            estimator.update(state,action,td_target)
            if done:
                print('Episode is ',i_episode)
                break
            state=next_state
    return(stats)

def two_step_tree_backup(env,estimator, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):


   
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),episode_rewards=np.zeros(num_episodes))  

    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):

        state = env.reset()

        #steps within each episode
        for t in itertools.count():
            
            #pick the first action
            #choose A from S using policy derived from Q (epsilon-greedy)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

            #reward and next state based on the action chosen according to epislon greedy policy
            next_state, reward, _ , _ = env.step(action)
            
            #reward by taking action under the policy pi
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p =next_action_probs )

            #V = sum_a pi(a, s_{t+1})Q(s_{t+1}, a)
            V = np.sum(next_action_probs * estimator.predict(next_state))


            next_next_state, next_reward, done, _ = env.step(next_action)
    
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)

            next_V = np.sum(next_next_action_probs * estimator.predict(next_next_state))            


            # print "Next Action:", next_action
            # print "Next Action probs :", next_action_probs

            #Main Update Equations for Two Step Tree Backup
            Q_next_state_next_action=estimator.predict(next_state)
            Q_next_state_next_action=Q_next_state_next_action[next_action]
            
            Delta = next_reward + discount_factor * next_V - Q_next_state_next_action

            # print "Delta :", Delta

            # print "Next Action Prob ", np.max(next_action_probs)

            next_action_selection_probability = np.max(next_action_probs)

            td_target = reward + discount_factor * V +  discount_factor *  next_action_selection_probability * Delta
            estimator.update(state,action,td_target)



            if done:
                print('Episode is ',i_episode)
                break

            state = next_state

    return stats
def plot_episode_stats(stats1,stats2,stats3,smoothing_window=200,noshow=False):
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed_1 = pd.Series(stats1.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(rewards_smoothed_1, label="Two step SARSA with Epsilon Greedy")
    cum_rwd_2, = plt.plot(rewards_smoothed_2, label="Two step Q_Learning with Epsilon Greedy")
    cum_rwd_3, = plt.plot(rewards_smoothed_3, label="Two step Tree backup with Epsilon Greedy")
    

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing 2 step SARSA and 2 step Q Learning and Two step Tree Backup")
    plt.show()


    return fig

    
    
        
def main():
    estimator=Estimator()
    number_of_episodes=500
    print('Two Step Sarsa')
    stats_sarsa = sarsa_2_step_TD(env,estimator, number_of_episodes, discount_factor=1.0, epsilon=0.1,epsilon_decay=1.0)
    plotting.plot_episode_stats(stats_sarsa, smoothing_window=25)
    #print('Two Step Q Learning')
    #stats_Q=Q_learning_2_step_TD(env,estimator,number_of_episodes,discount_factor=1,epsilon=0.1)
    #print('Two Step Tree Backup')
    #stats_tree=two_step_tree_backup(env,estimator, number_of_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1)
    
    #plot_episode_stats(stats_sarsa,stats_Q,stats_tree)
    
    
main()
                    
                



