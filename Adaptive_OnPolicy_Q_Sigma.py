# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 11:10:18 2017

@author: Raihan
"""

import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random


from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
env = CliffWalkingEnv()
def make_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += ( 1.0 - epsilon)
        return A

    return policy_fn

def chosen_action(Q):
    best_action = np.argmax(Q)
    return best_action


def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn
    
def behaviour_policy_epsilon_greedy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += ( 1.0 - epsilon)
        return A

    return policy_fn

from numpy.random import binomial
def binomial_sigma(p):
    sample = binomial(n=1, p=p)
    return sample


def adaptive_q_sigma_on_policy(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):

    Q = defaultdict(lambda : np.zeros(env.action_space.n))

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_error=np.zeros(num_episodes))  


    
    alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
    sigma_initialised = np.array([1, 0.75, 0.5, 0.25, 0])

    sigma_decay = 0.99

    All_Rwd_Sigma = np.zeros(shape=(num_episodes, len(sigma_initialised)))
    All_Rwd_Sigma_Alpha = np.zeros(shape=(len(sigma_initialised), len(alpha)))

    All_Error_Sigma = np.zeros(shape=(num_episodes, len(sigma_initialised)))
    All_Error_Sigma_Alpha = np.zeros(shape=(len(sigma_initialised), len(alpha)))

    num_experiments = num_episodes


    for s in range(len(sigma_initialised)):
      
        for alpha_param in range(len(alpha)):

            sigma = sigma_initialised[s]

            print "Sigma Initialisation", sigma

            for i_episode in range(num_episodes):
                
                print "Number of Episodes, Q(sigma) On Policy", i_episode
    

                policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
                state = env.reset()
                action_probs = policy(state)
        
                #choose a from policy derived from Q (which is epsilon-greedy)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    
                #steps within each episode
                for t in itertools.count():
                    #take a step in the environment
                    # take action a, observe r and the next state
                    next_state, reward, done, _ = env.step(action)
        
                    #reward by taking action under the policy pi
                    stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t
                    
        
                    next_action_probs = policy(next_state)
                    next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs )
        

                    V = np.sum(next_action_probs * Q[next_state])
                    Sigma_Effect = sigma * Q[next_state][next_action] + (1 - sigma) * V
        
        
                    td_target = reward + discount_factor * Sigma_Effect
                    td_delta = td_target - Q[state][action]
        
                    rms_error = np.sqrt(np.sum((td_delta)**2)/num_experiments)
                    stats.episode_error[i_episode] += rms_error

                    Q[state][action] += alpha[alpha_param] * td_delta
        

                    if done:
                        break

                    action = next_action
                    state = next_state


                if i_episode <= 300:
                    sigma = sigma

                else:
                    sigma = sigma * sigma_decay  

                    if sigma <= 0.00001:
                        sigma = 0.00001



            cum_rwd_per_episode = np.array([pd.Series(stats.episode_rewards).rolling(1, min_periods=1).mean()])
            cum_error_per_episode = np.array([pd.Series(stats.episode_error).rolling(1, min_periods=1).mean()])
    

            All_Rwd_Sigma[:, s] = cum_error_per_episode
            All_Error_Sigma[:, s] = cum_error_per_episode

            All_Rwd_Sigma_Alpha[s, alpha_param] = cum_rwd_per_episode.T[-1]
            All_Error_Sigma_Alpha[s, alpha_param] = cum_error_per_episode.T[-1]


    return All_Rwd_Sigma, All_Error_Sigma, All_Rwd_Sigma_Alpha, All_Error_Sigma_Alpha
    


def main():
    print "Adaptive Q(sigma) On Policy"
    env = CliffWalkingEnv()
    Total_num_experiments = 10
    num_episodes = 2000

    alpha = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
    sigma_initialised = np.array([1, 0.75, 0.5, 0.25, 0])


    Averaged_All_Rwd_Sigma = np.zeros(shape=(num_episodes, len(sigma_initialised)))
    Averaged_All_Rwd_Sigma_Alpha = np.zeros(shape=(len(sigma_initialised), len(alpha)))

    Averaged_All_Error_Sigma = np.zeros(shape=(num_episodes, len(sigma_initialised)))
    Averaged_All_Error_Sigma_Alpha = np.zeros(shape=(len(sigma_initialised), len(alpha)))



    for e in range(Total_num_experiments):
        All_Rwd_Sigma, All_Error_Sigma, All_Rwd_Sigma_Alpha, All_Error_Sigma_Alpha = adaptive_q_sigma_on_policy(env, num_episodes)

        Averaged_All_Rwd_Sigma = Averaged_All_Rwd_Sigma + All_Rwd_Sigma
        Averaged_All_Rwd_Sigma_Alpha = Averaged_All_Rwd_Sigma_Alpha + All_Rwd_Sigma_Alpha

        Averaged_All_Error_Sigma = Averaged_All_Error_Sigma + All_Error_Sigma
        Averaged_All_Error_Sigma_Alpha = Averaged_All_Error_Sigma_Alpha + All_Error_Sigma_Alpha      


    Averaged_All_Rwd_Sigma = np.true_divide(Averaged_All_Rwd_Sigma, Total_num_experiments)
    Averaged_All_Rwd_Sigma_Alpha = np.true_divide(Averaged_All_Rwd_Sigma_Alpha, Total_num_experiments)
    Averaged_All_Error_Sigma = np.true_divide(Averaged_All_Error_Sigma, Total_num_experiments)
    Averaged_All_Error_Sigma_Alpha = np.true_divide(Averaged_All_Error_Sigma_Alpha, Total_num_experiments)        


    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Adaptive_OnPolicy_Q_Sigma_Results/'  + 'Adaptive_On_Policy_Q_sigma' +  'Reward_Sigma_' + '.npy', Averaged_All_Rwd_Sigma)
    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Adaptive_OnPolicy_Q_Sigma_Results/'  + 'Adaptive_On_Policy_Q_sigma' +  'Sigma_Alpha' + '.npy', Averaged_All_Rwd_Sigma_Alpha)
    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Adaptive_OnPolicy_Q_Sigma_Results/'  + 'Adaptive_On_Policy_Q_sigma' +  'Error_Sigma_' + '.npy', Averaged_All_Error_Sigma)
    np.save('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Tabular/Adaptive_OnPolicy_Q_Sigma_Results/'  + 'Adaptive_On_Policy_Q_sigma' +  'Error_Sigma_Alpha' + '.npy', Averaged_All_Error_Sigma_Alpha)

    # plotting.plot_episode_stats(stats_tree_lambda)
    env.close()



if __name__ == '__main__':
    main()

    