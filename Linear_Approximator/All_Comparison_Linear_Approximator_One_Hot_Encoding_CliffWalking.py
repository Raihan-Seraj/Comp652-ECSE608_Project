# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 21:56:50 2017

@author: Raihan
"""

import sys
#sys.path.insert(0, "/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/")
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

#convert states to a feature representation:
#used an RBF sampler here for the feature map
feature_length=1
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
#     nA = env.action_space.n

#     for a in range(nA):
#         features = one_hot_encoding(state, a)


#     state_features_a1 =one_hot_encoding(state,action=0)
#      state_features_a2=one_hot_encoding(state,action=1)
#             state_features_a3=one_hot_encoding(state,action=2)
#             state_features_a4=one_hot_encoding(state,action=3)
#             state_features=np.concatenate((state_features_a1,state_features_a2,state_features_a3,state_features_a4),axis=1)
# return state_features













def q_learning(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,alpha=0.5):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    for i_episode in range(num_episodes):
        print ("Episode Number, Q Learning:", i_episode)
        #agent policy based on the greedy maximisation of Q
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
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
            features_state=featurize_state(state)
            q_value=np.dot(theta.T,features_state)
            q_value_state_action=q_value[action]#computing Q(s,a)
            next_features_state=featurize_state(next_state)
            next_q_value=np.dot(theta.T,next_features_state) #computing Q(s',for all as)
            
            
            

            #Q-value TD Target
            td_target = reward + discount_factor * np.max(next_q_value)
            delta=td_target-q_value_state_action
            #print(np.shape(theta))
            #sigma=alpha*delta*np.ones(192)
            #print(np.shape(sigma))
            #print (np.shape(features_state))
            theta+=alpha*delta*features_state[:,action]
            #print(theta)
            if done:
                break
            state = next_state
    return stats


def sarsa(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0,alpha=0.5):
    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        print ("Episode Number, SARSA:", i_episode)
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
        last_reward = stats.episode_rewards[i_episode - 1]
        state = env.reset()
        next_action = None

        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)        

        # if next_action is None:
        #     action_probs = policy(state)
        #     action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # else:
        #     action = next_action

        # action_probs = policy(state)
        # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():

            next_state, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            #update Q-values for the next state, next action
            features_state=featurize_state(state)
            q_values_state=np.dot(theta.T,features_state)
            q_state_action=q_values_state[action]
            features_next_state=featurize_state(next_state)  
            q_values_next = np.dot(theta.T,features_next_state)

            q_next_state_next_action = q_values_next[next_action] 

            td_target = reward + discount_factor * q_next_state_next_action

            delta=td_target-q_state_action
            theta+=alpha*delta*features_state[:,action]

            if done:
                break

            state = next_state
            action = next_action

    return stats


def on_policy_expected_sarsa(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,alpha=0.5):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    for i_episode in range(num_episodes):
        print ("Episode Number, Expected SARSA:", i_episode)
        #agent policy based on the greedy maximisation of Q
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
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
            features_state=featurize_state(state)
            q_values_state=np.dot(theta.T,features_state)
            q_values_state_action=q_values_state[action]
            features_state_next=featurize_state(next_state)
            
            q_values_next_state = np.dot(theta.T,features_state_next)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            V = np.sum( next_action_probs * q_values_next_state)


            #Q-value TD Target
            td_target = reward + discount_factor * V
            delta=td_target-q_values_state_action
            

            theta+=delta*alpha*features_state[:,action]
            if done:
                break
            state = next_state
    return stats


def two_step_tree_backup(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,alpha=0.5):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    for i_episode in range(num_episodes):
        print ("Episode Number, Two Step Tree Backup:", i_episode)
        #agent policy based on the greedy maximisation of Q
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
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
            features_state = featurize_state(state)
            q_values = np.dot(theta.T, features_state)
            q_values_state_action = q_values[action]
            if done:
                break
                
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            features_state=featurize_state(state)
            
            #update Q-values for the next state
            features_next_state=featurize_state(next_state)
        
            q_values_next = np.dot(theta.T,features_next_state)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            V = np.sum( next_action_probs * q_values_next)


            next_next_state, next_reward, _, _ = env.step(next_action)
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)
            features_next_next_state=featurize_state(next_next_state)
            q_values_next_next = np.dot(theta.T,features_next_next_state)

            next_V = np.sum(next_next_action_probs * q_values_next_next)
            

            q_next_next_state_next_next_action = q_values_next_next[next_next_action]

            Delta = next_reward + discount_factor * next_V - q_next_next_state_next_next_action

            next_action_selection_probability = np.max(next_action_probs)

            td_target = reward + discount_factor * V + discount_factor * next_action_selection_probability * Delta

            update=td_target-q_values_state_action
            theta+=alpha*update*features_state[:,action]
            

            # if done:
            #     break
            state = next_state

    return stats






def three_step_tree_backup(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,alpha=0.5):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    for i_episode in range(num_episodes):
        print ("Episode Number, Three Step Tree Backup:", i_episode)
        #agent policy based on the greedy maximisation of Q
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
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
            features_state=featurize_state(state)
            q_values = np.dot(theta.T,features_state)
            q_values_state_action = q_values[action]


            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)
            

            #update Q-values for the next state
            features_next_state=featurize_state(next_state)
            q_values_next = np.dot(theta.T,features_next_state)
            q_values_next_state_next_action = q_values_next[next_action]

            V = np.sum( next_action_probs * q_values_next)

            Delta = reward + discount_factor * V - q_values_state_action


            next_next_state, next_reward, _, _ = env.step(next_action)
            next_next_action_probs = policy(next_next_state)
            next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)
            features_next_next_state=featurize_state(next_next_state)

            q_values_next_next = np.dot(theta.T,features_next_next_state)
            q_values_next_next_state_next_next_action = q_values_next_next[next_next_action]

            next_V = np.sum(next_next_action_probs * q_values_next_next)

            Delta_t_1 = next_reward + discount_factor * next_V - q_values_next_state_next_action


            next_next_next_state, next_next_reward, _, _ = env.step(next_next_action)
            next_next_next_action_probs = policy(next_next_next_state)
            next_next_next_action = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)
            features_next_next_next_state=featurize_state(next_next_next_state)
            q_values_next_next_next = np.dot(theta.T,features_next_next_next_state)
            q_values_next_next_next_state_next_next_next_action = q_values_next_next_next[next_next_next_action]

            next_next_V = np.sum(next_next_next_action_probs * q_values_next_next_next)

            Delta_t_2 = next_next_reward + discount_factor * next_next_V - q_values_next_next_state_next_next_action


            next_action_selection_probability = np.max(next_action_probs)
            next_next_action_selection_probability = np.max(next_next_action_probs)


            td_target = q_values_state_action + Delta + discount_factor * next_action_selection_probability * Delta_t_1 + discount_factor * discount_factor * next_action_selection_probability * next_next_action_selection_probability * Delta_t_2

            td_error=td_target-q_values_state_action
            theta+=alpha*td_error*features_state[:,action]

            # if done:
            #     break
            state = next_state

    return stats




def Q_Sigma_On_Policy(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999,alpha=0.5):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    for i_episode in range(num_episodes):
        print ("Episode Number, Q(sigma):", i_episode)
        #agent policy based on the greedy maximisation of Q
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

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
            features_state=featurize_state(state)
            q_value_state=np.dot(theta.T,features_state)
            q_value_state_action=q_value_state[action]
            features_state_next=featurize_state(next_state)
            q_values_next = np.dot(theta.T,features_state_next)

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

            sigma = random.randint(0,1)

            V = np.sum( next_action_probs * q_values_next)

            q_next_state_next_action = q_values_next[next_action]

            Sigma_Effect = sigma * q_next_state_next_action+ (1 - sigma) * V

            td_target = reward + discount_factor * Sigma_Effect

            #Q-value TD Target
            td_target = reward + discount_factor * V
            delta=td_target-q_value_state_action
            ##check this update as well
            theta+=alpha*delta*features_state[:,action]
            if done:
                break
            state = next_state
    return stats















def sarsa_lambda(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, alpha=0.5, lambda_param=1):

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        print ("Episode Number, SARSA(lambda):", i_episode)

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


def epsilon_greedy_policy(theta, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        phi = featurize_state(observation)
        q_values = np.dot(theta.T, phi)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn



def Q_Sigma_Off_Policy(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    alpha = 0.01


    for i_episode in range(num_episodes):

        print ("Epsisode Number Off Policy Q(sigma)", i_episode)

        off_policy = behaviour_policy_epsilon_greedy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

        state = env.reset()
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



            # q_values = estimator.predict(state)
            # q_values_state_action = q_values[action]
            #evaluate Q(current state, current action)
            features_state = featurize_state(state)
            q_values = np.dot(theta.T, features_state)
            q_values_state_action = q_values[action]



            #select sigma value
            probability = 0.5
            sigma_t_1 = binomial_sigma(probability)

            #select next action based on the behaviour policy at next state
            next_action_probs = off_policy(state_t_1)
            action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


            # q_values_t_1 = estimator.predict(state_t_1)
            # q_values_next_state_next_action = q_values_t_1[action_t_1]
            features_state_1 = featurize_state(state_t_1)
            q_values_t_1 = np.dot(theta.T, features_state_1)
            q_values_next_state_next_action = q_values_t_1[action_t_1]


            on_policy_next_action_probs = policy(state_t_1)
            on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
            V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

            Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action


            """
            target for one step
            1 step TD Target --- G_t(1)
            """
            td_target = q_values_state_action + Delta_t 

            td_error = td_target -  q_values_state_action 

            # estimator.update(state, action, new_td_target)
            theta += alpha * td_error * features_state[:,action]


            state = state_t_1

    return stats





def Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  
    alpha = 0.01
    for i_episode in range(num_episodes):

        print ("Epsisode Number Off Policy Q(sigma) 2 Step", i_episode)

        off_policy = behaviour_policy_epsilon_greedy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

        state = env.reset()
        next_action = None

        for t in itertools.count():

            if next_action is None:
                action_probs = off_policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            state_t_1, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break            

            # q_values = estimator.predict(state)
            # q_values_state_action = q_values[action]
            #evaluate Q(current state, current action)
            features_state = featurize_state(state)
            q_values = np.dot(theta.T, features_state)
            q_values_state_action = q_values[action]


            #select sigma value
            probability = 0.5
            sigma_t_1 = binomial_sigma(probability)

            #select next action based on the behaviour policy at next state
            next_action_probs = off_policy(state_t_1)
            action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


            # q_values_t_1 = estimator.predict(state_t_1)
            # q_values_next_state_next_action = q_values_t_1[action_t_1]
            features_state_1 = featurize_state(state_t_1)
            q_values_t_1 = np.dot(theta.T, features_state_1)
            q_values_next_state_next_action = q_values_t_1[action_t_1]


            on_policy_next_action_probs = policy(state_t_1)
            on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
            V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

            Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action



            state_t_2, next_reward, _, _ = env.step(action_t_1)

            next_next_action_probs = off_policy(state_t_2)
            action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


            # q_values_t_2 = estimator.predict(state_t_2)
            # q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]
            features_state_2 = featurize_state(state_t_2)
            q_values_t_2 = np.dot(theta.T, features_state_2)
            q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]




            on_policy_next_next_action_probs = policy(state_t_2)
            on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
            V_t_2 = np.sum( on_policy_next_next_action_probs * q_values_t_2  )
            
            sigma_t_2 = binomial_sigma(probability)

            Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * q_values_next_next_state_next_next_action + (1 - sigma_t_2) * V_t_2   ) - q_values_next_state_next_action

            """
            2 step TD Target --- G_t(2)
            """
            on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
            off_policy_action_probability = next_action_probs[action_t_1]

            td_target = q_values_state_action + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1

            """
            Computing Importance Sampling Ratio
            """
            rho = np.divide( on_policy_action_probability, off_policy_action_probability )
            rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1

            td_error = td_target -  q_values_state_action 

            # estimator.update(state, action, new_td_target)
            theta += alpha * rho_sigma * td_error * features_state[:,action]

            state = state_t_1
            
    return stats





def Q_Sigma_Off_Policy_3_Step(env, theta, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):

    #q-learning algorithm with linear function approximation here

    #estimator : Estimator of Q^w(s,a)    - function approximator
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))  

    alpha = 0.01


    for i_episode in range(num_episodes):

        print ("Epsisode Number Off Policy Q(sigma) 3 Step", i_episode)

        off_policy = behaviour_policy_epsilon_greedy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)
        policy = epsilon_greedy_policy(theta, epsilon * epsilon_decay**i_episode, env.action_space.n)

        state = env.reset()

        next_action = None


        for t in itertools.count():

            if next_action is None:
                action_probs = off_policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            state_t_1, reward, done, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break            


            # q_values = estimator.predict(state)
            # q_values_state_action = q_values[action]
            #evaluate Q(current state, current action)
            features_state = featurize_state(state)
            q_values = np.dot(theta.T, features_state)
            q_values_state_action = q_values[action]


            #select sigma value
            probability = 0.5
            sigma_t_1 = binomial_sigma(probability)

            #select next action based on the behaviour policy at next state
            next_action_probs = off_policy(state_t_1)
            action_t_1 = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)


            # q_values_t_1 = estimator.predict(state_t_1)
            # q_values_next_state_next_action = q_values_t_1[action_t_1]
            features_state_1 = featurize_state(state_t_1)
            q_values_t_1 = np.dot(theta.T, features_state_1)
            q_values_next_state_next_action = q_values_t_1[action_t_1]


            on_policy_next_action_probs = policy(state_t_1)
            on_policy_a_t_1 = np.random.choice(np.arange(len(on_policy_next_action_probs)), p = on_policy_next_action_probs)
            V_t_1 = np.sum( on_policy_next_action_probs * q_values_t_1 )

            Delta_t = reward + discount_factor * ( sigma_t_1 * q_values_next_state_next_action + (1 - sigma_t_1) * V_t_1  ) - q_values_state_action



            state_t_2, next_reward, _, _ = env.step(action_t_1)
            # if done:
            #     break

            next_next_action_probs = off_policy(state_t_2)
            action_t_2 = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


            # q_values_t_2 = estimator.predict(state_t_2)
            # q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]
            features_state_2 = featurize_state(state_t_2)
            q_values_t_2 = np.dot(theta.T, features_state_2)
            q_values_next_next_state_next_next_action = q_values_t_2[action_t_2]




            on_policy_next_next_action_probs = policy(state_t_2)
            on_policy_a_t_2 = np.random.choice(np.arange(len(on_policy_next_next_action_probs)), p = on_policy_next_next_action_probs)
            V_t_2 = np.sum( on_policy_next_next_action_probs * q_values_t_2  )
            
            sigma_t_2 = binomial_sigma(probability)



            Delta_t_1 = next_reward + discount_factor * (  sigma_t_2 * q_values_next_next_state_next_next_action + (1 - sigma_t_2) * V_t_2   ) - q_values_next_state_next_action


            """
            3 step TD Target --- G_t(2)
            """
            state_t_3, next_next_reward, _, _ = env.step(action_t_2)
            # if done:
            #     break


            next_next_next_action_probs = off_policy(state_t_3)
            action_t_3 = np.random.choice(np.arange(len(next_next_next_action_probs)), p = next_next_next_action_probs)

            features_state_3 = featurize_state(state_t_3)
            q_values_t_3 = np.dot(theta.T,features_state_3)
            q_values_next_next_next_state_next_next_next_action = q_values_t_3[action_t_3]

            on_policy_next_next_next_action_probs = policy(state_t_3)
            on_policy_a_t_3 = np.random.choice(np.arange(len(on_policy_next_next_next_action_probs)), p = on_policy_next_next_next_action_probs)
            V_t_3 = np.sum(on_policy_next_next_next_action_probs * q_values_t_3)

            sigma_t_3 = binomial_sigma(probability)

            Delta_t_2 = next_next_reward + discount_factor * (sigma_t_3 * q_values_next_next_next_state_next_next_next_action + (1 - sigma_t_3) * V_t_3 ) -  q_values_next_next_state_next_next_action



            on_policy_action_probability = on_policy_next_action_probs[on_policy_a_t_1]
            off_policy_action_probability = next_action_probs[action_t_1]

            on_policy_next_action_probability = on_policy_next_next_action_probs[on_policy_a_t_2]
            off_policy_next_action_probability = next_next_action_probs[action_t_2]



            td_target = q_values_state_action + Delta_t + discount_factor * ( (1 - sigma_t_1) *  on_policy_action_probability + sigma_t_1 ) * Delta_t_1 + discount_factor * ( (1 - sigma_t_2)  * on_policy_next_action_probability + sigma_t_2 ) * Delta_t_2

            """
            Computing Importance Sampling Ratio
            """
            rho = np.divide( on_policy_action_probability, off_policy_action_probability )
            rho_1 = np.divide( on_policy_next_action_probability, off_policy_next_action_probability )

            rho_sigma = sigma_t_1 * rho + 1 - sigma_t_1
            rho_sigma_1 = sigma_t_2 * rho_1 + 1 - sigma_t_2

            all_rho_sigma = rho_sigma * rho_sigma_1

            td_error = td_target -  q_values_state_action 

            # estimator.update(state, action, new_td_target)
            theta += alpha * all_rho_sigma * td_error * features_state[:,action]

            state = state_t_1
            
    return stats






def main():

    
     theta = np.zeros(shape=(total_states))

    
     print ("Q Learning")
     num_episodes = 1000
     smoothing_window = 1
     stats_q_learning = q_learning(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_q_learning = pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_q_learning
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Q_Learning' + '.npy', cum_rwd)
     env.close()


     print ("SARSA")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_sarsa = sarsa(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_sarsa = pd.Series(stats_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_sarsa
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'SARSA' + '.npy', cum_rwd)
     env.close()


     print ("On Policy Expected SARSA")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_expected_sarsa = on_policy_expected_sarsa(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_expected_sarsa = pd.Series(stats_expected_sarsa.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_expected_sarsa
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Expected_SARSA' + '.npy', cum_rwd)
     env.close()


     print ("Two Step Tree Backup")
    
     num_episodes = 1000
     smoothing_window = 1
     stats_two_step_tree_backup = two_step_tree_backup(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_two_step_tree_backup = pd.Series(stats_two_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_two_step_tree_backup
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Two_Step_Tree_Backup' +'.npy', cum_rwd)
     env.close()


     print ("Three Step Tree Backup")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_three_step_tree_backup = three_step_tree_backup(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_three_step_tree_backup = pd.Series(stats_three_step_tree_backup.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_three_step_tree_backup
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Three_Step_Tree_Backup' + '.npy', cum_rwd)
     env.close()



     print ("On Policy Q(sigma)")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_q_sigma_on_policy = Q_Sigma_On_Policy(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_q_sigma_on_policy = pd.Series(stats_q_sigma_on_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_q_sigma_on_policy
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Q(sigma)_On_Policy' + '.npy', cum_rwd)
     env.close()



     print ("Off Policy Q(sigma)")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_q_sigma_off_policy = Q_Sigma_Off_Policy(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_q_sigma_off_policy = pd.Series(stats_q_sigma_off_policy.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_q_sigma_off_policy
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Off_Policy_Q_Sigma' + '.npy', cum_rwd)
     env.close()


    

     print ("Off Policy Q(sigma) 2 Step")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_q_sigma_off_policy_2 = Q_Sigma_Off_Policy_2_Step(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_q_sigma_off_policy_2 = pd.Series(stats_q_sigma_off_policy_2.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_q_sigma_off_policy_2
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Off_Policy_Q_Sigma_2_Step' + '.npy', cum_rwd)
     env.close()



     print ("Off Policy Q(sigma) 3 Step")
     
     num_episodes = 1000
     smoothing_window = 1
     stats_q_sigma_off_policy_3 = Q_Sigma_Off_Policy_3_Step(env, theta, num_episodes, epsilon=0.1)
     rewards_smoothed_stats_q_sigma_off_policy_3 = pd.Series(stats_q_sigma_off_policy_3.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()    
     cum_rwd = rewards_smoothed_stats_q_sigma_off_policy_3
     np.save('G:/studies/McGill/Machine Learning/Final_Project/Cliff_Walking_one_hot_results/'  + 'Off_Policy_Q_Sigma_3_Step' + '.npy', cum_rwd)
     env.close()


   
    
if __name__ == '__main__':
    main()