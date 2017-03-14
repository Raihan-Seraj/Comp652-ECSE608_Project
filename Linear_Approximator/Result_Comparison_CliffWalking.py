import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


#taking the final_values onwards, until len(eps)

eps = 1000
eps = range(eps)


sarsa = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/SARSA.npy')
q_learning = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Q_Learning.npy')
expected_sarsa = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Expected_SARSA.npy')
tb_2 = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Two_Step_Tree_Backup.npy')
tb_3 = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Three_Step_Tree_Backup.npy')
q_sigma_on_policy = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Q(sigma)_On_Policy.npy')
q_sigma_off_policy = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Off_Policy_Q_Sigma.npy')
q_sigma_off_policy_2 = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Off_Policy_Q_Sigma_2_Step.npy')
q_sigma_off_policy_3 = np.load('/Users/Riashat/Documents/PhD_Research/BASIC_ALGORITHMS/My_Implementations/Project_652/Code/Linear_Approximator/CliffWalking_Results/Off_Policy_Q_Sigma_3_Step.npy')






def single_plot_episode_stats(stats, eps,  smoothing_window=50, noshow=False):

    #higher the smoothing window, the better the differences can be seen

    ##Plot the episode reward over time
    fig = plt.figure(figsize=(20, 10))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd, = plt.plot(eps, rewards_smoothed, label="Deep Q Learning on Cart Pole")


    plt.legend(handles=[cum_rwd,])
    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("DQN on Cart Pole - Single Run - Larger Network (Layer 1, 512 Units, Layer 2, 256 Units)")
    plt.show()

    return fig





def multiple_plot_episode_stats(stats1, stats2, stats3,  stats4, stats5, stats6, stats7, stats8, stats9,  smoothing_window=200, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_7 = pd.Series(stats7).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_8 = pd.Series(stats8).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_9 = pd.Series(stats9).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="SARSA")    
    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, label="Q Learning")    
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, label="Expected SARSA")    
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, label="2-step Tree Backup")    
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, label="3-step Tree Backup")   
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, label="Q(sigma) On Policy")   
    cum_rwd_7, = plt.plot(eps, rewards_smoothed_7, label="Q(sigma) Off Policy")   
    cum_rwd_8, = plt.plot(eps, rewards_smoothed_8, label="Off Policy Q(sigma) 2-step ")       
    cum_rwd_9, = plt.plot(eps, rewards_smoothed_9, label="Off Policy Q(sigma) 3-step ")       

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3,cum_rwd_4, cum_rwd_5, cum_rwd_6, cum_rwd_7, cum_rwd_8, cum_rwd_9])
    # plt.legend(handles=[cum_rwd_4, cum_rwd_5, cum_rwd_6, cum_rwd_7, cum_rwd_8, cum_rwd_9])

    plt.xlabel("Epsiode")
    plt.ylabel("Epsiode Reward (Smoothed)")
    plt.title("Comparing Multi-Step TD Learning Algorithms - Cliff Walking Environment")  
    plt.show()

    return fig




def main():
	multiple_plot_episode_stats(sarsa, q_learning , expected_sarsa, tb_2, tb_3, q_sigma_on_policy, q_sigma_off_policy, q_sigma_off_policy_2, q_sigma_off_policy_3 )
    #multiple_plot_episode_stats(tb_2, tb_3, q_sigma_on_policy, q_sigma_off_policy, q_sigma_off_policy_2, q_sigma_off_policy_3 )





if __name__ == '__main__':
	main()

