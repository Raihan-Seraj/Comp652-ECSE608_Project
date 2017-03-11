import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=0.5):
        self.n_actions = n_actions
        self.value_func = value_function_model
        self.eps = eps

    def act(self, state):
        action_values = self.value_func.predict([state])[0]

        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps

        return np.random.choice(self.n_actions, p=policy)

    def train(self, states, targets):
        return self.value_func.train(states, targets)


    def predict_q_values(self, states):
        return self.value_func.predict(states)


    def evaluate_predicted_q_values(self, states, dropout_probability):
        return self.value_func.predict_stochastic(states, dropout_probability)



    def act_boltzmann(self, state):
        action_values = self.value_func.predict([state])[0]
        action_values_tau = action_values / self.eps
        policy = np.exp(action_values_tau) / np.sum(np.exp(action_values_tau), axis=0)
        action_value_to_take = np.argmax(policy)
        return action_value_to_take



    def make_epsilon_greedy_policy(self, Q, nA):

        def policy_fn(observation):

            A = np.ones(shape=(len(observation), nA), dtype=float) * self.eps/nA
            # A = np.ones(nA, dtype=float) * self.eps/nA
            best_action = np.argmax(Q, axis=1)

            A[:, best_action] += ( 1.0 - self.eps)

            return A

        return policy_fn






