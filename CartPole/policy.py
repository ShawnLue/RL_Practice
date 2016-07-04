#import gym
import sys
import random
# random.seed(0)
import numpy as np


class Policy:

    def __init__(self, env, epsilon):
        # how many actions can you execute
        n_action = env.action_space.n
        # state representation and its shape(n,)
        shape = env.observation_space.shape
        # feature length
        n_fea = n_action * shape[0]
        self.actions = list(range(n_action))
        self.theta = np.zeros(n_fea)
        self.epsilon = epsilon
    '''
    get_state_action_fea:
    construct the feature vector with state and action
    '''

    def get_state_action_fea(self, state_fea, action):
        f = np.zeros(self.theta.shape[0])
        f[int(action * state_fea.shape[0]):
          int((action + 1) * state_fea.shape[0])] = state_fea
        return f

    def qfunc(self, state_fea, action):
        f = self.get_state_action_fea(state_fea, action)
        return np.dot(f, self.theta)

    def greedy(self, fea):
        q_max = self.qfunc(fea, self.actions[0])
        i_max = 0
        for i in range(0, len(self.actions)):
            temp_max = self.qfunc(fea, self.actions[i])
            if temp_max > q_max:
                q_max = temp_max
                i_max = i
        return i_max

    def epsilon_greedy(self, fea):
        epsilon = self.epsilon
        q_max = self.qfunc(fea, self.actions[0])
        i_max = 0
        for i in range(0, len(self.actions)):
            temp_max = self.qfunc(fea, self.actions[i])
            if temp_max > q_max:
                q_max = temp_max
                i_max = i
        p_box = np.zeros(len(self.actions))
        e = epsilon / (len(self.actions) - 1)
        acc = 0.0
        for i in range(len(self.actions)):
            if i == i_max:
                acc += (1 - epsilon)
            else:
                acc += e
            p_box[i] = acc
        rand = random.random()
        return np.argwhere(rand < p_box)[0][0]
