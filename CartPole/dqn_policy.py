import random
# random.seed(0)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class Dqn_policy:

    def __init__(self, env, epsilon):
        # how many actions can you execute
        self.n_actions = env.action_space.n  # 2
        # state representation and its shape(n,)
        self.ob_shape = env.observation_space.shape  # (4,)
        # feature length
        self.n_fea = self.n_actions * self.ob_shape[0]  # 8
        # approximation model: DQN(8 * 30 * 20 * 1)
        self.actions = list(range(self.n_actions))
        self.model = Sequential()
        self.model.add(
            Dense(20, init='lecun_uniform', input_shape=(self.n_fea,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(20, init='lecun_uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(1, init='lecun_uniform'))
        self.model.add(Activation('linear'))
        self.rms = Adam()
        self.epsilon = epsilon

    '''
    get_state_action_fea:
    construct the feature vector with state and action
    '''

    def get_state_action_fea(self, state_fea, action):

        f = np.zeros(self.n_fea)
        f[int(action * state_fea.shape[0]):
          int((action + 1) * state_fea.shape[0])] = state_fea
        return f

    def qfunc(self, state_fea, action):
        f = self.get_state_action_fea(state_fea, action)
        return self.model.predict(f.reshape(1, self.n_fea), batch_size=1)

    def greedy(self, fea):
        q_max = self.qfunc(fea, self.actions[0])
        i_max = 0
        for i in range(1, len(self.actions)):
            temp_max = self.qfunc(fea, self.actions[i])
            if temp_max > q_max:
                q_max = temp_max
                i_max = i
        return self.actions[i_max]

    def epsilon_greedy(self, fea):
        epsilon = self.epsilon
        action_max = self.greedy(fea)
        rand = random.random()
        if rand < epsilon:
            return random.sample(self.actions, 1)[0]
        else:
            return action_max
