import gym
import numpy as np


def get_env(name):
    if name == 'Acrobot-v0':
        return gym.make('Acrobot-v0')
    elif name == 'MountainCar-v0':
        return gym.make('MountainCar-v0')
    elif name == 'CartPole-v0' == name:
        return gym.make('CartPole-v0')
    else:
        raise Exception('Not %s env found' % (name))
