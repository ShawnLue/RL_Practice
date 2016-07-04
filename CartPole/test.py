import gym
#from dqn_policy import *
env = gym.make('CartPole-v0')
#p = Dqn_policy(env, 0.3)
observation = env.reset()
print(env.observation_space.shape)
'''
for i_episode in range(1):
    observation = env.reset()
    print(env.observation_space.shape)
    for t in range(2):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            break
'''
