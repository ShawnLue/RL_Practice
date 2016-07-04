from mdp import get_env
#from policy import *
#from algorithm import *
from dqn_policy import *
from dqn_algorithm import *

if __name__ == '__main__':
    cartpole = get_env('CartPole-v0')
    policy = Dqn_policy(cartpole, epsilon=0.5)
    policy = qlearning(cartpole, policy, 8000, alpha=0.2, gamma=0.9)

    print('a')
    cartpole.monitor.start(
        './cartpole-experiment-1', force=True, video_callable=None, resume=False)

    for iter1 in range(200):
        s_f = cartpole.reset()
        for iter2 in range(5000):
            cartpole.render()
            a = policy.greedy(s_f)
            s_f, _, t, _ = cartpole.step(policy.actions[a])
            if t:
                print("Episode finished after {} timesteps".format(iter2 + 1))
                break
    cartpole.monitor.close()
