import random
import numpy as np


def update(policy, s_fea, a, tvalue, alpha):
    pvalue = policy.qfunc(s_fea, a)
    error = pvalue - tvalue
    s_a_fea = policy.get_state_action_fea(s_fea, a)
    policy.theta -= alpha * error * s_a_fea


def sarsa(env, policy, num_iter, alpha, gamma):
    for i in range(policy.theta.shape[0]):
        policy.theta[i] = 0.0
    epsilon_minus = (policy.epsilon - 0.01) / int(0.8 * num_iter)
    epsilon_divide = 1.01
    for i in range(num_iter):
        s_f = env.reset()
        a = policy.epsilon_greedy(s_f)
        terminate = False
        count = 0

        while not terminate and count < 10000:
            s_f1, r, terminate, _ = env.step(policy.actions[a])
            if terminate is False:
                r = -1
                a1 = policy.epsilon_greedy(s_f1)
                update(policy, s_f, a, r + gamma * policy.qfunc(s_f1, a1), alpha)
            else:
                r = 0
                update(policy, s_f, a, r, alpha)
                break
            s_f = s_f1
            a = a1
            count += 1
        #policy.epsilon -= epsilon_minus
        policy.epsilon /= epsilon_divide
        if i % 100 == 0:
            print("%d %% epochs" % (i * 100/num_iter))
    return policy


def qlearning(env, policy, num_iter, alpha, gamma):
    for i in range(policy.theta.shape[0]):
        policy.theta[i] = 0.0
    epsilon_minus = (policy.epsilon - 0.01) / int(0.8 * num_iter)
    epsilon_divide = 1.01
    for i in range(num_iter):
        s_f = env.reset()
        a = policy.epsilon_greedy(s_f)
        terminate = False
        count = 0

        while not terminate and count < 10000:
            s_f1, r, terminate, _ = env.step(policy.actions[a])
            if terminate is False:
                r = -1
                a1 = policy.greedy(s_f1)
                update(policy, s_f, a, r + gamma * policy.qfunc(s_f1, a1), alpha)
            else:
                r = 0
                a1 = policy.greedy(s_f1)
                update(policy, s_f, a, r, alpha)

            s_f = s_f1
            a = policy.epsilon_greedy(s_f)
            count += 1
        #policy.epsilon -= epsilon_minus
        policy.epsilon /= epsilon_divide
        if i % 100 == 0:
            print("%d %% epochs" % (i * 100/num_iter))
    return policy
