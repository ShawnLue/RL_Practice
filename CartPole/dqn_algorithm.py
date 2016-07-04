import random
import numpy as np
import matplotlib.pyplot as plt


def update(policy, minibatch, alpha, gamma, batchSize, target_weights):
    X_train, Y_train = [], []
    model_save = policy.model.get_weights()
    policy.model.set_weights(target_weights)

    for memory in minibatch:
        s0, a0, r, s1 = memory
        a1 = policy.greedy(s1)
        state_m = policy.get_state_action_fea(s0, a0)
        if r == 1:
            Q_next = policy.qfunc(s1, a1)
            y = r + gamma * Q_next
        else:
            y = r
        X_train.append(state_m.reshape(policy.n_fea,))
        Y_train.append(np.array(y).reshape(1,))
    policy.model.set_weights(model_save)
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    policy.model.fit(
        X_train, Y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

'''
def sarsa(env, policy, num_iter, alpha, gamma):

    epsilon_minus = (policy.epsilon - 0.01) / int(0.8 * num_iter)
    policy.model.compile(loss='mse', optimizer=policy.rms)

    ##
    batchSize = 10
    buff = 30
    replay = []
    ##
    for i in range(num_iter):
        s_f = env.reset()
        a = policy.epsilon_greedy(s_f)
        terminate = False
        count = 0
        h = -1
        while not terminate and count < 10000:
            s_f1, r, terminate, _ = env.step(policy.actions[a])
            if terminate is False:
                r = 1
            else:
                r = -100
            a1 = policy.epsilon_greedy(s_f1)
            if len(replay) < buff:
                replay.append((s_f, a, r, s_f1, a1))
            else:
                h += 1
                h = h % buff
                replay[h] = (s_f, a, r, s_f1, a1)
                minibatch = random.sample(replay, batchSize)
                update(policy, minibatch, alpha, gamma, batchSize)
            s_f = s_f1
            a = a1
            count += 1
        policy.epsilon -= epsilon_minus
        if i % 100 == 0:
            print("%d %% epochs" % (i * 100/num_iter))
    return policy
'''


def qlearning(env, policy, num_iter, alpha, gamma):

    epsilon_minus = (policy.epsilon - 0.01) / int(0.8 * num_iter)
    epsilon_divide = 0.99

    policy.model.compile(loss='mse', optimizer=policy.rms)

    ##
    batchSize = 50
    buff = 10000
    target_update_freq = 400
    replay = []
    ##
    target_weights = policy.model.get_weights()
    timestep_trace = []
    for i in range(num_iter):
        s_f = env.reset()
        a = policy.epsilon_greedy(s_f)
        terminate = False
        count = 0
        while not terminate and count < 10000:
            s_f1, r, terminate, _ = env.step(policy.actions[a])
            if terminate is True:
                r = -100
            replay.append((s_f, a, r, s_f1))
            if len(replay) > buff:
                replay.pop(0)
            if len(replay) > batchSize:
                minibatch = random.sample(replay, batchSize)
                update(
                    policy, minibatch, alpha, gamma, batchSize, target_weights)
            s_f = s_f1
            a = policy.epsilon_greedy(s_f)
            count += 1
        #policy.epsilon -= epsilon_minus
        if policy.epsilon > 0.01:
            policy.epsilon *= epsilon_divide
        if i % 100 == 0:
            print("%d %% epochs" % (i * 100/num_iter))
        if i % target_update_freq == 0:
            target_weights = policy.model.get_weights()
        timestep_trace.append(count)
    plt.plot(timestep_trace)
    plt.show()
    return policy
