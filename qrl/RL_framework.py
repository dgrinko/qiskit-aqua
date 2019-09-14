# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import matplotlib.pyplot as plt


def plot(total_rewards, average=100, instance=None):
    curves = 1
    if len(total_rewards.shape) == 2:
        n = total_rewards.shape[1]
        curves = total_rewards.shape[0]
    else:
        n = len(total_rewards)
        total_rewards = [total_rewards]

    for i in range(curves):
        running_avg = np.empty(n)
        for t in range(n):
            running_avg[t] = total_rewards[i][max(0, t - average):(t + 1)].mean()
        if instance:
            plt.plot(running_avg, label='{} {}'.format(instance, i))
        else:
            plt.plot(running_avg)
    plt.title('Rewards Running Average over {} steps'.format(average))
    plt.show()


class RLAgent:
    def __init__(self, environment, trainer, name, debug=False):
        self.environment = environment
        self.nb_states = self.environment.nb_states
        self.nb_actions = self.environment.nb_actions
        self.debug = debug
        self.name = name
        self.trainer = trainer
        if self.trainer:
            self.train()

    def loss_function(self, theta, t):
        pass

    def sample_action(self, state, params):
        pass

    def get_action(self, state):
        pass

    def train1(self, train_params, batch_size):
        pass

    def evaluate(self, nb_iterations):
        total_rewards = np.zeros(nb_iterations)
        for i in range(nb_iterations):
            s = self.environment.reset()
            done = False
            total_reward = 0
            while not done:
                a = self.get_action(s)
                s1, r, done = self.environment.step(a)
                total_reward += r
            total_rewards[i] = total_reward
        return total_rewards

    def train(self):
        total_rewards = np.zeros(self.trainer.nb_iterations)
        params = self.trainer.training_params

        for it in range(self.trainer.nb_iterations):
            for i in range(len(params)):
                params[i] = self.trainer.cooling_scheme[i](params[i], it)
            total_rewards[it] = self.train1(params, self.trainer.batch_size)

        if self.trainer.plot_training: plot(total_rewards=total_rewards, average=self.trainer.average)
        return total_rewards


class Evaluator:
    def __init__(self, agents_list, nb_iterations):
        self.agents_list = agents_list
        self.nb_iterations = nb_iterations

    def evaluate(self, plot_results = False, average=100):
        total_rewards_agents = np.zeros((len(self.agents_list), self.nb_iterations))
        for i in range(len(self.agents_list)):
            total_rewards_agents[i] = self.agents_list[i].evaluate(self.nb_iterations)

        if plot_results:
            plot(total_rewards_agents, average=average, instance='Agent')

        return total_rewards_agents


class Trainer:
    def __init__(self, nb_iterations, training_params, cooling_scheme, batch_size, plot_training=False, average=100):
        self.nb_iterations = nb_iterations
        self.training_params = training_params
        self.cooling_scheme = cooling_scheme
        self.batch_size = batch_size
        self.plot_training = plot_training
        self.average = average

class Environment:
    def __init__(self, nb_actions, nb_states):
        self.nb_actions = nb_actions
        self.nb_states = nb_states

    def reset(self):
        pass

    def step(self, action):
        pass

    def sample_actions_space(self):
        pass