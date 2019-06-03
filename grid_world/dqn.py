import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


class DoubleQNetworkAgent:

    def __init__(self, name: str, n_action: int, n_obs: int, units_layer: tuple,
                 learning_rate=1e-4, gamma=0.99, seed=1,
                 epsilon_init=0, epsilon_increase=0.003, epsilon_max=0.95,
                 buffer_size=50000, batch_size=32, target_change_step=200, min_buffer_size=10000,
                 save_path=None, load_path=None):
        """
        To initialize an agent based on double deep Q learning
        :param name: name of model
        :param n_action: dimension of action space
        :param n_obs: dimension of state space
        :param units_layer: number of hidden layers and numbers of units in each layer
        :param learning_rate: learning rate of optimizer
        :param gamma: reward discount
        :param seed: random seed
        :param epsilon_init: initial epsilon value for epsilon-greedy policy
        :param epsilon_increase: increase step for epsilon after each certain learning step
        :param epsilon_max: upper threshold for epsilon
        :param buffer_size: refers maximum size of transitions in buffer
        :param batch_size: transition used in each training step
        :param target_change_step: length of period for each refresh of Target Q-Network
        :param min_buffer_size: minimum transitions for training process to initiate
        :param save_path: path to save model variables
        :param load_path: path to reload model variables
        """
        self.name = name
        self.n_action = n_action
        self.n_obs = n_obs
        self.n_layer = len(units_layer)
        self.gamma = gamma
        # self.tau = tau
        self.learning_rate = learning_rate
        self.epsilon = epsilon_init
        self.epsilon_increase = epsilon_increase
        self.epsilon_max = epsilon_max
        self.initializer = tf.contrib.layers.xavier_initializer(seed=seed)
        self.units_layer = (self.n_obs,) + units_layer + (self.n_action,)
        self.sess = tf.Session()
        self.transition_count = 0
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.target_change_step = target_change_step
        self.buffer_list = []
        self.learn_step_count = 0
        self.l_history = []

        with tf.variable_scope(self.name + '_input', reuse=tf.AUTO_REUSE):
            self.s = tf.placeholder(tf.float32, [None, self.n_obs], name='s')
            self.s_next = tf.placeholder(tf.float32, [None, self.n_obs], name='s_next')
            self.a = tf.placeholder(tf.int32, name='a')
            self.r = tf.placeholder(tf.float32, name='r')
            self.d = tf.placeholder(tf.float32, name='d')

        with tf.variable_scope(self.name + '_main_params', reuse=tf.AUTO_REUSE):
            self.W_main, self.b_main = [], []
            for i in range(self.n_layer + 1):
                self.W_main.append(tf.get_variable('W' + str(i),
                                                   [self.units_layer[i], self.units_layer[i + 1]],
                                                   initializer=self.initializer))
                self.b_main.append(tf.get_variable('b' + str(i),
                                                   [1, self.units_layer[i + 1]],
                                                   initializer=self.initializer))

        with tf.variable_scope(self.name + '_main_layers', reuse=tf.AUTO_REUSE):
            self.layer_main = [self.s]
            for i in range(self.n_layer + 1):
                if i < self.n_layer:
                    self.layer_main.append(
                        tf.nn.relu(tf.add(tf.matmul(self.layer_main[i], self.W_main[i]), self.b_main[i])))
                else:
                    self.layer_main.append(
                        tf.add(tf.matmul(self.layer_main[i], self.W_main[i]), self.b_main[i]))

        with tf.variable_scope(self.name + '_target_params', reuse=tf.AUTO_REUSE):
            self.W_target, self.b_target = [], []
            for i in range(self.n_layer + 1):
                self.W_target.append(tf.Variable(self.W_main[i].initialized_value(), name='W' + str(i)))
                self.b_target.append(tf.Variable(self.b_main[i].initialized_value(), name='b' + str(i)))

        with tf.variable_scope(self.name + '_target_layers', reuse=tf.AUTO_REUSE):
            self.layer_target = [self.s_next]
            for i in range(self.n_layer + 1):
                if i < self.n_layer:
                    self.layer_target.append(
                        tf.nn.relu(tf.add(tf.matmul(self.layer_target[i], self.W_target[i]), self.b_target[i])))
                else:
                    self.layer_target.append(
                        tf.add(tf.matmul(self.layer_target[i], self.W_target[i]), self.b_target[i]))

        with tf.variable_scope(self.name + '_loss', reuse=tf.AUTO_REUSE):
            # Choose best action based on main Q-Network
            self.action_next_one_hot = tf.one_hot(tf.argmax(self.layer_main[-1], axis=1), self.n_action)
            self.action_one_hot = tf.one_hot(self.a, self.n_action)
            self.Q_real = tf.reduce_sum(self.layer_target[-1] * self.action_next_one_hot, axis=1) * self.gamma * (
                    tf.ones_like(self.d) - self.d) + self.r
            self.Q_eval = tf.reduce_sum(self.layer_main[-1] * self.action_one_hot, axis=1)
            self.td_error = tf.square(self.Q_eval - self.Q_real)
            self.loss = tf.reduce_mean(self.td_error)

        with tf.variable_scope(self.name + '_op', reuse=tf.AUTO_REUSE):
            self.op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # with tf.variable_scope(self.name + 'target_renew', reuse=tf.AUTO_REUSE):
        #     for i in range(self.n_layer+1):
        #         self.sess.run(self.W_target[i] += self.tau * (self.W_main[i] - self.W_target[i]))
        #         self.sess.run(self.b_target[i] += self.tau * (self.b_main[i] - self.b_target[i]))

        # Try to save and/or reload model
        self.save_path = save_path
        self.saver = tf.train.Saver()
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def act(self, obs, valid, test_mode=False):
        """
        To make an action when received a new observation based on epsilon-greedy policy.

        :param obs: observation
        :param test_mode: if True, will not apply epsilon-greedy policy
        :param valid: to see if this action is valid
        :return: an int indicating corresponding action
        """
        if np.random.uniform() < self.epsilon or test_mode:
            return np.nanargmax(
                self.sess.run(self.layer_main[-1], feed_dict={self.s: obs.reshape([1, self.n_obs])}) * valid)
        else:
            return np.random.choice(np.where(valid == 1)[0])

    def record(self, state: np.ndarray, action: int, reward: float, state_next: np.ndarray, done: bool):
        """
        To record a transition of (s,a,r,s_next,done)  into replay buffer
        """
        transition = np.hstack([state.reshape([1, self.n_obs]), np.array([action, reward, done]).reshape([1, 3]),
                                state_next.reshape([1, self.n_obs])])
        if self.transition_count < self.buffer_size:
            self.buffer_list.append(transition)
        else:
            index = self.transition_count % self.buffer_size
            self.buffer_list[index] = transition
        self.transition_count += 1

    def learn(self):
        """
        To train the model after each transition
        """
        if len(self.buffer_list) <= self.min_buffer_size:
            return
        # Sample from buffer
        sample = np.vstack(random.sample(self.buffer_list, self.batch_size))
        s = sample[:, :self.n_obs]
        a = sample[:, self.n_obs:self.n_obs + 1].flatten()
        r = sample[:, self.n_obs + 1: self.n_obs + 2].flatten()
        d = sample[:, self.n_obs + 2: self.n_obs + 3].flatten()
        s_next = sample[:, self.n_obs + 3:]

        # Train the main DQN and record loss
        self.sess.run(self.op, feed_dict={self.s: s, self.a: a, self.r: r, self.s_next: s_next, self.d: d})

        self.l_history.append(
            self.sess.run(self.loss, feed_dict={self.s: s, self.a: a, self.r: r, self.s_next: s_next, self.d: d}))

        # Count learning step and print current loss,  Increase epsilon
        self.learn_step_count += 1
        if self.learn_step_count % 5 == 0:
            self.epsilon = self.epsilon + self.epsilon_increase if self.epsilon < self.epsilon_max else self.epsilon_max

        # Update Target DQN, another way is to replace target with main Q network after certain step.
        if self.learn_step_count % self.target_change_step == 0:
            for i in range(self.n_layer + 1):
                self.sess.run(self.W_target[i].assign(self.W_main[i].value()))
                self.sess.run(self.b_target[i].assign(self.b_main[i].value()))
        # Another way to update Target Network
        #         for i in range(self.n_layer + 1):
        #             self.sess.run(self.W_target[i].assign(
        #                 self.tau * self.W_main[i].value() + (1-self.tau) * self.W_target[i].value()))
        #             self.sess.run(self.b_target[i].assign(
        #                 self.tau * self.b_main[i].value() + (1-self.tau) * self.b_target[i].value()))
        # self.W_target_value += self.tau * (self.sess.run(self.W_main) - self.W_target_value)
        # self.b_target_value += self.tau * (self.sess.run(self.b_main) - self.b_target_value)

    def save(self):
        """
        To save the model
        """
        if self.save_path is not None:
            self.saver.save(self.sess, self.save_path)
        else:
            print("Save Path needed")

    def plot_cost(self):
        """
        To print the loss change after each training episode
        """
        plt.plot(np.arange(len(self.l_history)), self.l_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
    #
    # def _one_hot(self, state):
    #     one_hot = np.zeros(self.n_obs)
    #     one_hot[state] = 1
    #     return one_hot
