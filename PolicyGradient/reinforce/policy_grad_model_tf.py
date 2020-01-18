#Policy gradient method:
#Its the probability the agent uses to pick actions. Deep Neural network is used to approximate the agents policy.
#The network takes observations of the environment as input and outputs actions selected based on softmax  action
#function. It generate an episode and keep track of SAR in agent's memory. At the end of episode, go back through
#SAR and compute the discounted future returns at each time step. Use these returns as weights and actions the
#agent took as labels to perform back propagation and update the weight of the deep RL. Rise and repeat.
#Weight is updated with some factor of learning rate times gradient of the performance.  As we would like to
#improve the performance over time, DRL uses gradient assent rather than gradient decent. Gradient of this
#performance metrics is promotional to overstates for the amount of time spent in a given state and sum over
#actions for the value of the state action pairs and the gradient of the policy.  Policy is the probability of
#tacking each action given some state.
#Policy Gradient work by approximating policy of the gradient. Policy is the probability distribution that the
#agent uses to select actions. Deep RL is used to approximate the probability distribution and feed in the input
#observations and getting out the probability distribution as output. The agent learnings by replying the memory
#of the rewards it received during the episodes and calculating the discounted future rewards that followed each
#particular time step. These discounted future rewards acts as weights in the update of the deep neural network so
#that the agent assigns a higher probability to actions whose future rewards are higher.
import os
import numpy as np
import tensorflow as tf

class PolicyGradientAgent(object):
    def __init__(self, lr, gamma, n_actions=4, l1_size=64, l2_size=54,
                 input_dims=8, chkpt_dir='tmp'):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.input_dims = input_dims
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.chkpt_file = os.path.join(chkpt_dir, 'policy.ckpt')

    def build_net(self):
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_dims],
                                    name='input')
        # actions the agents took. Will be used to calculate loss function
        self.label = tf.placeholder(tf.int32, shape=[None,], name='actions')
        # future reward
        self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

        with tf.variable_scope('layers'):
            l1 = tf.layers.dense(inputs=self.input, units=self.l1_size,
                                 activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            l2 = tf.layers.dense(inputs=l1, units=self.l2_size,
                                 activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            l3 = tf.layers.dense(inputs=l2, units=self.n_actions,
                                 activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.actions = tf.nn.softmax(l3, name='probabilities')

        with tf.variable_scope('loss'):
            negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=l3, labels=self.label)
            loss = negative_log_probability * self.G

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.sess.run(self.actions,
                                        feed_dict={self.input:state})[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transitions(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        _ = self.sess.run(self.train_op, feed_dict={self.input: state_memory,
                                                 self.label: action_memory,
                                                 self.G: G})

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_checkpoint(self):
        print('loading checkpoint')
        self.saver.restore(self.sess, self.chkpt_file)

    def save_checkpoint(self):
        self.saver.save(self.sess, self.chkpt_file)
