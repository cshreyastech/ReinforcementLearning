import os
import numpy as np
import tensorflow as tf

# Deep Q network, it takes batch of inputs from the environmentm, "Break Out" in this case
# pass it through neural network to do feature selection and pass it through fully connected
# layer to determine the value of each given action and uses maxmimum value of next action
# to determine its loss function and perform training on that network via backpropagation
class DeepQNetwork(object):
    # learning rate, number of actions, name of the network(select the action and value of the action),
    # dimentions of fully connected layer,
    # input dimentions of the environment (attari gym 210 by 160 resolution, take 4 frames to get sense of motion.)
    # directory to save model
    def __init__(self, lr, n_actions, name, fc1_dims=1024,
                 input_dims=(210,160,4), chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network() # add to graph
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'deepqnet.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)# keep track of trainable variables for the network
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='action_taken')
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name='q_value')

            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
                                     kernel_size=(8,8), strides=4, name='conv1',
                     kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)


            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                     kernel_size=(4,4), strides=2, name='conv2',
                      kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv2_activated = tf.nn.relu(conv2)


            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128,
                                     kernel_size=(3,3),strides=1, name='conv3',
                      kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)

            # get the output of convolution network, flatten them and pass it though dense network
            # get Q values or value of each state, action pai
            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims,
                                     activation=tf.nn.relu,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2))

            # state action pair which is the output of the neural network
            # it has one output for each action
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2))

            #self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))

            # actual value of Q for each network
            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def load_checkpoint(self):
        print('... loading checkpoing ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('... saving checkpoing ...')
        self.saver.save(self.sess, self.checkpoint_file)

class Agent(object):
    # alpha - learning rate
    # gamma - discount factor
    # epsilion - how often to take random action
    # tmp/q_next - directory to save q_next network - action to take
    # tmp/q_eval - directory to save q_eval network - value of that action
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=5000, input_dims=(210,160,4),
                 q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
        self.n_actions = n_actions
        self.action_space = [ i for i in range(self.n_actions)]
        self.gamma = gamma
        self.mem_size = mem_size

        # number of memory that has been stored
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        # Network to tell value of next action
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=q_next_dir)
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                  name='q_eval', chkpt_dir=q_eval_dir)

        # transition of four frames stacked four frames by number of memeories
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))

        # store one hot encoding of actions
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)

        self.reward = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)

        # saves memory of done flags. Takes around 48G
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state

        # actions would be one hot encoded
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    # way to choose action
    # epsilon - how often to choose a random action and decay it over time
    # it takes greedy action over time. that is take action which has highiest
    # reward for next state
    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # greedy action. Find next highest valued action.
            # use evaluation network. Use current state as Q evaluation network input
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                        feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        return action

    # learning:
    # 1) Check and update value of target network as needed
    # 2) Select batch of non-sequencial random memories to aviod ossilations in parameter space
    # 3) Calculate the value of current action and nex maximum action.
    # 4) Plug them in Bellman equation for Q learning algorithm
    # 5) Run update function on the loss
    def learn(self):
        # check for replacing target newtwork and replace if needed
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        # find where the memory ends
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        # randomly sample subset of the memory
        batch = np.random.choice(max_mem, self.batch_size)
        # get state and action transitions for that batch
        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        # As these are One-Hot encoded, these would have to turned ino integer encoding
        action_values = np.array([0, 1, 2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        # calculate the values of current states
        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                        feed_dict={self.q_eval.input: state_batch})

        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                        feed_dict={self.q_next.input: new_state_batch})

        # create a copy as we want the loss for all the non-optimal actions to be zero
        q_target = q_eval.copy()
        idx = np.arange(self.batch_size)
        # Calculate the value for Q trarget so the states in the batch for the action
        # taken have the maximum value of the next state is multipled by terminal quantity.
        # Reason: If the next state is end of he episode, we get the reward if not,
        # the future reward is taken into account.
        q_target[idx, action_indices] = reward_batch + \
                                self.gamma*np.max(q_next, axis=1)*terminal_batch

        #feed it through the neural network
        _ = self.q_eval.sess.run(self.q_eval.train_op,
                                feed_dict={self.q_eval.input: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target})
        # decrese eqsilon over time
        if self.mem_cntr > 100000:
            if self.epsilon > 0.01:
                self.exsilon *= 0.9999999
            elif self.epsilon <= 0.01:
                self.epsilon = 0.01

        #save model
        def save_models(self):
            self.q_eval.save_checkpoint()
            self.q_next.save_checkpoint()

        #load model
        def load_models(self):
            self.q_eval.load_checkpoint()
            self.q_next.load_checkpoint()

        #update graph to copy the evaluation network to target network
        def update_graph(self):
            # take target parameters and perform copy operation on them
            # pass in a session by passing the sesssion for the value
            # we try to copy from and not copy to. So with Q next we get an error
            for t, e in zip(t_params, e_params):
                self.q_eval.sess.run(tf.assign(t, e))
