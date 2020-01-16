import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        # gray scale channel
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)

        #128 channel, image isze 19*8 size
        self.fc1 = nn.Linear(128*19*8, 512)

        # action space of 6 in space invaders
        # FYI.. attari has 20 action space
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss() # Target and current predected Q functions
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)




    def forward(self, observation):
        # sequence of frames
        # send network as well as the variables to the device
        observation = T.Tensor(observation).to(self.device)
        # convers height x with x channels to channels to come first ast in conv1
        # 1 channel, 185 x 95 image
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        #observation = observation.view(-1, 128*23*16).to(self.device)
        observation = observation.view(-1, 128*19*8)
        observation = F.relu(self.fc1(observation))

        # Q value for each of the function
        # sequence of frames x numbe of actions(6)
        actions = self.fc2(observation)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemorySize, epsEnd =0.05,
                 replace=10000, actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace

        # agent estimate of current set of states
        self.Q_eval = DeepQNetwork(alpha)

        # agent estimate of successor set of states
        # max value is selected as greedy action
        # this is target policy
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    # random sample state transition is passed manged by batch_size
    def learn(self, batch_size):
        #zero gradiance
        self.Q_eval.optimizer.zero_grad()

        # check if its time to replace target network, if so do it
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        # calculate start of memory sub sampling
        if self.memCntr+batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]

        # numpy array of nd array is convertered to list
        memory = np.array(miniBatch)

        # feedforward current and successive state
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:,3][:])).to(self.Q_eval.device)

        # maximum action for successive state
        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)

        # max action for next successive state
        Qtarget = Qpred

        # value of maximum action
        Qtarget[:,maxA] = rewards + self.GAMMA * T.max(Qnext[1])

        # decrement EPSILION
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        # calculate loss
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
