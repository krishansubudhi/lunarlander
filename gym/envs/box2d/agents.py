import numpy as np
class UpdateWhileNotTrainingException(Exception):
    pass
class QLearningAgent:
    def __init__(self):
        self.reset()
    def reset(self):
        self.episodes = 0
        self.gamma = 1
        self.alpha = 0.1
        self.epsilon = 0.1
        self.training = False
        self.totalRewardsEp = 0
        self.exploredEp = 0
        self.actionsEp = 0
    def train(self, epsilon = 0.1):
        if self.training:
            return
        self.training = True
        self.epsilon = epsilon
        self.episodes += 1
        self.totalRewardsEp = 0
        self.exploredEp = 0
        self.actionsEp = 0
    def evaluate(self):
        self.training = False
        self.epsilon = 0
    def getAction(self, state, actions):
        '''Get action based on policy. Use exploration while training
        '''
        self.actionsEp += 1
        explore = flipCoin(self.epsilon)
        if explore:
            self.exploredEp+= 1
            return np.random.choice(actions)
        else:
            return self.getBestAction(state,actions)
    def getBestAction(self,state, actions):
        raise NotImplemented()
    def update(self, state, action, reward, next_state):
        self.totalRewardsEp += reward
        if not self.training:
            raise UpdateWhileNotTrainingException('Update called. But training finished. Call train()')
        self._update(state, action, reward, next_state)
    def _update(self, state, action, reward):
        raise NotImplemented()
    def getStats(self):
        stat = {'actions' : self.actionsEp,
            'explored' : self.exploredEp,
            'total rewards' : self.totalRewardsEp
        }
        return stat
    def finishEpisode(self):
        stat = self.getStats()
        print('Episode Stats = ',stat)
        self.training = False
def flipCoin(p):
    '''flip a coin with prob p of returning True
    '''
    r = np.random.random()
    return r < p

class ApproximateQLearningAgent(QLearningAgent):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        #add 1 for bias. Remember to add a bias feature to the state
        self.weights = np.zeros(shape = (num_actions, num_features + 1 ))#+ 1
        self.qValSumEp = 0
        self.minQ = -200
    def train(self, epsilon=0.1):
        return super().train(epsilon)
        self.qValSum = 0
    def getFeatures(self, state, action):
        '''transforms the input features. 
            Example: add a bias feature
        '''
        return np.append(state,1)
    def getWeights(self, action):
        return self.weights[action]
    def getBestAction(self, state, actions):
        '''Given a list of actions, return the best action as per current policy.
        '''
        qValues = [self.getQValue(state, action)
            for action in actions
        ]
        self.qValSumEp +=np.max(qValues)
        return actions[np.argmax(qValues)]
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        weights = self.getWeights(action)
        return np.dot(features, weights)
    def _update(self, state, action, reward, next_state):
        '''Update Q based on the actual reward vs estimated reward.
        Remember - we don't have any right or wrong labels
        '''
        next_state_maxQ = self.getQValue(next_state, 
            self.getBestAction(next_state, range(self.num_actions)))
        
        # reward for moving close to target
        if action == 2:
            reward += 0.3
            # m_power * 0.30
        # reward -= s_power * 0.03

        rewardBasedQ = reward + self.gamma * next_state_maxQ
        estimatedQ = self.getQValue(state, action)
        diff = rewardBasedQ - estimatedQ
        features = self.getFeatures(state,action)
        weights = self.getWeights(action)
        weights += self.alpha * diff * features
        weights = np.clip(weights, -100, 100)
    def getStats(self):
        stat = super().getStats()
        stat['averageQVal'] = self.qValSumEp/self.actionsEp
        return stat
    def __str__(self):
        return str(self.weights)