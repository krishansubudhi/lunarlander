import numpy as np
class QLearningAgent:
    def __init__(self):
        self.reset()
    def reset(self):
        self.episodes = 0
        self.gamma = 1
        self.alpha = 0.1
        self.epsilon = 0.1
        self.training = False
        self.rewardsThisEpisode = 0
        self.rewardsAllEpisodes = []
    def train(self, epsilon = 0.1):
        self.training = True
        self.epsilon = epsilon
        self.episodes += 1
        self.rewardsThisEpisode = 0
    def evaluate(self):
        self.training = False
        self.epsilon = 0
    def getAction(self, state, actions):
        '''Get action based on policy. Use exploration while training
        '''
        explore = flipCoin(self.epsilon)
        if explore:
            return np.random.choice(actions)
        else:
            self.getBestAction(state,actions)
    def getBestAction(self,state, actions):
        raise NotImplemented()
    def update(self, state, action, reward):
        if not self.training:
            raise Exception('Update called. But training finished. Call train()')
            _update(state, action, reward)
    def _update(self, state, action, reward):
        raise NotImplemented()
    def finishEpisode(self):
        self.rewardsAllEpisodes.append(self.rewardsThisEpisode)
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
        self.weights = np.zeros(shape = (num_actions, num_features ))#+ 1
    def getFeatures(self, state, action):
        '''transforms the input features. 
            Example: add a bias feature
        '''
        return state
    def getWeights(self, action):
        return self.weights[action]
    def getBestAction(self, state, actions):
        '''Given a list of actions, return the best action as per current policy.
        '''
        qValues = [
            (self.getQValue(state, action) ,action)
            for action in actions
        ]
        return np.argmax(qValues)[1]
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        weights = self.getWeights(action)
        return np.dot(features, weights)
    def update(self, state, action, reward, next_state):
        '''Update Q based on the actual reward vs estimated reward.
        Remember - we don't have any right or wrong labels
        '''
        next_state_maxQ = self.getQValue(next_state, 
            self.getBestAction(next_state, range(self.num_actions)))
        
        rewardBasedQ = reward + self.gamma * next_state_maxQ
        estimatedQ = self.getQValue(state, action)
        diff = rewardBasedQ - estimatedQ
        features = self.getFeatures(state,action)
        weights = self.getWeights(action)
        weights += diff * features
        
        