import unittest
from .. import agents
import numpy as np

class ApproximateQLearningAgentTest(unittest.TestCase):
    '''Class to test all functionalities of ../agents.py
    '''
    def setUp(self) -> None:
        super().setUp()
        self.num_features = 3
        self.num_actions = 3
        self.actions = list(range(self.num_actions))
        self.agent = agents.ApproximateQLearningAgent(
            self.num_features, self.num_actions)
    def test_initializtion(self):
        assert(all(self.agent.getWeights(0) == 0))
        assert(self.agent.training == False)
        self.assertEqual(
            self.agent.weights.shape,
            (self.num_actions, self.num_features)
        )
    def test_initialQValue(self):
        random_state = np.ones(self.num_features)
        #initial Q
        assert self.agent.getQValue(random_state, 0) == 0
        #update weights
    def testUpdate(self):
        random_state = np.ones(self.num_features)
        with self.assertRaises(
            agents.UpdateWhileNotTrainingException):
            self.agent.update(random_state,0, 0, random_state)
        
        self.agent.train()
        assert self.agent.getQValue(random_state, 2) ==  0
        self.agent.update(random_state, action = 2, 
            reward = 100, next_state = random_state)
        expected_qvalue = 30 # self.agent.alpha * 100 * sum(random_state)
        assert self.agent.getQValue(random_state, 2) == expected_qvalue
        
        # check if action 3 is returned because of the positive reward
        action = self.agent.getBestAction(random_state, self.actions)
        assert action ==2

    def test_str(self):
        s = str(self.agent)
        assert s == str(np.zeros((self.num_actions, self.num_features)))
    
    def test_exploration(self):
        random_state = np.ones(self.num_features)
        self.agent.train = False
        actions = [self.agent.getAction(random_state, self.actions) for _ in range(100)]
        assert all(actions == actions[0])

        self.agent.train = True
        actions = [self.agent.getAction(random_state, self.actions) for _ in range(100)]
        assert not all(actions == actions[0])