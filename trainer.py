from gym_env import TradingEnv
from agent import DQNagent
from utils.misc import *
import torch.optim as optim
from tqdm.notebook import tqdm

class Trainer(object):

    def __init__(self, agent, **kwargs):
        
        self.params = kwargs
        # define parameters
        self.num_episodes = get_attr(kwargs, 'num_episodes', 50)
        self.batch_size = get_attr(kwargs, 'batch_size', 200)
        self.record_var_list = get_attr(kwargs, 'record_var_list', None)
        self.learning_rate = get_attr(kwargs, 'learning_rate', 0.001)
        self.target_update_threshold = get_attr(kwargs, 'target_update_threshold', 10)

        self.agent = agent
        
        self.optimizer = optim.Adam(self.agent.dqn.parameters(), lr=self.learning_rate)

    def train(self, df, display=False, save_model=True):
        self.env = TradingEnv(df = df, **self.params)
        state = self.env.reset()
        done = False
        episode = 0

        self.record_list = []
        self.trade_record_list = []

        for episode in tqdm(range(self.num_episodes)):
            while not done:
                # Select an action using the DQN agent
                action1, action2 = self.agent.select_action(state)

                # Take the action in the environment
                next_state, reward, done, info = self.env.step([action1, action2])
                # if reward == 0: print(next_state, reward, done, info)

                # Add the experience to the replay memory
                self.agent.memory.push(state, [action1, action2], next_state, reward, done)

                # Move to the next state
                state = next_state

                # Update the DQN agent
                if len(self.agent.memory) > self.batch_size:
                    self.agent.learn(self.optimizer)
            
            if episode % self.target_update_threshold == 0:
                self.agent.update_target_network()

            self.record_list.append(self.env.record)
            self.trade_record_list.append(self.env.trade_record)

            state = self.env.reset()
            done = False
        
        self.agent.save_model()
        
        return self.record_list, self.trade_record_list
    
    def test(self, df):
        self.env = TradingEnv(df = df, **self.params)
        state = self.env.reset()
        done = False
        while not done:
            # Select an action using the DQN agent
            action1, action2 = self.agent.select_action(state, explore=False)

            # Take the action in the environment
            next_state, reward, done, info = self.env.step([action1, action2])
            # if reward == 0: print(next_state, reward, done, info)

            # Add the experience to the replay memory
            self.agent.memory.push(state, [action1, action2], next_state, reward, done)

            # Move to the next state
            state = next_state
        
        return self.env

            



