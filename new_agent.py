from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'dones'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''
        Save a transition
        '''
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        '''
        Sample a batch of transitions
        '''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQNagent(object):
    def __init__(self, DQN, **kwargs):
        # Define state size
        self.state_size = get_attr(kwargs, 'state_size', 16)
        # Define action size * 2 as action is split into price and quantity
        self.action_size = get_attr(kwargs, 'action_size', 9)
        
        # Define the exploration parameters
        self.epsilon = get_attr(kwargs, 'epsilon', 1.0)
        self.epsilon_decay = get_attr(kwargs, 'epsilon_decay', 0.9995)
        self.epsilon_min = get_attr(kwargs, 'epsilon_min', 0.01)

        # Define replay memory size
        self.memory_size = get_attr(kwargs, 'replay_memory_size', 100000)

        # Define batch_size
        self.batch_size = get_attr(kwargs, 'batch_size', 128)
        
        # Define gamme
        self.gamma = get_attr(kwargs, 'gamma', 0.99)

        self.new_model = get_attr(kwargs, 'new_model', True)
        self.save_model_path = get_attr(kwargs, 'save_mode_path', 'dqn.pth')
        self.load_model_path = get_attr(kwargs, 'load_model_path', 'dqn.pth')


        # Define policy network
        self.dqn = DQN(self.state_size, self.action_size)
        # Define target network
        self.target_dqn = DQN(self.state_size, self.action_size)
        
        if not self.new_model:
            # load model weights
            self.dqn.load_state_dict(torch.load(self.load_model_path))
        
        # Copy the weights from the policy network to the target network
        self.update_target_network()

        # Define replay memory
        self.memory = ReplayMemory(self.memory_size)
        
        # Define the device to use for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn.to(self.device)
        
        
    def select_action(self, state, explore=True):
        # Decay the exploration parameter
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Choose a random action with probability epsilon
        if np.random.rand() < self.epsilon and explore:
            action = np.random.randint(self.action_size)
            
        else:
            with torch.no_grad():
                # Convert the state to a PyTorch tensor
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Compute the Q-values for the state
                q_values = self.dqn(state)
                
                # Compute the index of the action with the highest Q-value for each half
                action = q_values.argmax().item()

        return action
    

    def learn(self, optimizer):
        # Sample a batch of experiences from the replay memory
        experiences = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*experiences))
        states = torch.tensor(np.array(batch.state), dtype=torch.float32)
        actions = torch.tensor(np.array(batch.action), dtype=torch.int64).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32)
        dones = torch.tensor(np.array(batch.dones), dtype=torch.float32)
        

        # Compute the Q-values for the current state-action pairs using the DQN network
        current_q_values = self.dqn(states).gather(1, actions)
        # print(f'current_q_values: {current_q_values.shape}')
        target_actions = self.target_dqn(next_states)
        # print(f'target_actions: {target_actions.shape}')

        # Reshape to have shape (batch_size, 2, single_action_size)
        # target_actions_reshaped = target_actions.view(self.batch_size, 2, self.single_action_size)

        # Take the maximum value along the last dimension (which has size 9)
        # next_q_values, _ = target_actions_reshaped.max(dim=-1)
        next_q_values, _ = target_actions.max(dim=-1)
        next_q_values = next_q_values.unsqueeze(1)
        # print(f'next_q_values: {next_q_values.shape}')
        
        # Compute the Q-values for the next states using the target network
        rewards = rewards.unsqueeze(1)
        # print(f'rewards {rewards.shape}')
        # print((self.gamma * next_q_values).shape)
        # rewards = torch.cat((rewards, rewards), dim=1)
        # Compute the expected Q-values using the Bellman equation
        expected_q_values = rewards + self.gamma * next_q_values
        # print(f'expected_q_values: {expected_q_values.shape}')

        
        # Compute the loss between the current Q-values and the expected Q-values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Update the DQN network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def update_target_network(self):
        # Copy the weights from the DQN network to the target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())   

    
    def save_model(self):
        # save model
        torch.save(self.dqn.state_dict(), self.save_model_path)