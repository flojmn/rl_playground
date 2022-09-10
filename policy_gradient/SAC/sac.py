import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import copy

# Actor Net
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = nn.Softsign(self.linear3(x))
        return x


# Critic Net
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        x = F.relu(self.linear1(torch.cat([s, a], 1)))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# Replay Buffer
class ReplayBuffer(object):
    def __init__(self, buffersize, state_dim, action_dim):
        self.buffersize = buffersize
        self.count = 0
        self.size = 0
        self.state_buffer = np.zeros((self.buffersize, state_dim))
        self.action_buffer = np.zeros((self.buffersize, action_dim))
        self.reward_buffer = np.zeros((self.buffersize, 1))
        self.nextstate_buffer = np.zeros((self.buffersize, state_dim))
        self.done_buffer = np.zeros((self.buffersize, 1))

    def add(self, state, action, reward, next_state, done):

        self.state_buffer[self.count] = state
        self.action_buffer[self.count] = action
        self.reward_buffer[self.count] = reward
        self.nextstate_buffer[self.count] = next_state
        self.done_buffer[self.count] = done
        self.count = (
            self.count + 1
        ) % self.buffersize  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(
            self.size + 1, self.buffersize
        )  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling

        state_batch = self.state_buffer[index]
        action_batch = self.action_buffer[index]
        reward_batch = self.reward_buffer[index]
        next_state_batch = self.nextstate_buffer[index]
        done_batch = self.done_buffer[index]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class SAC(object):
    def __init__(self, env, params):

        # Gym Enviroment
        self.env = env

        # Hyperparameters
        self.max_epochs = params["max_epochs"]
        self.n_warmup = params["n_warmup"]
        self.lr_actor = params["lr_actor"]
        self.lr_critic = params["lr_critic"]
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch_size"]
        self.hidden_size = params["hidden_size"]

        # State and Action Dimension
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Replay Buffer
        self.memory = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic
        )

        # Path where models are saved
        self.models_path = os.path.join(os.path.dirname(__file__), "models")

    def get_action(self, state):

        # Check if state is tensor, else transform it to a Tensor
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)

        # Forward Pass in Actor Network
        action = self.actor.forward(state)

        # Since Env asks for numpy array as action, transform action to np.array
        # Also remove action from gradient path
        action = action.detach().numpy()

        return action

    def warmup(self):
        # Warmp to improve exploration at the start of training
        for _ in range(self.n_warmup):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                observation, reward, trunc, term, _ = self.env.step(action)
                done = trunc or term
                self.memory.add(state, action, reward, observation, done)
                state = observation

    def train(self):
        # Perform Training
        # Sample a batch
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample(self.batch_size)

    def simulate(self):
        # Rolling out current policy for one epoch
        state = self.env.reset()
        sum_rewards = 0
        done = False
        while not done:
            action = self.get_action(state)
            observation, reward, trunc, term, _ = self.env.step(action)
            sum_rewards += reward
            done = trunc or term
            state = observation

        return sum_rewards

    def save(self):
        # Create Models Folder if it doesnt exist
        os.makedirs(self.models_path, exist_ok=True)

        # Save Nets
        torch.save(self.actor, os.path.join(self.models_path, "actor"))
        torch.save(self.critic, os.path.join(self.models_path, "critic"))
        torch.save(self.critic_target, os.path.join(self.models_path, "critic_target"))

    def load(self):
        # Load Saved Models
        self.actor = torch.load(os.path.join(self.models_path, "actor"))
        self.critic = torch.load(os.path.join(self.models_path, "critic"))
        self.critic_target = torch.load(os.path.join(self.models_path, "critic_target"))


def main():

    # Define Hyperparameters
    params = {
        "max_epochs": 1000,  # Maximum Training Epochs
        "n_warmup": 1000,  # Number of Warmup Steps with random policy
        "lr_actor": 1e-3,  # Actor Learning Rate
        "lr_critic": 1e-3,  # Critc Learning Rate
        "gamma": 0.95,  # Discount Factor
        "tau": 0.05,  # Target Network Update Factor
        "buffer_size": 10000000,  # Total Replay Buffer Size
        "batch_size": 32,  # Batch Size
        "hidden_size": 32,  # Hidden Dim of NN
    }

    # Create Environment
    env = gym.make("Pendulum-v1", g=9.81)

    # Initialize Agent
    agent = SAC(env, params)

    # Initialize Tensorboard
    writer = SummaryWriter()

    # Start Training
    agent.train()


if __name__ == "__main__":
    main()
