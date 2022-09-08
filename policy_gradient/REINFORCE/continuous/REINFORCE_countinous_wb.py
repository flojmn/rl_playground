import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Policy NN
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):

        super(Policy, self).__init__()

        # Linear functions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc2_ = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        mean = self.fc2(x)
        log_std = self.fc2_(x)
        log_std = torch.clamp(log_std, min=-2, max=2)

        std = log_std.exp()

        return mean, std


# Baseline NN
class Value(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):

        super(Value, self).__init__()

        # Linear functions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# REINFORCE agent
class REINFORCEAgent:
    def __init__(self, obs_space, act_space, lr_actor, lr_value, gamma, eps):

        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.eps = eps
        self.actor = Policy(obs_space, act_space)
        self.value = Value(obs_space)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.lr_value)

    def get_action(self, state):

        state = torch.FloatTensor(state)
        mean, std = self.actor(state)
        m = Normal(mean, std + self.eps)
        action = m.sample()
        log_prob = m.log_prob(action)
        log_prob = log_prob.sum()
        action = F.softsign(action)
        action = action.numpy()

        return action, log_prob

    def train(self, observations, log_probs, rewards):

        # Calculate discounted reward for trajectory
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)

        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)

        # Estimate Value as Baseline
        for _ in range(2):
            value_pred = []
            for obs in observations:
                obs = torch.FloatTensor(obs)
                value_pred.append(self.value(obs))

            value_pred = torch.stack(value_pred).squeeze()
            loss_value = F.mse_loss(value_pred, returns)
            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

        loss_actor = []
        for log_prob, G, V in zip(log_probs, returns, value_pred):
            loss_actor.append(-log_prob * (G - V.detach()))

        loss_actor = torch.stack(loss_actor).sum()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        return loss_actor, loss_value


def main():

    # Hyperparameters
    max_epochs = 100000
    learning_rate_actor = 1e-3
    learning_rate_value = 1e-3
    gamma = 0.99

    eps = np.finfo(np.float32).eps.item()

    # Environment
    env = gym.make("LunarLander-v2", continuous=True, enable_wind=False)

    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.shape[0]

    agent = REINFORCEAgent(
        obs_space, act_space, learning_rate_actor, learning_rate_value, gamma, eps
    )

    writer = SummaryWriter()

    # Loop over epochs
    rev_hist = deque([0], maxlen=100)

    for ep in range(max_epochs):

        state, _ = env.reset()

        observations = []
        logprobs = []
        rewards = []

        for _ in range(1000):

            action, log_prob = agent.get_action(state)
            observation, reward, trunc, term, _ = env.step(action)

            observations.append(state)
            logprobs.append(log_prob)
            rewards.append(reward)

            state = observation

            # If episode is finished
            if trunc or term:

                # Start training
                loss_actor, loss_value = agent.train(observations, logprobs, rewards)

                rev_hist.append(sum(rewards))

                avg_rew = sum(rev_hist) / len(rev_hist)

                if ep % 10 == 0:
                    print(
                        f"Epoch: {ep}, Reward: {sum(rewards):.4}, Avg. Reward: {avg_rew:.4}"
                    )

                writer.add_scalar("Reward/last", sum(rewards), ep)
                writer.add_scalar("Reward/avg", avg_rew, ep)
                writer.add_scalar("Loss/actor", loss_actor, ep)
                writer.add_scalar("Loss/value", loss_value, ep)

                break

    env.close()


if __name__ == "__main__":
    main()
