import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Policy NN
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):

        super(Policy, self).__init__()

        # Linear functions
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)

        m = Categorical(x)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action, log_prob


# REINFORCE agent
class REINFORCEAgent:
    def __init__(self, obs_space, act_space, lr_actor, gamma):

        self.actor = Policy(obs_space, act_space)
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

    def get_action(self, state):

        state = torch.Tensor(state)
        action, log_prob = self.actor(state)
        return action, log_prob

    def train(self, log_probs, rewards):

        # Calculate discounted reward for trajectory
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        returns = (
            torch.FloatTensor(returns) - torch.mean(torch.FloatTensor(returns))
        ) / (torch.std(torch.FloatTensor(returns)))

        loss_actor = []
        for log_prob, G in zip(log_probs, returns):
            loss_actor.append(-log_prob * G)

        self.optimizer_actor.zero_grad()
        loss_actor = torch.stack(loss_actor).sum()
        loss_actor.backward()
        self.optimizer_actor.step()


def main():

    # Hyperparameters
    max_epochs = 100000
    lr_actor = 1e-4
    gamma = 0.9

    # Environment
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n

    agent = REINFORCEAgent(obs_space, act_space, lr_actor, gamma)

    writer = SummaryWriter()

    # Loop over epochs
    rev_hist = deque([0], maxlen=100)

    for ep in range(max_epochs):

        state, _ = env.reset()

        rewards = []
        logprobs = []

        for _ in range(1000):

            action, log_prob = agent.get_action(state)

            observation, reward, trunc, term, _ = env.step(action.item())

            rewards.append(reward)
            logprobs.append(log_prob)

            state = observation

            # If episode is finished
            if trunc or term:

                # Start training
                agent.train(logprobs, rewards)

                rev_hist.append(sum(rewards))

                avg_rew = sum(rev_hist) / len(rev_hist)

                if ep % 10 == 0:
                    print(
                        f"Epoch: {ep}, Reward: {sum(rewards):.4}, Avg. Reward: {avg_rew:.4}"
                    )

                writer.add_scalar("Reward/last", sum(rewards), ep)
                writer.add_scalar("Reward/avg", avg_rew, ep)

                break
    env.close()


if __name__ == "__main__":
    main()
