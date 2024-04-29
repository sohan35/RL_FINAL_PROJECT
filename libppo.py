import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.actor_critic = ActorCritic(input_dim, output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.actor_critic(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        old_probs, values = self.actor_critic(states)
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()

        _, next_values = self.actor_critic(next_states)
        td_targets = rewards + self.gamma * next_values.squeeze(1) * (1 - dones)
        advantages = td_targets - values.squeeze(1)
        
        for _ in range(3):  # Number of gradient descent steps
            probs, values = self.actor_critic(states)
            probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            critic_loss = (advantages ** 2).mean()

            new_probs = probs.clone()  # Create a copy to avoid inplace operation
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy = -(probs * torch.log(probs)).mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Retain the graph
            self.optimizer.step()
            
    def learn(self, total_timesteps, max_episodes, max_steps):
        env = gym.make('CartPole-v1')
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update([state], [action], [reward], [next_state], [done])
                episode_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode + 1}, Reward: {episode_reward}")
                    break    

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Instantiate PPO
ppo_agent = PPO(input_dim=4, output_dim=2)

# Call the learn method
total_timesteps = 10000
max_episodes = 100
max_steps = 1000
ppo_agent.learn(total_timesteps, max_episodes, max_steps)
