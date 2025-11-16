import torch
import torch.nn as nn
from common import Actor, DoubleCritic



class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, alpha, logp, target_entropy):
        alpha_loss = -alpha * (logp + target_entropy)
        return alpha_loss.mean()


class ActorLoss(nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()

    def forward(self, alpha, logp, sampled_q):
        actor_loss = alpha * logp - sampled_q
        return actor_loss.mean()


class SACAgent:
    def __init__(self, obs_dim, act_dim, tau = 0.005, gamma = 0.99, lr = 3e-4):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = DoubleCritic(obs_dim)

        self.log_alpha = nn.Parameter(torch.tensor(0.0))

        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters()), lr=lr)

        self.alpha_loss_func = AlphaLoss()
        self.target_entropy = -act_dim

        self.actor_loss_func = ActorLoss()

    
    def actor_alpha_train_step(self, batch):
        self.actor_opt.zero_grad()
        self.log_alpha_opt.zero_grad()

        actions, logp = self.actor.forward(batch.states)

        alpha = torch.exp(self.log_alpha)
        alpha_loss = self.alpha_loss_func.forward(alpha, logp.detach(), self.target_entropy)

        q1, q2 = self.critic.forward(batch.states, actions)
        sampled_q = torch.minimum(q1, q2)
        actor_loss = self.actor_loss_func.forward(alpha.detach(), logp, sampled_q)

        total_loss = alpha_loss + actor_loss
        total_loss.backward()

        self.log_alpha_opt.step()
        self.actor_opt.step()


    def critic_train_step(self, batch):
        self.critic_opt.zero_grad()

        q1, q2 = self.critic.forward(batch.states, batch.actions)
        
        next_actions, next_logp = self.actor.forward(batch.next_states, batch.next_actions)

        with torch.no_grad():
            next_q1, next_q2 = self.target_critic.forward(batch.next_states, next_actions)
            next_q = torch.minimum(next_q1, next_q2) - torch.exp(self.log_alpha) * next_logp
            target_q = batch.rewards + self.gamma * (1 - batch.done) * next_q

        critic_loss1 = nn.functional.mse_loss(q1, target_q)
        critic_loss2 = nn.functional.mse_loss(q2, target_q)

        critic_loss = (critic_loss1 + critic_loss2).mean()

        critic_loss.backward()
        self.critic_opt.step()

        for param, target_param in zip(self.critic.parameters, self.target_critic.parameters()):
            target_param.data.copy(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        