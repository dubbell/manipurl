import torch
import torch.nn as nn
from manipurl.models.common import Actor, DoubleCritic
from manipurl.utils.logger import NoLogger
import click

torch.set_float32_matmul_precision('high')
# torch.autograd.set_detect_anomaly(True)

class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, alpha, logp, target_entropy) -> torch.Tensor:
        alpha_loss = -alpha * (logp + target_entropy)
        return alpha_loss.mean()


class ActorLoss(nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()

    def forward(self, alpha, logp, sampled_q) -> torch.Tensor:
        actor_loss = alpha * logp - sampled_q
        return actor_loss.mean()


class SACAgent:
    def __init__(self, obs_dim, act_dim, tau = 0.005, gamma = 0.99, lr = 3e-4, logger = NoLogger()):
        self.tau = tau
        self.gamma = gamma
        self.logger = logger

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            click.echo("cuda not available")

        self.actor = Actor(obs_dim, act_dim).to(device=self.device)
        self.critic = DoubleCritic(obs_dim + act_dim).to(device=self.device)
        self.target_critic = DoubleCritic(obs_dim + act_dim).to(device=self.device)

        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))

        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters()), lr=lr)

        self.alpha_loss_func = AlphaLoss()
        self.target_entropy = -act_dim

        self.actor_loss_func = ActorLoss()
    

    @torch.compile
    def _compiled_sample_action(self, state):
        _, sampled_action, _ = self.actor.forward(state)
        return sampled_action.squeeze()

    def sample_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self._compiled_sample_action(state).detach()


    @torch.compile
    def _compute_critic_update(self, states, actions, rewards, next_states, done):
        q1, q2 = self.critic.forward(states, actions)

        _, next_actions, next_logp = self.actor.forward(next_states)

        with torch.no_grad():
            next_q1, next_q2 = self.target_critic.forward(next_states, next_actions)
            next_q = torch.minimum(next_q1, next_q2) - torch.exp(self.log_alpha) * next_logp
            target_q = rewards + self.gamma * (1 - done) * next_q

        critic_loss1 = nn.functional.mse_loss(q1, target_q)
        critic_loss2 = nn.functional.mse_loss(q2, target_q)

        critic_loss = (critic_loss1 + critic_loss2).mean()

        return critic_loss
    
    @torch.compile
    def _compute_actor_update(self, states):
        _, actions, actor_logp = self.actor.forward(states)

        alpha = torch.exp(self.log_alpha)
        alpha_loss = self.alpha_loss_func.forward(alpha, actor_logp.detach(), self.target_entropy)

        q1, q2 = self.critic.forward(states, actions)
        sampled_q = torch.minimum(q1, q2)
        actor_loss = self.actor_loss_func.forward(alpha.detach(), actor_logp, sampled_q)

        return actor_loss, alpha_loss, actor_logp.mean(), sampled_q.mean()

    
    def train_step(self, batch):
        critic_loss = self._compute_critic_update(batch.states, batch.actions, batch.rewards, batch.next_states, batch.done)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss, alpha_loss, actor_logp, q = self._compute_actor_update(batch.states)

        self.log_alpha_opt.zero_grad()
        self.actor_opt.zero_grad()

        (alpha_loss + actor_loss).backward()

        self.log_alpha_opt.step()
        self.actor_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.logger.log_metrics({
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "actor_logp": actor_logp.detach(),
            "log_alpha": self.log_alpha.detach(),
            "q": q.detach()})