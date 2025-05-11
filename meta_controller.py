import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional
import torch.optim as optim  # Import optimizer
import math  # For pi

class MetaController(nn.Module):
    """
    Outputs a distribution over goals in the latent space based on the current state.
    """

    def __init__(self, latent_dim, goal_dim, hidden_dim=256):
        super(MetaController, self).__init__()
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim  # Typically goal_dim == latent_dim

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output layer predicts mean and log variance for the goal distribution
        )
        self.fc_mu = nn.Linear(hidden_dim, goal_dim)
        self.fc_log_var = nn.Linear(hidden_dim, goal_dim)

    def forward(self, z_t):
        """
        Predicts the parameters of the goal distribution.

        Args:
            z_t (torch.Tensor): Current latent state, shape (batch_size, latent_dim).

        Returns:
            tuple: (goal_mu, goal_log_var)
                   goal_mu (torch.Tensor): Mean of the goal distribution, shape (batch_size, goal_dim).
                   goal_log_var (torch.Tensor): Log variance of the goal distribution, shape (batch_size, goal_dim).
        """
        hidden = self.network(z_t)
        goal_mu = self.fc_mu(hidden)
        goal_log_var = self.fc_log_var(hidden)  # Predict log variance for stability
        return goal_mu, goal_log_var

    def sample_goal(self, z_t):
        """
        Samples a goal from the predicted distribution. Also returns the distribution parameters.

        Args:
            z_t (torch.Tensor): Current latent state, shape (batch_size, latent_dim).

        Returns:
            tuple: (goal_sample, goal_mu, goal_log_var)
                   goal_sample (torch.Tensor): Sampled goal, shape (batch_size, goal_dim).
                   goal_mu (torch.Tensor): Mean of the goal distribution.
                   goal_log_var (torch.Tensor): Log variance of the goal distribution.
        """
        goal_mu, goal_log_var = self.forward(z_t)
        std_dev = torch.exp(0.5 * goal_log_var)
        epsilon = torch.randn_like(std_dev)
        goal_sample = goal_mu + std_dev * epsilon
        return goal_sample, goal_mu, goal_log_var

    @staticmethod
    def calculate_pdf(goal_mu, goal_log_var, goal_sample):
        """
        Calculates the probability density of a goal sample under the Gaussian distribution.
        P(g | s) = PDF(g; mu(s), var(s))

        Args:
            goal_mu (torch.Tensor): Mean of the goal distribution, shape (batch_size, goal_dim).
            goal_log_var (torch.Tensor): Log variance of the goal distribution, shape (batch_size, goal_dim).
            goal_sample (torch.Tensor): The sampled goal, shape (batch_size, goal_dim).

        Returns:
            torch.Tensor: Probability density for each item in the batch, shape (batch_size,).
        """
        var = torch.exp(goal_log_var)
        log_prob = -0.5 * (
            goal_log_var + math.log(2 * math.pi) + ((goal_sample - goal_mu) ** 2) / var
        )
        # Sum log probabilities across dimensions and exponentiate for joint PDF
        # Assuming independence across goal dimensions
        pdf = torch.exp(torch.sum(log_prob, dim=-1))
        return pdf