import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional
import torch.optim as optim  # Import optimizer
import math  # For pi

class AleatoricPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim, rnn_hidden_dim=256):
        super(AleatoricPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # RNN layer (LSTM)
        # Input size is latent_dim + action_dim
        self.rnn = nn.LSTM(latent_dim + action_dim, rnn_hidden_dim, batch_first=True)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(rnn_hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(rnn_hidden_dim, latent_dim)

    def forward(self, z_t, a_t, hidden_state=None):
        """
        Predicts the parameters of the next latent state distribution.

        Args:
            z_t (torch.Tensor): Current latent state, shape (batch_size, latent_dim).
            a_t (torch.Tensor): Action taken at time t, shape (batch_size, action_dim).
            hidden_state (tuple, optional): Previous hidden state of the RNN. Defaults to None.

        Returns:
            tuple: (mu, log_var, next_hidden_state)
                   mu (torch.Tensor): Mean of the predicted next latent state, shape (batch_size, latent_dim).
                   log_var (torch.Tensor): Log variance of the predicted next latent state, shape (batch_size, latent_dim).
                   next_hidden_state (tuple): The updated hidden state of the RNN.
        """
        # Concatenate latent state and action
        # z_t shape: (batch_size, latent_dim) -> (batch_size, 1, latent_dim)
        # a_t shape: (batch_size, action_dim) -> (batch_size, 1, action_dim)
        rnn_input = torch.cat((z_t.unsqueeze(1), a_t.unsqueeze(1)), dim=-1)

        # Pass through RNN
        # rnn_output shape: (batch_size, 1, rnn_hidden_dim)
        rnn_output, next_hidden_state = self.rnn(rnn_input, hidden_state)

        # Squeeze the sequence dimension
        # rnn_output shape: (batch_size, rnn_hidden_dim)
        rnn_output_squeezed = rnn_output.squeeze(1)

        # Predict mu and log_var
        mu = self.fc_mu(rnn_output_squeezed)
        log_var = self.fc_log_var(
            rnn_output_squeezed
        )  # Predict log variance for stability

        return mu, log_var, next_hidden_state


def nll_loss(mu, log_var, z_target):
    """
    Calculates the Negative Log-Likelihood loss for a Gaussian distribution.

    Args:
        mu (torch.Tensor): Predicted mean, shape (batch_size, latent_dim).
        log_var (torch.Tensor): Predicted log variance, shape (batch_size, latent_dim).
        z_target (torch.Tensor): Target latent state, shape (batch_size, latent_dim).

    Returns:
        torch.Tensor: The calculated NLL loss (scalar).
    """
    var = torch.exp(log_var)
    loss_per_dim = 0.5 * (
        log_var + math.log(2 * math.pi) + ((z_target - mu) ** 2) / var
    )
    # Sum over the latent dimensions and average over the batch
    loss = torch.mean(torch.sum(loss_per_dim, dim=1))
    return loss