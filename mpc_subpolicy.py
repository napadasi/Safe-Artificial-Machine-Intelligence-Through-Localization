import torch

class MPCSubPolicy:
    """Selects actions using Model Predictive Control to reach a goal."""

    def __init__(
        self,
        world_model,
        action_dim,
        horizon=12,
        num_samples=500,
        variance_penalty=0.1,
        action_penalty=0.01,
        variance_threshold=10.0,
        device="cpu",
    ):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.variance_penalty = variance_penalty
        self.action_penalty = action_penalty
        self.variance_threshold = variance_threshold
        self.device = device

    @torch.no_grad()
    def select_action(self, current_z, goal_z, current_hidden_state=None):
        """
        Selects the best action using MPC and estimates uncertainty (avg variance).

        Args:
            current_z (torch.Tensor): Current latent state, shape (1, latent_dim).
            goal_z (torch.Tensor): Goal latent state, shape (1, goal_dim).
            current_hidden_state (tuple, optional): Current RNN hidden state.

        Returns:
            tuple: (best_first_action, best_first_hidden_state, avg_variance_of_best)
                   best_first_action (torch.Tensor): Shape (1, action_dim).
                   best_first_hidden_state (tuple): Hidden state after the first step.
                   avg_variance_of_best (float): Average variance along the best path, or float('inf') if invalid.
        """
        batch_size, latent_dim = current_z.shape
        assert batch_size == 1, "MPC currently supports batch size 1 for planning."

        action_sequences = torch.randn(
            self.num_samples, self.horizon, self.action_dim, device=self.device
        )
        total_costs = torch.full(
            (self.num_samples,), float("inf"), device=self.device
        )  # Initialize costs to infinity
        first_step_hidden_states = [None] * self.num_samples
        sequence_avg_variances = torch.full(
            (self.num_samples,), float("inf"), device=self.device
        )  # Store avg variance per sequence

        for i in range(self.num_samples):
            z = current_z
            hidden = current_hidden_state
            sequence_cost = 0.0
            sequence_variances = []  # Store variances for this sequence

            for t in range(self.horizon):
                action = action_sequences[i : i + 1, t, :]
                mu_next, log_var_next, hidden_next = self.world_model(z, action, hidden)
                var_next = torch.exp(log_var_next)

                if torch.any(var_next > self.variance_threshold):
                    sequence_cost = float("inf")
                    sequence_variances = [
                        float("inf")
                    ]  # Mark sequence variance as infinite
                    break  # Stop evaluating this invalid sequence

                sequence_variances.append(
                    torch.mean(var_next).item()
                )  # Store mean variance of this step

                if t == 0:
                    first_step_hidden_states[i] = hidden_next

                dist_cost = torch.sum((mu_next - goal_z) ** 2)
                var_cost = self.variance_penalty * torch.sum(var_next)
                act_cost = self.action_penalty * torch.sum(action**2)
                step_cost = dist_cost + var_cost + act_cost
                sequence_cost += step_cost

                z = mu_next
                hidden = hidden_next

            # Only update cost and variance if the sequence completed without exceeding threshold
            if sequence_cost != float("inf"):
                total_costs[i] = sequence_cost
                if (
                    sequence_variances
                ):  # Avoid division by zero if horizon is 0 or sequence broke early
                    sequence_avg_variances[i] = sum(sequence_variances) / len(
                        sequence_variances
                    )
                else:
                    sequence_avg_variances[i] = 0.0  # Or handle as appropriate

        if torch.all(total_costs == float("inf")):
            print(
                "Warning: MPC found no valid action sequence below variance threshold."
            )
            best_sequence_index = (
                0  # Fallback: choose the first (likely invalid) sequence
            )
            avg_variance_of_best = float("inf")
        else:
            best_sequence_index = torch.argmin(total_costs)
            avg_variance_of_best = sequence_avg_variances[best_sequence_index].item()

        best_first_action = action_sequences[best_sequence_index, 0, :].unsqueeze(0)
        best_first_hidden_state = first_step_hidden_states[best_sequence_index]

        # Handle case where the chosen sequence might have been invalid (cost=inf),
        # ensure hidden state is somewhat reasonable if possible.
        if (
            best_first_hidden_state is None
            and current_hidden_state is not None
            and total_costs[best_sequence_index] == float("inf")
        ):
            # If the best (or fallback) sequence was invalid from step 0,
            # maybe return the initial hidden state? Or None? Returning None for now.
            pass

        return best_first_action, best_first_hidden_state, avg_variance_of_best