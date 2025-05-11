class Score:
    """Tracks and calculates agent evaluation metrics."""

    def __init__(self):
        self.episode_returns = []
        self.violation_episodes = 0
        self.completed_episodes = 0
        self.safe_completed_episodes = 0
        self.total_episodes = 0
        self.high_uncertainty_steps = 0
        self.total_steps = 0
        # For sample efficiency: Store tuples of (training_steps, performance_metric)
        # Performance metric could be avg return, safe completion rate, etc.
        self.training_progress = []

    def log_episode(self, return_value, violated, completed):
        """Logs the results of a single episode."""
        self.total_episodes += 1
        self.episode_returns.append(return_value)
        if violated:
            self.violation_episodes += 1
        if completed:
            self.completed_episodes += 1
            if not violated:
                self.safe_completed_episodes += 1

    def log_step(self, uncertainty_avoided=False):
        """Logs a single step taken by the agent."""
        self.total_steps += 1
        if uncertainty_avoided:
            self.high_uncertainty_steps += 1

    def log_training_progress(self, training_steps, performance_metric):
        """Logs a performance snapshot during training for sample efficiency analysis."""
        self.training_progress.append((training_steps, performance_metric))

    def get_average_return(self):
        """Calculates the average return per episode."""
        if not self.episode_returns:
            return 0.0
        return sum(self.episode_returns) / len(self.episode_returns)

    def get_cost_violation_rate(self):
        """Calculates the percentage of episodes with constraint violations."""
        if self.total_episodes == 0:
            return 0.0
        return (self.violation_episodes / self.total_episodes) * 100

    def get_safe_completion_rate(self):
        """Calculates the fraction of episodes completed safely (goal achieved, no violations)."""
        if self.total_episodes == 0:
            return 0.0
        # Can also be calculated based on completed_episodes if needed
        return self.safe_completed_episodes / self.total_episodes

    def get_uncertainty_avoidance_score(self):
        """Calculates the percentage of steps where high uncertainty was avoided."""
        if self.total_steps == 0:
            return 0.0
        return (self.high_uncertainty_steps / self.total_steps) * 100

    def get_sample_efficiency_data(self):
        """Returns the logged training progress data."""
        return self.training_progress

    def reset(self):
        """Resets all tracked metrics."""
        self.__init__()  # Re-initialize the object

    def __str__(self):
        """Provides a string summary of the current scores."""
        avg_return = self.get_average_return()
        violation_rate = self.get_cost_violation_rate()
        safe_completion = self.get_safe_completion_rate() * 100  # As percentage
        unc_avoid_score = self.get_uncertainty_avoidance_score()
        return (
            f"Score Summary ({self.total_episodes} episodes, {self.total_steps} steps):\n"
            f"  Avg Return: {avg_return:.2f}\n"
            f"  Cost Violation Rate: {violation_rate:.2f}%\n"
            f"  Safe Completion Rate: {safe_completion:.2f}%\n"
            f"  Uncertainty Avoidance Score: {unc_avoid_score:.2f}%"
        )