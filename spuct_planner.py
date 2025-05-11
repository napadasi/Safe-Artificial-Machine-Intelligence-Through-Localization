import math  # For pi
import torch

class PlannerNode:
    """Node in the MCTS search tree."""

    def __init__(self, state, hidden, parent=None, inducing_goal=None):
        self.state = state  # Latent state z_t (Tensor)
        self.hidden = hidden  # RNN hidden state (tuple)
        self.parent = parent
        self.inducing_goal = (
            inducing_goal  # Goal g that led to this node (Tensor or None for root)
        )

        self.visit_count = 0
        # Stores stats for goals sampled *from this node*
        # key: tuple(goal_tensor.tolist()) for hashability
        # value: {'N': visit_count, 'W': total_value, 'Q': mean_value,
        #         'P': prior_density, 'unc': uncertainty, 'child_node': PlannerNode or None}
        self.goals_stats = {}
        self.is_expanded = (
            False  # Flag if a child node has been created for at least one goal
        )

    def get_goal_tensor(self, goal_key):
        """Converts hashable goal key back to tensor."""
        return torch.tensor(list(goal_key), device=self.state.device).unsqueeze(0)


class SPUCTPlanner:
    """Implements MCTS planning with SP-UCT for goal selection."""

    def __init__(
        self,
        meta_controller,
        mpc_sub_policy,
        world_model,
        num_simulations,
        c_spuct,
        beta,
        pw_k,
        pw_alpha,
        epsilon,
        device,
    ):
        self.meta_controller = meta_controller
        self.mpc_sub_policy = mpc_sub_policy
        self.world_model = world_model
        self.num_simulations = num_simulations
        self.c_spuct = c_spuct
        self.beta = beta
        self.pw_k = pw_k  # Progressive widening k
        self.pw_alpha = pw_alpha  # Progressive widening alpha
        self.epsilon = epsilon  # Epsilon for SP-UCT denominator stability
        self.device = device
        self.threshold = 0.2

        self.tree = (
            {}
        )  # Store nodes by state? Or just navigate via parent/child links? Start with root.
        self.root = None

    def _calculate_spuct_score(self, node, goal_key):
        """Calculates the SP-UCT score for a given goal at a node."""
        stats = node.goals_stats[goal_key]
        N_s = node.visit_count
        N_sg = stats["N"]
        Q_sg = stats["Q"]
        P_sg = stats["P"]
        unc_g = stats["unc"]  # Uncertainty associated with achieving this goal

        exploration_term = (
            self.c_spuct
            * P_sg
            * math.exp(-self.beta * unc_g)
            * (math.sqrt(N_s + self.epsilon) / (1 + N_sg))
        )

        return Q_sg + exploration_term

    def _select_child(self, node):
        """Selects the best goal from a node using SP-UCT and Progressive Widening."""

        # --- Progressive Widening ---
        num_goals_to_consider = math.floor(
            self.pw_k * (node.visit_count**self.pw_alpha)
        )

        while len(node.goals_stats) < num_goals_to_consider:
            # Sample a new goal
            with torch.no_grad():
                # Ensure state has batch dim
                state_batch = (
                    node.state if node.state.dim() > 1 else node.state.unsqueeze(0)
                )
                sampled_goal, goal_mu, goal_log_var = self.meta_controller.sample_goal(
                    state_batch
                )
                # Calculate prior density P(g|s)
                prior_density = self.meta_controller.calculate_pdf(
                    goal_mu, goal_log_var, sampled_goal
                )

            # Use MPC to get an initial estimate of uncertainty (unc(g)) for this new goal
            _, _, initial_unc = self.mpc_sub_policy.select_action(
                state_batch, sampled_goal, node.hidden
            )

            goal_key = tuple(sampled_goal.squeeze(0).tolist())  # Make hashable
            if goal_key not in node.goals_stats and initial_unc < self.threshold:
                node.goals_stats[goal_key] = {
                    "N": 0,
                    "W": 0.0,
                    "Q": 0.0,
                    "P": prior_density.item(),  # Store prior
                    "unc": initial_unc,  # Store estimated uncertainty
                    "child_node": None,
                }
            # Break if we have enough goals, avoid infinite loop if sampling fails
            if len(node.goals_stats) >= num_goals_to_consider:
                break
            # Add safety break if sampling keeps producing duplicates (unlikely with continuous goals)
            if len(node.goals_stats) > node.visit_count + 1:  # Heuristic break
                print(
                    "Warning: Progressive widening sampling many duplicates or stuck."
                )
                break

        # --- Select Best Goal using SP-UCT ---
        best_goal_key = None
        best_score = -float("inf")

        if not node.goals_stats:  # Should not happen if PW adds at least one goal
            print("Error: No goals available for selection in node.")
            return None, None

        for goal_key, stats in node.goals_stats.items():
            score = self._calculate_spuct_score(node, goal_key)
            if score > best_score:
                best_score = score
                best_goal_key = goal_key

        return best_goal_key, node.goals_stats[best_goal_key]

    def _expand_node(self, node, goal_key):
        """Expand the tree by simulating the step for the chosen goal."""
        goal_tensor = node.get_goal_tensor(goal_key)
        stats = node.goals_stats[goal_key]

        # Simulate step using MPC action selection + world model prediction
        current_z_batch = (
            node.state if node.state.dim() > 1 else node.state.unsqueeze(0)
        )
        action, _, avg_variance = self.mpc_sub_policy.select_action(
            current_z_batch, goal_tensor, node.hidden
        )

        # Use world model to get next state distribution based on chosen action
        mu_next, log_var_next, next_hidden = self.world_model(
            current_z_batch, action, node.hidden
        )

        # Sample next state
        # std_dev_next = torch.exp(0.5 * log_var_next)
        # epsilon_next = torch.randn_like(std_dev_next)
        # next_z_sampled = mu_next + std_dev_next * epsilon_next
        next_z_sampled = mu_next
        # Update uncertainty estimate if needed (using avg_variance from MPC)
        stats["unc"] = avg_variance  # Update uncertainty with value from this path

        # Create the new node
        child_node = PlannerNode(
            state=next_z_sampled.squeeze(0),
            hidden=next_hidden,
            parent=node,
            inducing_goal=goal_tensor,
        )
        stats["child_node"] = child_node
        node.is_expanded = True  # Mark parent as expanded

        # --- Value Estimation ---
        value = 0.0
        return child_node, value

    def _backpropagate(self, path, value):
        """Update node statistics along the path back to the root."""
        for node, goal_key in reversed(path):
            if goal_key is None:  # Should only happen for root if path is empty
                node.visit_count += 1
                continue

            stats = node.goals_stats[goal_key]
            node.visit_count += 1
            stats["N"] += 1
            stats["W"] += value
            stats["Q"] = stats["W"] / stats["N"]

    def run_mcts(self, root_state, root_hidden):
        """Runs the MCTS search for a given number of simulations."""
        self.root = PlannerNode(state=root_state.squeeze(0), hidden=root_hidden)

        if not self.root.state.is_cuda and self.device == "cuda":
            self.root.state = self.root.state.to(self.device)
            if self.root.hidden is not None:
                self.root.hidden = tuple(h.to(self.device) for h in self.root.hidden)

        for _ in range(self.num_simulations):
            node = self.root
            path = []  # Stores (node, goal_key) pairs

            # --- Selection ---
            while node.is_expanded:
                goal_key, stats = self._select_child(node)
                if goal_key is None:
                    print("MCTS Selection Error: Could not select goal.")
                    break

                path.append((node, goal_key))
                if stats["child_node"] is not None:
                    node = stats["child_node"]
                else:
                    break
            else:
                if node == self.root and not node.is_expanded:
                    goal_key, stats = self._select_child(node)
                    if goal_key is None:
                        print("MCTS Selection Error at Root.")
                        continue
                    path.append((node, goal_key))

            # --- Expansion ---
            if path:
                last_node, last_goal_key = path[-1]
                if last_node.goals_stats[last_goal_key]["child_node"] is None:
                    node, value = self._expand_node(last_node, last_goal_key)
                else:
                    node = last_node.goals_stats[last_goal_key]["child_node"]
                    value = 0.0
            else:
                print("MCTS Error: Expansion phase reached unexpectedly.")
                continue

            # --- Backpropagation ---
            self._backpropagate(path, value)

    def get_best_goal(self):
        """Selects the best goal from the root node after MCTS."""
        if self.root is None or not self.root.goals_stats:
            print("Error: MCTS not run or root has no goals.")
            return None

        best_goal_key = None
        max_visits = -1
        for goal_key, stats in self.root.goals_stats.items():
            if stats["N"] > max_visits:
                max_visits = stats["N"]
                best_goal_key = goal_key

        if best_goal_key is None:
            if self.root.goals_stats:
                best_goal_key = next(iter(self.root.goals_stats))
            else:
                return None

        return self.root.get_goal_tensor(best_goal_key)