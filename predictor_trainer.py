import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional
import torch.optim as optim  # Import optimizer
import math  # For pi
import numpy as np  # For PDF calculation
import gymnasium as gym
import safety_gymnasium  # Import Safety Gymnasium
from PIL import Image
import torchvision.transforms as T

class PredictorTrainer:
    def __init__(
        self,
        po_encoder,
        world_model,
        optimizer,
        loss_fn,
        device="cpu",
        env_id="PointGoal1-v0",
    ):
        self.po_encoder = po_encoder  # Replace separate encoders with unified encoder
        self.world_model = world_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # Move models to the specified device
        self.po_encoder.to(self.device)
        self.world_model.to(self.device)

        # Initialize Safety Gymnasium environment
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.observation, self.info = self.env.reset()

        # Transform to prepare observations for the model
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

        # Store previous observation for sequence generation
        self.prev_observation = None

    def _process_observation(self, obs_array):
        """Convert observation array to tensor of shape (3, 64, 64)"""
        # Convert numpy array to PIL Image
        image = Image.fromarray(obs_array)
        # Apply transformations and move to appropriate device
        return self.transform(image).to(self.device)

    def _generate_dummy_batch(
        self, batch_size, vocab_size, seq_len, k_values, action_dim, latent_dim
    ):
        """Generates a single batch of data from Safety Gymnasium environment."""
        # Initialize tensors for batch
        x_image_t_batch = []
        x_image_t_plus_1_batch = []
        a_t_batch = []

        for _ in range(batch_size):
            # If this is the first step or after reset
            if self.prev_observation is None:
                obs_array = self.env.render()
                x_image_t = self._process_observation(obs_array)
                self.prev_observation = self.observation.copy()
            else:
                # Use previous observation
                obs_array = self.env.render()
                x_image_t = self._process_observation(obs_array)

            # Sample a random action
            action = self.env.action_space.sample()

            # Take a step in the environment
            self.observation, reward, terminated, truncated, info = self.env.step(
                action
            )

            # Convert action to tensor
            a_t = torch.tensor(action, dtype=torch.float32, device=self.device)

            # Get the next observation image
            obs_array_next = self.env.render()
            x_image_t_plus_1 = self._process_observation(obs_array_next)

            # Reset if episode is done
            if terminated or truncated:
                self.observation, self.info = self.env.reset()
                self.prev_observation = None
            else:
                self.prev_observation = self.observation.copy()

            # Append to batch
            x_image_t_batch.append(x_image_t)
            x_image_t_plus_1_batch.append(x_image_t_plus_1)
            a_t_batch.append(a_t)

        # Convert lists to tensors
        x_image_t = torch.stack(x_image_t_batch)
        x_image_t_plus_1 = torch.stack(x_image_t_plus_1_batch)
        a_t = torch.stack(a_t_batch)

        # For text tokens, we still generate dummy data as Safety Gymnasium doesn't provide text
        x_values_tokens = torch.randint(
            1, vocab_size, (batch_size, k_values, seq_len), device=self.device
        )

        # Ensure action tensor has correct shape
        if a_t.shape[1] < action_dim:
            # Pad with zeros if needed
            padding = torch.zeros(
                batch_size, action_dim - a_t.shape[1], device=self.device
            )
            a_t = torch.cat([a_t, padding], dim=1)

        return x_image_t, x_image_t_plus_1, x_values_tokens, a_t

    def train_step(
        self,
        x_image_t,  # Image before action
        x_image_t_plus_1,  # Image after action (target)
        x_values_tokens,
        a_t,
        k_values,
        seq_len,
        latent_dim,
        threshold,
        scale_factor_a,
    ):
        """Performs a single training step."""
        self.world_model.train()  # Set world model to training mode
        # Set encoder to eval if not training it simultaneously
        self.po_encoder.eval()

        # --- Get Encodings and Target ---
        with torch.no_grad():  # No gradients needed for encoder part if not training them
            # Encode current state (image before action) -> z_t
            z_t = self.po_encoder.encode_image(x_image_t)

            # Process target state and values to get z_target (modified latent)
            # Use the PartiallyObfuscatingEncoder's process_image_and_values method
            _, _, _, z_target = self.po_encoder.process_image_and_values(
                x_image_t_plus_1,  # Target image
                x_values_tokens,  # Value tokens
                k_values,  # Number of values per image
                seq_len,  # Sequence length
                threshold,  # Alignment threshold
                scale_factor_a,  # Scaling factor for alignment
            )

        # --- World Model Prediction ---
        # Predict mu and log_var for z_{t+1} based on z_t (state before action) and a_t
        hidden_state = None  # Assuming single step prediction for now
        mu_pred, log_var_pred, _ = self.world_model(z_t, a_t, hidden_state)

        # --- Calculate Loss ---
        # Loss compares world model prediction with z_target (derived from z_{t+1}_actual)
        loss = self.loss_fn(mu_pred, log_var_pred, z_target)

        # --- Backpropagation ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        num_epochs,
        batch_size,
        vocab_size,
        seq_len,
        k_values,
        action_dim,
        latent_dim,
        threshold,
        scale_factor_a,
    ):
        """Runs the training loop."""
        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            # --- Generate Dummy Data for one step ---
            # In a real scenario, you would use a DataLoader here
            x_image_t, x_image_t_plus_1, x_values_tokens, a_t = (
                self._generate_dummy_batch(
                    batch_size, vocab_size, seq_len, k_values, action_dim, latent_dim
                )
            )

            # --- Perform Training Step ---
            loss = self.train_step(
                x_image_t,  # Pass current image
                x_image_t_plus_1,  # Pass target image
                x_values_tokens,
                a_t,
                k_values,
                seq_len,
                latent_dim,
                threshold,
                scale_factor_a,
            )

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

        print("Training Finished.")