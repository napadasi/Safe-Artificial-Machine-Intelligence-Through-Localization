import torch
import torch.nn as nn
import torch.nn.functional as F  # Import functional
import torch.optim as optim  # Import optimizer

class PartiallyObfuscatingEncoder(nn.Module):
    """
    A unified encoder that combines image and text encoding, and calculates
    alignment scores and modified latent representations.
    """

    def __init__(self, image_encoder, text_encoder, latent_dim=128, device="cpu"):
        super(PartiallyObfuscatingEncoder, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.latent_dim = latent_dim
        self.device = device

        # Move encoders to the specified device
        self.image_encoder.to(self.device)
        self.text_encoder.to(self.device)

    def encode_image(self, x_image):
        """Encodes an image into latent space."""
        return self.image_encoder(x_image)

    def encode_text(self, text_tokens):
        """Encodes text tokens into latent space."""
        return self.text_encoder(text_tokens)

    def encode_text_batch_values(self, x_values_tokens, k_values, seq_len):
        """
        Encodes a batch of value texts and reshapes the result.

        Args:
            x_values_tokens (torch.Tensor): Value text tokens, shape (batch_size, k, seq_len).
            k_values (int): Number of values per image.
            seq_len (int): Sequence length for text tokens.

        Returns:
            torch.Tensor: Encoded values, shape (batch_size, k, latent_dim).
        """
        batch_size = x_values_tokens.size(0)
        x_values_tokens_flat = x_values_tokens.view(-1, seq_len)
        z_values_flat = self.text_encoder(x_values_tokens_flat)
        return z_values_flat.view(batch_size, k_values, self.latent_dim)

    def calculate_alignment_score(self, z_image, z_values, threshold=0.5, a=1.0):
        """
        Calculates the alignment score between image and value encodings.

        Args:
            z_image (torch.Tensor): Encoded image tensor, shape (batch_size, latent_dim).
            z_values (torch.Tensor): Encoded value tensors, shape (batch_size, k, latent_dim).
            threshold (float): The threshold value used in the formula.
            a (float): The scaling factor 'a' used in the formula.

        Returns:
            torch.Tensor: The alignment score for each item in the batch, shape (batch_size,).
        """
        # Ensure inputs are on the same device
        z_values = z_values.to(z_image.device)

        # Expand z_image to allow broadcasting for cosine similarity
        # z_image shape: (batch_size, 1, latent_dim)
        z_image_expanded = z_image.unsqueeze(1)

        # Calculate cosine similarity between the image and each value vector
        # F.cosine_similarity computes similarity along dim=-1 (latent_dim)
        # Result shape: (batch_size, k)
        cos_sim = F.cosine_similarity(z_image_expanded, z_values, dim=-1)

        # Normalize cosine similarity to [0, 1]
        # Result shape: (batch_size, k)
        normalized_sim = (cos_sim + 1) / 2

        # Average the normalized similarities over the k values
        # Result shape: (batch_size,)
        mean_sim = torch.mean(normalized_sim, dim=1)

        # Apply threshold, scaling factor 'a', and sigmoid
        score = torch.sigmoid(a * (mean_sim - threshold))

        return score

    def calculate_modified_latent(self, z_image, alignment_score):
        """
        Calculates the modified latent representation based on the alignment score.

        Args:
            z_image (torch.Tensor): Encoded image tensor, shape (batch_size, latent_dim).
            alignment_score (torch.Tensor): Alignment score, shape (batch_size,).

        Returns:
            torch.Tensor: The modified latent tensor, shape (batch_size, latent_dim).
        """
        batch_size, latent_dim = z_image.shape
        device = z_image.device

        # Sample epsilon from N(0, I)
        epsilon = torch.randn(batch_size, latent_dim, device=device)

        # Unsqueeze alignment_score for broadcasting: (batch_size,) -> (batch_size, 1)
        score_unsqueezed = alignment_score.unsqueeze(1)

        # Calculate sqrt factors, ensuring score is within [0, 1] for sqrt
        # Clamp score to avoid potential NaN from floating point inaccuracies near 0 or 1
        score_clamped = torch.clamp(score_unsqueezed, 0.0, 1.0)
        sqrt_score = torch.sqrt(score_clamped)
        sqrt_one_minus_score = torch.sqrt(1.0 - score_clamped)

        # Apply the formula
        z_tilde = sqrt_score * z_image + sqrt_one_minus_score * epsilon

        return z_tilde

    def process_image_and_values(
        self, x_image, x_values_tokens, k_values, seq_len, threshold=0.5, a=1.0
    ):
        """
        Complete pipeline for processing an image and values to get the modified latent.

        Args:
            x_image (torch.Tensor): Image tensor, shape (batch_size, 3, 64, 64).
            x_values_tokens (torch.Tensor): Value text tokens, shape (batch_size, k, seq_len).
            k_values (int): Number of values per image.
            seq_len (int): Sequence length for text tokens.
            threshold (float): Threshold for alignment score calculation.
            a (float): Scaling factor for alignment score calculation.

        Returns:
            tuple: (z_image, z_values, alignment_score, z_modified)
                z_image (torch.Tensor): Encoded image.
                z_values (torch.Tensor): Encoded text values.
                alignment_score (torch.Tensor): Calculated alignment score.
                z_modified (torch.Tensor): Modified latent representation.
        """
        # Encode image
        z_image = self.encode_image(x_image)

        # Encode values
        z_values = self.encode_text_batch_values(x_values_tokens, k_values, seq_len)

        # Calculate alignment score
        alignment_score = self.calculate_alignment_score(
            z_image, z_values, threshold, a
        )

        # Calculate modified latent
        z_modified = self.calculate_modified_latent(z_image, alignment_score)

        return z_image, z_values, alignment_score, z_modified

    def forward(
        self,
        x_image,
        x_values_tokens=None,
        k_values=None,
        seq_len=None,
        threshold=0.5,
        a=1.0,
    ):
        """
        Forward pass that handles different input scenarios.

        If only x_image is provided, returns the encoded image.
        If all parameters are provided, returns the full processing pipeline result.

        Args:
            x_image (torch.Tensor): Image tensor.
            x_values_tokens (torch.Tensor, optional): Value text tokens.
            k_values (int, optional): Number of values per image.
            seq_len (int, optional): Sequence length for text tokens.
            threshold (float, optional): Threshold for alignment score.
            a (float, optional): Scaling factor for alignment score.

        Returns:
            If only x_image: Encoded image
            If full inputs: (z_image, z_values, alignment_score, z_modified)
        """
        if x_values_tokens is None:
            return self.encode_image(x_image)
        else:
            return self.process_image_and_values(
                x_image, x_values_tokens, k_values, seq_len, threshold, a
            )