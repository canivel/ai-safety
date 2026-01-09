"""
Cross-Coder Wrapper for Model Diffing

This module provides utilities for loading and using Cross-Coders,
which are special Sparse Autoencoders trained on concatenated activations
from both Base and Chat models.

Cross-Coders enable "Model Diffing" by separating features into:
- Shared: Present in both models (math, grammar, factual knowledge)
- Base-Specific: Present only in Base model
- Chat-Specific: Present only in Chat model (user modeling, safety, sycophancy)

References:
- Anthropic's Cross-Coder work
- Butanium/gemma-2-2b-crosscoder (Open Source replication)
- dictionary_learning library (saprmarks/dictionary_learning)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
import json

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


@dataclass
class CrossCoderConfig:
    """Configuration for a Cross-Coder model."""
    d_model: int = 2304  # Gemma 2 2B hidden size
    d_sae: int = 16384   # Typical SAE latent dimension
    n_models: int = 2    # Number of models being compared (Base + Chat)
    sparsity_coefficient: float = 1e-3
    layer: int = 13


class CrossCoderWrapper:
    """
    Wrapper for loading and using Cross-Coder SAEs.

    Supports multiple backends:
    1. dictionary_learning library (recommended)
    2. Custom PyTorch implementation
    3. Pre-trained weights from HuggingFace
    """

    def __init__(
        self,
        config: CrossCoderConfig,
        encoder: nn.Module,
        decoder: nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize CrossCoder wrapper.

        Args:
            config: CrossCoder configuration
            encoder: Encoder network (concat activations -> latents)
            decoder: Decoder network (latents -> reconstructed activations)
            device: Device to run on
        """
        self.config = config
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

        # Metadata about latent types (populated when loading pre-trained)
        self.latent_types: Optional[Dict[int, str]] = None
        self.chat_specific_indices: Optional[List[int]] = None
        self.base_specific_indices: Optional[List[int]] = None
        self.shared_indices: Optional[List[int]] = None

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        device: str = "cuda",
        from_hub: bool = True,
    ) -> "CrossCoderWrapper":
        """
        Load a pre-trained Cross-Coder.

        Args:
            path_or_repo: Local path or HuggingFace repo ID
            device: Device to load to
            from_hub: Whether to download from HuggingFace Hub

        Returns:
            Loaded CrossCoderWrapper instance
        """
        if from_hub and HF_HUB_AVAILABLE:
            return cls._load_from_hub(path_or_repo, device)
        else:
            return cls._load_from_local(path_or_repo, device)

    @classmethod
    def _load_from_hub(cls, repo_id: str, device: str) -> "CrossCoderWrapper":
        """Load Cross-Coder from HuggingFace Hub."""
        print(f"Loading Cross-Coder from HuggingFace: {repo_id}")

        # Try to load using dictionary_learning format first
        try:
            return cls._load_dictionary_learning_format(repo_id, device)
        except Exception as e:
            print(f"dictionary_learning format failed: {e}")

        # Fall back to generic format
        return cls._load_generic_format(repo_id, device)

    @classmethod
    def _load_dictionary_learning_format(cls, repo_id: str, device: str) -> "CrossCoderWrapper":
        """
        Load using dictionary_learning library format.

        The dictionary_learning library (saprmarks/dictionary_learning) provides
        a CrossCoder class specifically designed for model diffing.
        """
        try:
            # Try importing the dictionary_learning CrossCoder
            from dictionary_learning import CrossCoder

            # Load the pre-trained CrossCoder
            cross_coder = CrossCoder.from_pretrained(repo_id, from_hub=True)
            cross_coder = cross_coder.to(device)

            # Wrap it in our interface
            config = CrossCoderConfig(
                d_model=cross_coder.d_model if hasattr(cross_coder, 'd_model') else 2304,
                d_sae=cross_coder.d_sae if hasattr(cross_coder, 'd_sae') else 16384,
            )

            # Create wrapper that delegates to the dictionary_learning CrossCoder
            wrapper = cls.__new__(cls)
            wrapper.config = config
            wrapper.device = device
            wrapper._dict_learning_model = cross_coder
            wrapper.latent_types = None
            wrapper.chat_specific_indices = None
            wrapper.base_specific_indices = None
            wrapper.shared_indices = None

            # Override encode/decode to use the underlying model
            wrapper._use_dict_learning = True

            return wrapper

        except ImportError:
            raise ImportError(
                "dictionary_learning library not found. Install with:\n"
                "pip install dictionary-learning\n"
                "or: git clone https://github.com/saprmarks/dictionary_learning"
            )

    @classmethod
    def _load_generic_format(cls, repo_id: str, device: str) -> "CrossCoderWrapper":
        """Load from a generic safetensors/pytorch format."""
        # Download model files
        model_dir = snapshot_download(repo_id)

        # Look for config
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = CrossCoderConfig(**config_dict)
        else:
            config = CrossCoderConfig()  # Use defaults

        # Look for weights
        weights_path = None
        for name in ["model.safetensors", "pytorch_model.bin", "crosscoder.pt"]:
            path = Path(model_dir) / name
            if path.exists():
                weights_path = path
                break

        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {model_dir}")

        # Load weights
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=device)

        # Create encoder/decoder
        encoder = cls._create_encoder(config, state_dict)
        decoder = cls._create_decoder(config, state_dict)

        return cls(config, encoder, decoder, device)

    @classmethod
    def _load_from_local(cls, path: str, device: str) -> "CrossCoderWrapper":
        """Load Cross-Coder from local path."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = CrossCoderConfig(**json.load(f))
        else:
            config = CrossCoderConfig()

        # Load weights (similar to hub loading)
        weights_path = None
        for name in ["model.safetensors", "pytorch_model.bin", "crosscoder.pt"]:
            if (path / name).exists():
                weights_path = path / name
                break

        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {path}")

        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=device)

        encoder = cls._create_encoder(config, state_dict)
        decoder = cls._create_decoder(config, state_dict)

        return cls(config, encoder, decoder, device)

    @staticmethod
    def _create_encoder(config: CrossCoderConfig, state_dict: Dict) -> nn.Module:
        """Create encoder from state dict."""
        # Input: concatenated activations from both models
        # Shape: [batch, seq, 2 * d_model]
        input_dim = config.n_models * config.d_model

        encoder = nn.Sequential(
            nn.Linear(input_dim, config.d_sae),
            nn.ReLU(),  # SAEs typically use ReLU for sparsity
        )

        # Load weights if available
        if "encoder.0.weight" in state_dict:
            encoder[0].weight.data = state_dict["encoder.0.weight"]
            encoder[0].bias.data = state_dict["encoder.0.bias"]
        elif "W_enc" in state_dict:
            encoder[0].weight.data = state_dict["W_enc"].T
            if "b_enc" in state_dict:
                encoder[0].bias.data = state_dict["b_enc"]

        return encoder

    @staticmethod
    def _create_decoder(config: CrossCoderConfig, state_dict: Dict) -> nn.Module:
        """Create decoder from state dict."""
        output_dim = config.n_models * config.d_model

        decoder = nn.Linear(config.d_sae, output_dim)

        # Load weights if available
        if "decoder.weight" in state_dict:
            decoder.weight.data = state_dict["decoder.weight"]
            decoder.bias.data = state_dict["decoder.bias"]
        elif "W_dec" in state_dict:
            decoder.weight.data = state_dict["W_dec"]
            if "b_dec" in state_dict:
                decoder.bias.data = state_dict["b_dec"]

        return decoder

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode concatenated activations to latent space.

        Args:
            activations: Concatenated Base + Chat activations
                        Shape: [batch, seq_len, 2 * d_model]

        Returns:
            Latent activations, Shape: [batch, seq_len, d_sae]
        """
        if hasattr(self, '_use_dict_learning') and self._use_dict_learning:
            return self._dict_learning_model.encode(activations)

        with torch.no_grad():
            return self.encoder(activations.to(self.device))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to activation space.

        Args:
            latents: Latent activations, Shape: [batch, seq_len, d_sae]

        Returns:
            Reconstructed activations, Shape: [batch, seq_len, 2 * d_model]
        """
        if hasattr(self, '_use_dict_learning') and self._use_dict_learning:
            return self._dict_learning_model.decode(latents)

        with torch.no_grad():
            return self.decoder(latents.to(self.device))

    def forward(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            activations: Input activations

        Returns:
            Tuple of (latents, reconstructed)
        """
        latents = self.encode(activations)
        reconstructed = self.decode(latents)
        return latents, reconstructed

    def get_active_latents(
        self,
        activations: torch.Tensor,
        threshold: float = 0.1,
        return_values: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Get indices of active latents (above threshold).

        Args:
            activations: Input activations
            threshold: Activation threshold
            return_values: Whether to also return activation values

        Returns:
            List of active latent indices, optionally with values
        """
        latents = self.encode(activations)

        # Average across batch and sequence
        mean_latents = latents.mean(dim=(0, 1))

        # Find active
        active_mask = mean_latents.abs() > threshold
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

        if return_values:
            active_values = mean_latents[active_mask]
            return active_indices, active_values

        return active_indices

    def get_chat_specific_latents(
        self,
        activations: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, List[int]]:
        """
        Identify Chat-specific latents from concatenated activations.

        This is the key function for Model Diffing. It identifies features
        that are present in the Chat model but not in the Base model.

        The method depends on the Cross-Coder architecture:
        1. If the Cross-Coder has explicit partitioning, use that
        2. Otherwise, infer from activation patterns

        Args:
            activations: Concatenated Base + Chat activations
                        Shape: [batch, seq, 2 * d_model]
            threshold: Threshold for considering a latent "active"

        Returns:
            Dictionary with:
            - chat_specific: Latents active only in Chat reconstructions
            - base_specific: Latents active only in Base reconstructions
            - shared: Latents active in both
        """
        latents = self.encode(activations)
        reconstructed = self.decode(latents)

        # Split reconstructed back into Base and Chat portions
        d_model = self.config.d_model
        recon_base = reconstructed[..., :d_model]
        recon_chat = reconstructed[..., d_model:]

        # Compute contribution of each latent to each model's reconstruction
        # This requires looking at the decoder weights
        if hasattr(self, '_use_dict_learning') and self._use_dict_learning:
            # dictionary_learning models may have different structure
            decoder_weights = self._dict_learning_model.W_dec  # [d_sae, 2*d_model]
        else:
            decoder_weights = self.decoder.weight.T  # [d_sae, 2*d_model]

        # Split decoder weights
        dec_base = decoder_weights[:, :d_model]  # [d_sae, d_model]
        dec_chat = decoder_weights[:, d_model:]  # [d_sae, d_model]

        # Compute "contribution scores" for each latent to each model
        # A latent is Chat-specific if its decoder weights point mostly to Chat dimensions
        base_contribution = dec_base.norm(dim=1)  # [d_sae]
        chat_contribution = dec_chat.norm(dim=1)  # [d_sae]

        # Normalize
        total = base_contribution + chat_contribution + 1e-8
        base_ratio = base_contribution / total
        chat_ratio = chat_contribution / total

        # Classify latents
        chat_specific = (chat_ratio > 0.7).nonzero(as_tuple=True)[0].tolist()
        base_specific = (base_ratio > 0.7).nonzero(as_tuple=True)[0].tolist()
        shared = ((chat_ratio > 0.3) & (base_ratio > 0.3)).nonzero(as_tuple=True)[0].tolist()

        # Filter to only active latents
        mean_latents = latents.mean(dim=(0, 1)).abs()
        active_mask = mean_latents > threshold

        chat_specific = [i for i in chat_specific if active_mask[i]]
        base_specific = [i for i in base_specific if active_mask[i]]
        shared = [i for i in shared if active_mask[i]]

        return {
            "chat_specific": chat_specific,
            "base_specific": base_specific,
            "shared": shared,
            "latent_activations": mean_latents.cpu().numpy(),
            "chat_ratio": chat_ratio.cpu().numpy(),
            "base_ratio": base_ratio.cpu().numpy(),
        }

    def get_latent_info(self, latent_idx: int) -> Dict:
        """
        Get information about a specific latent.

        Args:
            latent_idx: Index of the latent to inspect

        Returns:
            Dictionary with latent metadata
        """
        if hasattr(self, '_use_dict_learning') and self._use_dict_learning:
            decoder_weights = self._dict_learning_model.W_dec
        else:
            decoder_weights = self.decoder.weight.T

        d_model = self.config.d_model
        latent_weights = decoder_weights[latent_idx]

        base_weights = latent_weights[:d_model]
        chat_weights = latent_weights[d_model:]

        base_norm = base_weights.norm().item()
        chat_norm = chat_weights.norm().item()

        return {
            "index": latent_idx,
            "base_contribution": base_norm,
            "chat_contribution": chat_norm,
            "chat_ratio": chat_norm / (base_norm + chat_norm + 1e-8),
            "type": self._classify_latent(base_norm, chat_norm),
            "base_weights": base_weights.cpu().numpy(),
            "chat_weights": chat_weights.cpu().numpy(),
        }

    def _classify_latent(self, base_norm: float, chat_norm: float) -> str:
        """Classify a latent as chat-specific, base-specific, or shared."""
        total = base_norm + chat_norm + 1e-8
        chat_ratio = chat_norm / total

        if chat_ratio > 0.7:
            return "chat_specific"
        elif chat_ratio < 0.3:
            return "base_specific"
        else:
            return "shared"


def create_mock_cross_coder(
    d_model: int = 2304,
    d_sae: int = 1024,
    device: str = "cuda",
) -> CrossCoderWrapper:
    """
    Create a mock Cross-Coder for testing without loading pretrained weights.

    Useful for testing the pipeline before downloading large model files.

    Args:
        d_model: Model hidden dimension
        d_sae: SAE latent dimension
        device: Device to create on

    Returns:
        Mock CrossCoderWrapper
    """
    config = CrossCoderConfig(d_model=d_model, d_sae=d_sae)

    input_dim = 2 * d_model
    encoder = nn.Sequential(
        nn.Linear(input_dim, d_sae),
        nn.ReLU(),
    )
    decoder = nn.Linear(d_sae, input_dim)

    # Initialize with small random weights
    nn.init.xavier_uniform_(encoder[0].weight)
    nn.init.xavier_uniform_(decoder.weight)

    return CrossCoderWrapper(config, encoder, decoder, device)


if __name__ == "__main__":
    # Quick test
    print("Creating mock Cross-Coder...")
    cc = create_mock_cross_coder(device="cpu")

    # Test with random input
    x = torch.randn(1, 10, 2 * 2304)  # [batch, seq, 2*d_model]
    latents, recon = cc.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Latents shape: {latents.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print("Mock Cross-Coder working!")
