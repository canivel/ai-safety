"""
Analysis Utilities for Model Diffing

This module provides tools for analyzing Cross-Coder outputs to find
Chat-specific latents related to user modeling and sycophancy.

Key analyses:
1. Finding User Model features (e.g., "User is a child" feature)
2. Differential analysis between prompt pairs (Novice vs Expert)
3. Correlation analysis between latents and response content
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class LatentInfo:
    """Information about a single latent/feature."""
    index: int
    type: str  # 'chat_specific', 'base_specific', 'shared'
    mean_activation: float
    max_activation: float
    chat_ratio: float  # How much this latent contributes to Chat vs Base
    top_tokens: Optional[List[Tuple[str, float]]] = None
    interpretation: Optional[str] = None


@dataclass
class PromptPairAnalysis:
    """Analysis results for a pair of related prompts."""
    prompt_a: str
    prompt_b: str
    label_a: str
    label_b: str
    differential_latents: List[Tuple[int, float]]  # (latent_idx, diff_score)
    top_in_a: List[LatentInfo]
    top_in_b: List[LatentInfo]


class LatentAnalyzer:
    """
    Analyzer for Cross-Coder latents.

    Provides methods for:
    - Finding Chat-specific features
    - Comparing latent activations across prompts
    - Identifying User Modeling features
    """

    def __init__(
        self,
        cross_coder,
        tokenizer=None,
    ):
        """
        Initialize the analyzer.

        Args:
            cross_coder: CrossCoderWrapper instance
            tokenizer: Tokenizer for interpreting token positions
        """
        self.cross_coder = cross_coder
        self.tokenizer = tokenizer

        # Cache for latent analyses
        self._cache = {}

    def analyze_latent(
        self,
        latent_idx: int,
        activations: Optional[torch.Tensor] = None,
    ) -> LatentInfo:
        """
        Get detailed information about a specific latent.

        Args:
            latent_idx: Index of the latent to analyze
            activations: Optional activations to compute statistics from

        Returns:
            LatentInfo with detailed analysis
        """
        info = self.cross_coder.get_latent_info(latent_idx)

        mean_act = 0.0
        max_act = 0.0

        if activations is not None:
            latents = self.cross_coder.encode(activations)
            latent_acts = latents[..., latent_idx]
            mean_act = latent_acts.mean().item()
            max_act = latent_acts.max().item()

        return LatentInfo(
            index=latent_idx,
            type=info["type"],
            mean_activation=mean_act,
            max_activation=max_act,
            chat_ratio=info["chat_ratio"],
        )

    def find_chat_specific_latents(
        self,
        activations: torch.Tensor,
        threshold: float = 0.5,
        top_k: int = 50,
    ) -> List[LatentInfo]:
        """
        Find latents that are specific to the Chat model.

        These are the RLHF-induced features, which may include:
        - User modeling features ("User is a child", "User is an expert")
        - Safety behaviors
        - Sycophancy tendencies

        Args:
            activations: Concatenated Base + Chat activations
            threshold: Minimum activation threshold
            top_k: Maximum number of latents to return

        Returns:
            List of LatentInfo for Chat-specific latents
        """
        result = self.cross_coder.get_chat_specific_latents(activations, threshold)

        chat_specific = result["chat_specific"]
        latent_acts = result["latent_activations"]
        chat_ratios = result["chat_ratio"]

        # Sort by activation strength
        scored = [(idx, latent_acts[idx], chat_ratios[idx]) for idx in chat_specific]
        scored.sort(key=lambda x: x[1], reverse=True)

        latent_infos = []
        for idx, act, ratio in scored[:top_k]:
            latent_infos.append(LatentInfo(
                index=idx,
                type="chat_specific",
                mean_activation=act,
                max_activation=act,  # We only have mean here
                chat_ratio=ratio,
            ))

        return latent_infos

    def differential_analysis(
        self,
        activations_a: torch.Tensor,
        activations_b: torch.Tensor,
        label_a: str = "A",
        label_b: str = "B",
        top_k: int = 20,
    ) -> Dict:
        """
        Compare latent activations between two sets of activations.

        This is the core analysis for finding User Modeling features:
        - Compare Novice vs Expert prompts
        - Look for latents that fire strongly for one but not the other

        Args:
            activations_a: Activations for first prompt set
            activations_b: Activations for second prompt set
            label_a: Label for first set
            label_b: Label for second set
            top_k: Number of top differential latents to return

        Returns:
            Dictionary with differential analysis results
        """
        latents_a = self.cross_coder.encode(activations_a)
        latents_b = self.cross_coder.encode(activations_b)

        # Mean across batch and sequence
        mean_a = latents_a.mean(dim=(0, 1))
        mean_b = latents_b.mean(dim=(0, 1))

        # Compute difference
        diff = mean_a - mean_b

        # Top latents more active in A
        top_a_vals, top_a_idx = diff.topk(top_k)

        # Top latents more active in B (negative diff)
        top_b_vals, top_b_idx = (-diff).topk(top_k)

        return {
            "label_a": label_a,
            "label_b": label_b,
            f"top_in_{label_a}": [
                {"latent_idx": idx.item(), "diff_score": val.item()}
                for idx, val in zip(top_a_idx, top_a_vals)
            ],
            f"top_in_{label_b}": [
                {"latent_idx": idx.item(), "diff_score": val.item()}
                for idx, val in zip(top_b_idx, top_b_vals)
            ],
            "mean_activations_a": mean_a.cpu().numpy(),
            "mean_activations_b": mean_b.cpu().numpy(),
            "difference": diff.cpu().numpy(),
        }

    def find_user_model_features(
        self,
        prompt_pairs: List[Dict],
        pipeline,
    ) -> Dict:
        """
        Find features that encode user attributes by analyzing prompt pairs.

        This is the key experiment for Research Idea 6:
        1. Run Novice and Expert versions of same questions
        2. Find latents that differentiate between them
        3. These are candidate "User Model" features

        Args:
            prompt_pairs: List of prompt pair dictionaries with 'novice' and 'expert' keys
            pipeline: ModelDiffingPipeline instance

        Returns:
            Dictionary with identified user modeling features
        """
        all_diffs = defaultdict(list)

        for pair in prompt_pairs:
            novice_prompt = pair.get("novice", {}).get("prompt", "")
            expert_prompt = pair.get("expert", {}).get("prompt", "")

            if not novice_prompt or not expert_prompt:
                continue

            # Run both prompts through pipeline
            novice_result = pipeline.analyze_prompt(novice_prompt)
            expert_result = pipeline.analyze_prompt(expert_prompt)

            if novice_result.latent_activations is None:
                continue

            # Differential analysis
            diff = self.differential_analysis(
                novice_result.concatenated_activations[pipeline.target_layer],
                expert_result.concatenated_activations[pipeline.target_layer],
                label_a="novice",
                label_b="expert",
            )

            # Track which latents consistently appear
            for entry in diff["top_in_novice"]:
                all_diffs[entry["latent_idx"]].append(entry["diff_score"])

        # Find latents that consistently fire more for novice prompts
        consistent_novice_latents = []
        for latent_idx, scores in all_diffs.items():
            if len(scores) >= 3:  # Appeared in at least 3 pairs
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if mean_score > 0.1:  # Consistently positive
                    consistent_novice_latents.append({
                        "latent_idx": latent_idx,
                        "mean_diff": mean_score,
                        "std_diff": std_score,
                        "n_pairs": len(scores),
                        "interpretation": "Candidate: 'User is Novice' feature",
                    })

        # Sort by mean difference
        consistent_novice_latents.sort(key=lambda x: x["mean_diff"], reverse=True)

        return {
            "user_model_candidates": consistent_novice_latents[:20],
            "all_latent_diffs": {k: {"scores": v, "mean": np.mean(v)} for k, v in all_diffs.items()},
            "interpretation": (
                "Latents with high positive mean_diff fire more strongly when the user "
                "appears to be a novice. These are candidates for 'User Modeling' features "
                "that encode the model's inference about user expertise."
            ),
        }


def find_chat_specific_latents(
    activations: torch.Tensor,
    cross_coder,
    threshold: float = 0.5,
) -> List[int]:
    """
    Convenience function to find Chat-specific latents.

    Args:
        activations: Concatenated Base + Chat activations
        cross_coder: CrossCoderWrapper instance
        threshold: Activation threshold

    Returns:
        List of Chat-specific latent indices
    """
    result = cross_coder.get_chat_specific_latents(activations, threshold)
    return result["chat_specific"]


def compute_latent_token_correlations(
    activations: torch.Tensor,
    cross_coder,
    tokens: torch.Tensor,
    tokenizer,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute which tokens most strongly activate each latent.

    This helps interpret what each latent represents.

    Args:
        activations: Input activations [batch, seq, 2*d_model]
        cross_coder: CrossCoderWrapper instance
        tokens: Token IDs [batch, seq]
        tokenizer: Tokenizer for decoding

    Returns:
        Dictionary mapping latent index to list of (token, activation) pairs
    """
    latents = cross_coder.encode(activations)  # [batch, seq, d_sae]

    # For each latent, find which tokens activate it most
    correlations = {}

    n_latents = latents.shape[-1]
    for latent_idx in range(min(n_latents, 1000)):  # Limit for efficiency
        latent_acts = latents[..., latent_idx]  # [batch, seq]

        # Find top activating positions
        flat_acts = latent_acts.flatten()
        flat_tokens = tokens.flatten()

        top_k = min(10, flat_acts.numel())
        top_vals, top_indices = flat_acts.topk(top_k)

        token_acts = []
        for val, idx in zip(top_vals, top_indices):
            token_id = flat_tokens[idx].item()
            token_str = tokenizer.decode([token_id])
            token_acts.append((token_str, val.item()))

        correlations[latent_idx] = token_acts

    return correlations


def interpret_latent(
    latent_idx: int,
    cross_coder,
    sample_activations: List[Tuple[str, torch.Tensor]],
) -> str:
    """
    Attempt to interpret what a latent represents.

    Args:
        latent_idx: Index of the latent to interpret
        cross_coder: CrossCoderWrapper instance
        sample_activations: List of (prompt, activations) pairs

    Returns:
        String interpretation of the latent
    """
    activating_prompts = []
    non_activating_prompts = []

    for prompt, acts in sample_activations:
        latents = cross_coder.encode(acts)
        latent_act = latents[..., latent_idx].mean().item()

        if latent_act > 0.5:
            activating_prompts.append((prompt, latent_act))
        else:
            non_activating_prompts.append((prompt, latent_act))

    # Sort by activation
    activating_prompts.sort(key=lambda x: x[1], reverse=True)

    # Create interpretation
    interpretation = f"Latent {latent_idx}:\n"
    interpretation += "ACTIVATES on:\n"
    for prompt, act in activating_prompts[:5]:
        interpretation += f"  [{act:.2f}] {prompt[:100]}...\n"

    interpretation += "DOES NOT ACTIVATE on:\n"
    for prompt, act in non_activating_prompts[:5]:
        interpretation += f"  [{act:.2f}] {prompt[:100]}...\n"

    return interpretation


class SycophancyDetector:
    """
    Detector for sycophancy-related latents.

    Uses the Cross-Coder analysis to identify features that:
    1. Are Chat-specific (introduced by RLHF)
    2. Correlate with sycophantic behavior
    """

    def __init__(self, cross_coder, pipeline):
        """
        Initialize the detector.

        Args:
            cross_coder: CrossCoderWrapper instance
            pipeline: ModelDiffingPipeline instance
        """
        self.cross_coder = cross_coder
        self.pipeline = pipeline
        self.analyzer = LatentAnalyzer(cross_coder)

        # Known sycophancy-related latent indices (populated during analysis)
        self.sycophancy_latents = []
        self.user_model_latents = []

    def identify_sycophancy_features(
        self,
        sycophancy_prompts: List[Dict],
    ) -> Dict:
        """
        Identify latents related to sycophancy by analyzing biased vs neutral prompts.

        Args:
            sycophancy_prompts: List of prompt pairs with 'biased' and 'neutral' versions

        Returns:
            Dictionary with sycophancy-related latents
        """
        all_diffs = defaultdict(list)

        for pair in sycophancy_prompts:
            biased_prompt = pair.get("biased", {}).get("prompt", "")
            neutral_prompt = pair.get("neutral", {}).get("prompt", "")

            if not biased_prompt or not neutral_prompt:
                continue

            biased_result = self.pipeline.analyze_prompt(biased_prompt)
            neutral_result = self.pipeline.analyze_prompt(neutral_prompt)

            if biased_result.latent_activations is None:
                continue

            # Find latents more active in biased prompts
            diff = self.analyzer.differential_analysis(
                biased_result.concatenated_activations[self.pipeline.target_layer],
                neutral_result.concatenated_activations[self.pipeline.target_layer],
                label_a="biased",
                label_b="neutral",
            )

            for entry in diff["top_in_biased"]:
                all_diffs[entry["latent_idx"]].append(entry["diff_score"])

        # Find consistent sycophancy-related latents
        sycophancy_candidates = []
        for latent_idx, scores in all_diffs.items():
            if len(scores) >= 2:
                mean_score = np.mean(scores)
                if mean_score > 0.1:
                    sycophancy_candidates.append({
                        "latent_idx": latent_idx,
                        "mean_diff": mean_score,
                        "n_prompts": len(scores),
                        "interpretation": "Candidate: Sycophancy-related feature",
                    })

        sycophancy_candidates.sort(key=lambda x: x["mean_diff"], reverse=True)
        self.sycophancy_latents = [c["latent_idx"] for c in sycophancy_candidates[:20]]

        return {
            "sycophancy_candidates": sycophancy_candidates[:20],
            "interpretation": (
                "These latents fire more strongly when prompts contain user beliefs "
                "that might trigger sycophantic validation. High-scoring latents may "
                "encode 'User has a belief to validate' or similar features."
            ),
        }

    def score_prompt_sycophancy_risk(self, prompt: str) -> Dict:
        """
        Score a prompt for sycophancy risk based on latent activations.

        Args:
            prompt: Input prompt to score

        Returns:
            Dictionary with sycophancy risk score and details
        """
        if not self.sycophancy_latents:
            return {"error": "Run identify_sycophancy_features first"}

        result = self.pipeline.analyze_prompt(prompt)

        if result.latent_activations is None:
            return {"error": "Could not get latent activations"}

        # Get mean activation of sycophancy-related latents
        mean_acts = result.latent_activations.mean(dim=(0, 1))
        syc_acts = [mean_acts[idx].item() for idx in self.sycophancy_latents]

        risk_score = np.mean(syc_acts) if syc_acts else 0.0

        return {
            "prompt": prompt,
            "sycophancy_risk_score": risk_score,
            "top_active_sycophancy_latents": [
                {"latent": idx, "activation": act}
                for idx, act in zip(self.sycophancy_latents, syc_acts)
                if act > 0.1
            ],
            "interpretation": (
                f"Risk score {risk_score:.2f}: "
                f"{'High' if risk_score > 0.5 else 'Medium' if risk_score > 0.2 else 'Low'} "
                "sycophancy risk based on latent activations."
            ),
        }


if __name__ == "__main__":
    print("Analysis module loaded successfully!")
    print("Key classes: LatentAnalyzer, SycophancyDetector")
    print("Key functions: find_chat_specific_latents, compute_latent_token_correlations")
