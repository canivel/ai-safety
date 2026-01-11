"""
Model Diffing for Sycophancy Research

This package provides tools for comparing Base vs Chat (Instruct) models
using Cross-Coders to identify Chat-specific features related to user modeling
and sycophancy.

Main components:
- model_diffing: Core utilities for running both models and comparing activations
- cross_coder: Wrapper for loading and using Cross-Coder SAEs
- analysis: Tools for finding Chat-specific latents
- visualization: Dashboard and plotting utilities
"""

from .model_diffing import ModelDiffingPipeline, StereoModelRunner
from .cross_coder import CrossCoderWrapper
from .analysis import LatentAnalyzer, find_chat_specific_latents
from .visualization import create_latent_dashboard

__version__ = "0.1.0"
__all__ = [
    "ModelDiffingPipeline",
    "StereoModelRunner",
    "CrossCoderWrapper",
    "LatentAnalyzer",
    "find_chat_specific_latents",
    "create_latent_dashboard",
]
