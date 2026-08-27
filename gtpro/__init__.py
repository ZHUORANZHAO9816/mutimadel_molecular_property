"""
GTpro: Graph-Text Alignment for Molecular Property Prediction.

This package contains the shared model architectures and utilities used by
both the pretraining and finetuning pipelines.
"""

__version__ = "0.1.0.dev0"

from .encoder import GTproEncoder

__all__ = ["GTproEncoder", "__version__"]
