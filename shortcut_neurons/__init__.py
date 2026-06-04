"""
Shortcut Neuron Identification and Erasing Pipeline
=====================================================
Implements Ju et al. (2024) "Investigating Multi-Hop Factual Shortcuts in
Knowledge Editing of Large Language Models", adapted for HuggingFace Qwen models.

Key concepts:
  - A shortcut is the model's direct e1 → e3 association, bypassing the
    compositional e1 → e2 → e3 chain.
  - Shortcut neurons are FFN/MLP intermediate neurons (z_l dimensions) that
    carry this shortcut signal, identified via Integrated Gradients.
  - Erasing sets those neuron activations to zero at inference time without
    permanently modifying model weights.
"""

from .data import MultiHopExample, load_dataset_as_multihop, filter_shortcut_prone
from .modeling_qwen_hooks import QwenMLPHookManager, load_qwen_model
from .integrated_gradients import compute_ig_attribution, select_knowledge_neurons
from .locate import locate_shortcut_neurons, save_shortcut_neuron_results
from .erase import erase_neurons_inference, EraseMode
from .evaluate import evaluate_pipeline

__all__ = [
    "MultiHopExample",
    "load_dataset_as_multihop",
    "filter_shortcut_prone",
    "QwenMLPHookManager",
    "load_qwen_model",
    "compute_ig_attribution",
    "select_knowledge_neurons",
    "locate_shortcut_neurons",
    "save_shortcut_neuron_results",
    "erase_neurons_inference",
    "EraseMode",
    "evaluate_pipeline",
]
