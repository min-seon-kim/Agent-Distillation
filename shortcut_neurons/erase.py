"""
Inference-time shortcut neuron erasing.

Implements two erasing modes:

local  – erase only the shortcut neurons identified for the current example.
global – erase the union of shortcut neurons across all examples.

Erasing is performed by registering ablations on the QwenMLPHookManager:
the hook sets z_l[..., token_pos, neuron_j] = 0 before the down_proj call.

Model weights are never modified by default.  An optional
permanent_weight_edit is available but must be explicitly requested.
"""

from __future__ import annotations

import enum
import logging
from typing import Any, Dict, List, Optional

import torch

from .modeling_qwen_hooks import QwenMLPHookManager, NeuronList

logger = logging.getLogger(__name__)


class EraseMode(str, enum.Enum):
    """
    LOCAL  – use only the neurons for the current example.
    GLOBAL – use the union of neurons across all examples.
    """
    LOCAL  = "local"
    GLOBAL = "global"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def erase_neurons_inference(
    manager: QwenMLPHookManager,
    tokenizer,
    prompt: str,
    shortcut_neurons: NeuronList,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> str:
    """
    Run inference with the given shortcut neurons zeroed out.

    Registers the ablation, generates a greedy response, then immediately
    removes the ablation so subsequent calls are unaffected.

    Parameters
    ----------
    manager:
        Wrapped QwenMLPHookManager.
    tokenizer:
        Matching tokenizer.
    prompt:
        The multi-hop question string.
    shortcut_neurons:
        List of (layer_idx, neuron_idx) to erase.
    token_pos:
        Position at which to erase.  -1 = last prompt token (default).
        None = all token positions.
    max_new_tokens:
        Maximum tokens to generate.
    device:
        Device string.  Inferred from model if None.
    """
    if not manager._is_wrapped:
        raise RuntimeError("Call manager.wrap_all_mlps() before erasing.")

    if device is None:
        device = str(next(manager.model.parameters()).device)

    manager.set_ablate_neurons(shortcut_neurons, token_pos=token_pos)
    try:
        result = manager.generate_text(
            tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    finally:
        manager.clear_ablations()

    return result


def build_global_neuron_set(results: List[Dict[str, Any]]) -> NeuronList:
    """
    Compute the union of shortcut neurons across all locate-stage results.

    Returns deduplicated sorted list of (layer_idx, neuron_idx) tuples.
    """
    union: set = set()
    for r in results:
        for l_j in r.get("shortcut_neurons", []):
            union.add((int(l_j[0]), int(l_j[1])))
    global_neurons = sorted(union, key=lambda t: (t[0], t[1]))
    logger.info(
        "Global neuron set: %d unique neurons from %d examples",
        len(global_neurons), len(results),
    )
    return global_neurons


def erase_batch(
    manager: QwenMLPHookManager,
    tokenizer,
    examples,
    locate_results: List[Dict[str, Any]],
    erase_mode: EraseMode = EraseMode.LOCAL,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run inference with erasing for a batch of examples.

    Returns list of dicts {id, prompt, prediction_before, prediction_after, answer}.
    """
    if len(examples) != len(locate_results):
        raise ValueError(
            f"examples ({len(examples)}) and locate_results ({len(locate_results)}) "
            "must have the same length."
        )

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    records = []
    for ex, loc in zip(examples, locate_results):
        neurons = (
            [(int(l), int(j)) for l, j in loc.get("shortcut_neurons", [])]
            if erase_mode == EraseMode.LOCAL
            else global_neurons
        )
        prediction_after = erase_neurons_inference(
            manager=manager, tokenizer=tokenizer,
            prompt=ex.multi_hop_prompt,
            shortcut_neurons=neurons,
            token_pos=token_pos,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        records.append({
            "id": ex.id,
            "prompt": ex.multi_hop_prompt,
            "answer": ex.answer,
            "prediction_before": loc.get("prediction_before", ""),
            "prediction_after": prediction_after,
            "num_erased_neurons": len(neurons),
            "erase_mode": str(erase_mode),
        })

    return records


# ---------------------------------------------------------------------------
# Optional: permanent weight edit (off by default)
# ---------------------------------------------------------------------------

def apply_permanent_weight_edit(
    manager: QwenMLPHookManager,
    neurons: NeuronList,
) -> None:
    """
    Permanently zero out down_proj columns corresponding to shortcut neurons.

    WARNING: This modifies model weights in-place and is irreversible without
    reloading the checkpoint.  Only call when explicitly requested.

    Setting down_proj.weight[:, j] = 0 means neuron j contributes nothing to
    the residual stream regardless of its activation value.
    """
    with torch.no_grad():
        layers = manager.model.model.layers
        for layer_idx, neuron_idx in neurons:
            down_proj = layers[layer_idx].mlp.down_proj
            down_proj.weight[:, neuron_idx] = 0.0

    logger.info(
        "Permanent weight edit: zeroed %d neurons in down_proj.", len(neurons)
    )
