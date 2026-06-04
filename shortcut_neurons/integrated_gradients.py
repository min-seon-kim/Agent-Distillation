"""
Integrated Gradients (IG) attribution for Knowledge Neuron identification.

Implements the Riemann-sum approximation from Ju et al. (2024), which itself
follows Dai et al. (2022) "Knowledge Neurons in Pretrained Transformers".

Attribution formula
-------------------
For layer l, neuron j, prompt x, and answer token a:

    Attr_{l,j}(x, a) = z_{l,j}^0
                       × (1/m) × Σ_{k=1}^{m}
                         ∂ p(a | x ; z_l = (k/m)·z_l^0) / ∂ z_{l,j}

where:
  z_l^0      – original activation vector at the answer-prediction position
  m          – Riemann steps (default 20, paper value)
  baseline   – zero vector

Implementation approach
-----------------------
We compute IG for all layers in m total forward passes by scaling all layers'
intermediate activations to (k/m)·z_l^0 simultaneously (path-IG).
This reduces computation from O(m·L) to O(m) passes.

Set layer_by_layer=True for the exact layer-by-layer IG (O(m·L) passes),
which matches the paper's neuron-by-neuron formula exactly.

Target mode
-----------
first_token_prob (default, paper-compatible):
    Log-probability of the first answer token at the last prompt position.
sequence_logprob:
    Sum of log-probs over all answer tokens (robust for multi-token answers).

Gradient scope
--------------
Gradients are computed only with respect to z_l.
Model parameters are kept frozen.
Attribution accumulation is cast to fp32 regardless of inference dtype.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .modeling_qwen_hooks import QwenMLPHookManager, NeuronList

logger = logging.getLogger(__name__)

DEFAULT_M_STEPS: int = 20
DEFAULT_ATTR_THRESHOLD: float = 0.2


# ---------------------------------------------------------------------------
# Core attribution function
# ---------------------------------------------------------------------------

def compute_ig_attribution(
    manager: QwenMLPHookManager,
    input_ids: torch.Tensor,
    answer_token_id: int,
    target_token_pos: Optional[int] = None,
    m_steps: int = DEFAULT_M_STEPS,
    target_mode: str = "first_token_prob",
    layer_by_layer: bool = False,
    answer_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution for every FFN neuron.

    Parameters
    ----------
    manager:
        QwenMLPHookManager with all MLP layers wrapped.
    input_ids:
        Tokenised prompt tensor, shape [1, seq_len], on the model's device.
    answer_token_id:
        Token id of the first answer token.
    target_token_pos:
        Prompt position at which to measure the answer probability.
        None (default) → last prompt token position (paper-compatible).
    m_steps:
        Number of Riemann approximation steps (default 20, paper value).
    target_mode:
        "first_token_prob" (default, paper-compatible) or "sequence_logprob".
    layer_by_layer:
        If True, run exact layer-by-layer IG (O(m·L) passes, slower but exact).
    answer_token_ids:
        Required when target_mode="sequence_logprob".

    Returns
    -------
    attr_matrix : torch.Tensor
        Shape [num_layers, intermediate_size], dtype float32.
    """
    if not manager._is_wrapped:
        raise RuntimeError("Call manager.wrap_all_mlps() before computing attribution.")

    if target_mode == "sequence_logprob" and not answer_token_ids:
        raise ValueError(
            "answer_token_ids must be provided when target_mode='sequence_logprob'."
        )

    num_layers = manager.num_layers
    intermediate_size = manager.intermediate_size
    device = input_ids.device

    seq_len = input_ids.shape[1]
    if target_token_pos is None:
        target_token_pos = seq_len - 1
    elif target_token_pos < 0:
        target_token_pos = seq_len + target_token_pos

    # Step 1: Baseline forward pass to cache z_l^0
    manager.cached_activations.clear()
    manager.clear_injections()
    with torch.no_grad():
        _ = manager.model(input_ids)

    baseline: List[torch.Tensor] = []
    for l in range(num_layers):
        z0 = manager.cached_activations[l]  # [1, seq_len, intermediate_size]
        baseline.append(z0.float().cpu())

    # Step 2: Riemann summation
    if layer_by_layer:
        attr_matrix = _ig_layer_by_layer(
            manager, input_ids, baseline, num_layers, intermediate_size,
            target_token_pos, answer_token_id, m_steps, target_mode,
            answer_token_ids, device,
        )
    else:
        attr_matrix = _ig_all_layers(
            manager, input_ids, baseline, num_layers, intermediate_size,
            target_token_pos, answer_token_id, m_steps, target_mode,
            answer_token_ids, device,
        )

    manager.clear_injections()
    return attr_matrix


def _get_target_score(
    model_output,
    target_token_pos: int,
    answer_token_id: int,
    target_mode: str,
    answer_token_ids: Optional[List[int]],
) -> torch.Tensor:
    logits = model_output.logits  # [B, T, V]

    if target_mode == "first_token_prob":
        log_probs = F.log_softmax(logits[0, target_token_pos, :].float(), dim=-1)
        return log_probs[answer_token_id]

    # sequence_logprob
    score = torch.tensor(0.0, device=logits.device)
    for offset, tok_id in enumerate(answer_token_ids):
        pos = target_token_pos + offset
        if pos >= logits.shape[1]:
            break
        lp = F.log_softmax(logits[0, pos, :].float(), dim=-1)
        score = score + lp[tok_id]
    return score


def _ig_all_layers(
    manager, input_ids, baseline, num_layers, intermediate_size,
    target_token_pos, answer_token_id, m_steps, target_mode,
    answer_token_ids, device,
) -> torch.Tensor:
    """
    Path-IG: scale all layers' z_l simultaneously at each step.
    Requires m_steps total forward passes (fastest variant).
    """
    grad_sums = [torch.zeros(intermediate_size, dtype=torch.float32) for _ in range(num_layers)]

    for k in range(1, m_steps + 1):
        alpha = k / m_steps

        z_ks: List[torch.Tensor] = []
        for l in range(num_layers):
            z_k = (alpha * baseline[l]).to(device=device, dtype=manager.model.dtype)
            z_k.requires_grad_(True)
            manager.inject_activation(l, z_k)
            z_ks.append(z_k)

        output = manager.model(input_ids)
        score = _get_target_score(
            output, target_token_pos, answer_token_id, target_mode, answer_token_ids
        )
        score.backward()

        for l in range(num_layers):
            if z_ks[l].grad is not None:
                g = z_ks[l].grad[:, target_token_pos, :].float().cpu()
                grad_sums[l] += g.squeeze(0)

        manager.clear_injections()
        del output, score, z_ks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attr_matrix = torch.zeros(num_layers, intermediate_size, dtype=torch.float32)
    for l in range(num_layers):
        z0_l = baseline[l][0, target_token_pos, :]
        attr_matrix[l] = z0_l * (grad_sums[l] / m_steps)

    return attr_matrix


def _ig_layer_by_layer(
    manager, input_ids, baseline, num_layers, intermediate_size,
    target_token_pos, answer_token_id, m_steps, target_mode,
    answer_token_ids, device,
) -> torch.Tensor:
    """
    Exact layer-by-layer IG: for each layer l, scale only z_l while all
    other layers run with their natural activations.
    Requires num_layers × m_steps forward passes (paper-exact but slower).
    """
    attr_matrix = torch.zeros(num_layers, intermediate_size, dtype=torch.float32)

    for l in range(num_layers):
        z0 = baseline[l]  # [1, seq_len, intermediate_size], float32
        grad_sum = torch.zeros(intermediate_size, dtype=torch.float32)

        for k in range(1, m_steps + 1):
            alpha = k / m_steps
            z_k = (alpha * z0).to(device=device, dtype=manager.model.dtype)
            z_k.requires_grad_(True)
            manager.inject_activation(l, z_k)

            output = manager.model(input_ids)
            score = _get_target_score(
                output, target_token_pos, answer_token_id, target_mode, answer_token_ids
            )
            score.backward()

            if z_k.grad is not None:
                grad_sum += z_k.grad[:, target_token_pos, :].float().cpu().squeeze(0)

            manager.clear_injections()
            del output, score, z_k
            if device.type == "cuda":
                torch.cuda.empty_cache()

        z0_l = z0[0, target_token_pos, :]
        attr_matrix[l] = z0_l * (grad_sum / m_steps)
        logger.debug("  Layer %d/%d IG done", l + 1, num_layers)

    return attr_matrix


# ---------------------------------------------------------------------------
# Neuron selection
# ---------------------------------------------------------------------------

def select_knowledge_neurons(
    attr_matrix: torch.Tensor,
    threshold: float = DEFAULT_ATTR_THRESHOLD,
) -> NeuronList:
    """
    Return neurons whose IG attribution exceeds threshold.

    Per Ju et al. (2024):
        KN(x, a) = { (l, j) : Attr_{l,j}(x, a) > threshold }

    Parameters
    ----------
    attr_matrix:
        Shape [num_layers, intermediate_size], float32.
    threshold:
        Attribution threshold (paper value: 0.2).

    Returns
    -------
    List of (layer_idx, neuron_idx) tuples sorted by attribution (descending).
    """
    indices = (attr_matrix > threshold).nonzero(as_tuple=False)
    if indices.numel() == 0:
        return []

    scored = sorted(
        [(int(indices[i, 0]), int(indices[i, 1]),
          float(attr_matrix[indices[i, 0], indices[i, 1]]))
         for i in range(len(indices))],
        key=lambda t: -t[2],
    )
    return [(l, j) for l, j, _ in scored]


def intersect_neuron_sets(*neuron_sets: NeuronList) -> NeuronList:
    """
    Compute the set intersection of multiple neuron lists.

    Implements:
        SKN_i = KN(q_i^1, e3) ∩ KN(q_i^2, e3) ∩ KN(q_i^3, e3)

    per Ju et al. (2024).  If only one set is supplied it is returned as-is
    with a logged warning about the deviation from the paper.
    """
    if len(neuron_sets) == 0:
        return []
    if len(neuron_sets) == 1:
        logger.warning(
            "intersect_neuron_sets: only one prompt variant available; "
            "returning KN(q, e3) directly without intersection. "
            "Deviation from Ju et al. paper (which uses intersection over 3 variants)."
        )
        return list(neuron_sets[0])

    sets = [set(map(tuple, s)) for s in neuron_sets]
    intersection = sets[0]
    for s in sets[1:]:
        intersection = intersection & s
    return [list(t) for t in intersection]


def layer_histogram(neurons: NeuronList, num_layers: int) -> dict:
    """Return {layer_idx: count} for layer-wise neuron distribution."""
    hist: dict = {str(l): 0 for l in range(num_layers)}
    for l, _j in neurons:
        hist[str(l)] = hist.get(str(l), 0) + 1
    return hist
