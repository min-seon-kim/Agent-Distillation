"""
Shortcut neuron identification pipeline (locate stage).

For each shortcut-prone multi-hop example this module:

1. Queries the model with up to three prompt variants.
2. Computes Integrated Gradients attribution for all MLP neurons.
3. Selects neurons above the attribution threshold.
4. Takes the intersection across prompt variants to obtain shortcut neurons.

Output schema (per example inside the saved JSON)
--------------------------------------------------
{
  "id": "...",
  "shortcut_frequency": 15,
  "prediction_before": "...",
  "shortcut_neurons": [[layer_idx, neuron_idx], ...],
  "num_shortcut_neurons": 42,
  "layer_histogram": {"0": 1, "1": 3, ...},
  "attr_values": [[layer_idx, neuron_idx, attr_value], ...],
  "num_prompt_variants": 3,
  "used_intersection": true
}

Top-level saved JSON
--------------------
{
  "model_name": "...",
  "m_steps": 20,
  "attr_threshold": 0.2,
  "min_shortcut_frequency": 10,
  "examples": [...]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .data import MultiHopExample
from .integrated_gradients import (
    compute_ig_attribution,
    select_knowledge_neurons,
    intersect_neuron_sets,
    layer_histogram,
    DEFAULT_M_STEPS,
    DEFAULT_ATTR_THRESHOLD,
)
from .modeling_qwen_hooks import QwenMLPHookManager, NeuronList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-example locate
# ---------------------------------------------------------------------------

def locate_shortcut_neurons(
    manager: QwenMLPHookManager,
    tokenizer,
    example: MultiHopExample,
    m_steps: int = DEFAULT_M_STEPS,
    attr_threshold: float = DEFAULT_ATTR_THRESHOLD,
    target_mode: str = "first_token_prob",
    device: Optional[str] = None,
    max_new_tokens: int = 30,
) -> Dict[str, Any]:
    """
    Identify shortcut neurons for a single multi-hop example.

    Returns a dict with keys:
        id, shortcut_frequency, prediction_before, shortcut_neurons,
        num_shortcut_neurons, layer_histogram, attr_values,
        num_prompt_variants, used_intersection
    """
    if device is None:
        device = str(next(manager.model.parameters()).device)

    # Baseline prediction (before erasing)
    prediction_before = manager.generate_text(
        tokenizer, example.multi_hop_prompt,
        max_new_tokens=max_new_tokens, device=device,
    )

    # Prompt variants (up to 3)
    prompts = example.single_hop_prompts[:3] if example.single_hop_prompts else [example.multi_hop_prompt]
    seen, dedup_prompts = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            dedup_prompts.append(p)
    if not dedup_prompts:
        dedup_prompts = [example.multi_hop_prompt]

    num_variants = len(dedup_prompts)
    if num_variants < 3:
        logger.warning(
            "[%s] Only %d prompt variant(s) available (Ju et al. use 3). "
            "Intersection will be taken over fewer variants. "
            "This deviates from the paper's erasing setup.",
            example.id, num_variants,
        )

    # Tokenise answer
    answer_str = example.answer.strip()
    answer_ids = tokenizer(
        " " + answer_str, add_special_tokens=False, return_tensors="pt",
    )["input_ids"][0].tolist()

    if not answer_ids:
        logger.warning("[%s] Empty answer token ids; skipping.", example.id)
        return _empty_result(example.id, example.shortcut_frequency, prediction_before)

    first_answer_token_id = answer_ids[0]

    # Compute IG for each prompt variant
    kn_per_variant: List[NeuronList] = []
    attr_per_variant: List[torch.Tensor] = []

    for v_idx, prompt in enumerate(dedup_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            attr_matrix = compute_ig_attribution(
                manager=manager,
                input_ids=inputs["input_ids"],
                answer_token_id=first_answer_token_id,
                target_token_pos=None,
                m_steps=m_steps,
                target_mode=target_mode,
                answer_token_ids=answer_ids if target_mode == "sequence_logprob" else None,
            )
        except Exception as exc:
            logger.error("[%s] IG failed on variant %d: %s; skipping.", example.id, v_idx, exc)
            continue

        kn = select_knowledge_neurons(attr_matrix, threshold=attr_threshold)
        kn_per_variant.append(kn)
        attr_per_variant.append(attr_matrix)
        logger.debug("[%s] Variant %d: %d neurons above threshold", example.id, v_idx, len(kn))

    if not kn_per_variant:
        logger.warning("[%s] All IG computations failed; returning empty result.", example.id)
        return _empty_result(example.id, example.shortcut_frequency, prediction_before)

    # Intersect → SKN_i
    used_intersection = len(kn_per_variant) > 1
    shortcut_neurons = intersect_neuron_sets(*kn_per_variant)

    if not shortcut_neurons and len(kn_per_variant) > 1:
        logger.warning(
            "[%s] Intersection is empty; falling back to union of variant neurons.",
            example.id,
        )
        union = set()
        for kn in kn_per_variant:
            union.update(map(tuple, kn))
        shortcut_neurons = [list(t) for t in union]

    # Attribution values for saved neurons
    attr_values: List = []
    if attr_per_variant:
        avg_attr = torch.stack(attr_per_variant, dim=0).mean(dim=0)
        for l, j in shortcut_neurons:
            attr_values.append([int(l), int(j), float(avg_attr[l, j])])
    attr_values.sort(key=lambda x: -x[2])

    hist = layer_histogram(shortcut_neurons, manager.num_layers)

    return {
        "id": example.id,
        "shortcut_frequency": example.shortcut_frequency,
        "prediction_before": prediction_before,
        "shortcut_neurons": [[int(l), int(j)] for l, j in shortcut_neurons],
        "num_shortcut_neurons": len(shortcut_neurons),
        "layer_histogram": hist,
        "attr_values": attr_values,
        "num_prompt_variants": len(kn_per_variant),
        "used_intersection": used_intersection,
    }


def _empty_result(ex_id: str, freq: int, pred: str) -> Dict[str, Any]:
    return {
        "id": ex_id,
        "shortcut_frequency": freq,
        "prediction_before": pred,
        "shortcut_neurons": [],
        "num_shortcut_neurons": 0,
        "layer_histogram": {},
        "attr_values": [],
        "num_prompt_variants": 0,
        "used_intersection": False,
    }


# ---------------------------------------------------------------------------
# Batch locate
# ---------------------------------------------------------------------------

def locate_shortcut_neurons_batch(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    m_steps: int = DEFAULT_M_STEPS,
    attr_threshold: float = DEFAULT_ATTR_THRESHOLD,
    target_mode: str = "first_token_prob",
    device: Optional[str] = None,
    max_new_tokens: int = 30,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run locate_shortcut_neurons for each example in the list."""
    results = []
    n = len(examples)
    for i, ex in enumerate(examples):
        if verbose:
            logger.info("[%d/%d] Locating shortcut neurons for id=%s", i + 1, n, ex.id)
        try:
            res = locate_shortcut_neurons(
                manager=manager, tokenizer=tokenizer, example=ex,
                m_steps=m_steps, attr_threshold=attr_threshold,
                target_mode=target_mode, device=device,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.error("[%d/%d] id=%s failed: %s", i + 1, n, ex.id, exc)
            res = _empty_result(ex.id, ex.shortcut_frequency, "ERROR")
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_shortcut_neuron_results(
    results: List[Dict[str, Any]],
    output_path: str | Path,
    model_name: str,
    m_steps: int = DEFAULT_M_STEPS,
    attr_threshold: float = DEFAULT_ATTR_THRESHOLD,
    min_shortcut_frequency: int = 10,
) -> None:
    """
    Save shortcut neuron identification results to JSON.

    Structure:
    {
      "model_name": "...",
      "m_steps": 20,
      "attr_threshold": 0.2,
      "min_shortcut_frequency": 10,
      "examples": [...]
    }
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "m_steps": m_steps,
        "attr_threshold": attr_threshold,
        "min_shortcut_frequency": min_shortcut_frequency,
        "examples": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d results to %s", len(results), output_path)


def load_shortcut_neuron_results(
    path: str | Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load a JSON file produced by save_shortcut_neuron_results.

    Returns (metadata_dict, examples_list).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {k: v for k, v in data.items() if k != "examples"}
    examples = data.get("examples", [])
    return meta, examples
