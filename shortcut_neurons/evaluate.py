"""
Evaluation metrics for the shortcut neuron erasing pipeline.

Five metrics:

1. Multi-hop accuracy       — does the model still answer correctly?
2. Shortcut failure rate    — does the shortcut persist on ablated prompts?
3. Single-hop retention     — are unrelated single-hop facts preserved?
4. Bridge consistency (CoT) — does CoT output mention the correct bridge e2?
5. Ablated-prompt score     — can the model still answer with e1/bridge masked?
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .data import MultiHopExample
from .erase import erase_neurons_inference, build_global_neuron_set, EraseMode
from .modeling_qwen_hooks import QwenMLPHookManager, NeuronList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String matching helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _answer_matches(prediction: str, gold: str) -> bool:
    p = _normalize(prediction)
    g = _normalize(gold)
    return g in p or p in g or p == g


def _contains_entity(text: str, entity: str) -> bool:
    return _normalize(entity) in _normalize(text)


# ---------------------------------------------------------------------------
# Metric 1: Multi-hop accuracy
# ---------------------------------------------------------------------------

def evaluate_multihop_accuracy(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    locate_results: List[Dict[str, Any]],
    erase_mode: EraseMode = EraseMode.LOCAL,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute multi-hop answer accuracy before and after erasing."""
    if device is None:
        device = str(next(manager.model.parameters()).device)

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    correct_before = correct_after = 0
    details = []

    for ex, loc in zip(examples, locate_results):
        pred_before = loc.get("prediction_before", "") or manager.generate_text(
            tokenizer, ex.multi_hop_prompt, max_new_tokens=max_new_tokens, device=device
        )
        neurons = (
            [(int(l), int(j)) for l, j in loc.get("shortcut_neurons", [])]
            if erase_mode == EraseMode.LOCAL else global_neurons
        )
        pred_after = erase_neurons_inference(
            manager=manager, tokenizer=tokenizer,
            prompt=ex.multi_hop_prompt, shortcut_neurons=neurons,
            token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
        )
        cb = _answer_matches(pred_before, ex.answer)
        ca = _answer_matches(pred_after, ex.answer)
        correct_before += cb
        correct_after  += ca
        details.append({
            "id": ex.id, "answer": ex.answer,
            "prediction_before": pred_before, "prediction_after": pred_after,
            "correct_before": cb, "correct_after": ca,
        })

    n = len(examples)
    acc_before = correct_before / n if n else 0.0
    acc_after  = correct_after  / n if n else 0.0
    return {
        "before_accuracy": acc_before,
        "after_accuracy":  acc_after,
        "delta":           acc_after - acc_before,
        "n":               n,
        "details":         details,
    }


# ---------------------------------------------------------------------------
# Metric 2: Shortcut failure rate
# ---------------------------------------------------------------------------

def evaluate_shortcut_failure_rate(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    locate_results: List[Dict[str, Any]],
    erase_mode: EraseMode = EraseMode.LOCAL,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure how often the model answers ablated prompts correctly (shortcut
    still active) before and after erasing.  A decrease indicates shortcut
    suppression.
    """
    if device is None:
        device = str(next(manager.model.parameters()).device)

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    shortcut_active_before = shortcut_active_after = 0
    details = []

    for ex, loc in zip(examples, locate_results):
        ablated_prompt = ex.ablated_prompt_no_subject or ex.multi_hop_prompt
        pred_before = manager.generate_text(
            tokenizer, ablated_prompt, max_new_tokens=max_new_tokens, device=device,
        )
        shortcut_before = _answer_matches(pred_before, ex.answer)
        neurons = (
            [(int(l), int(j)) for l, j in loc.get("shortcut_neurons", [])]
            if erase_mode == EraseMode.LOCAL else global_neurons
        )
        pred_after = erase_neurons_inference(
            manager=manager, tokenizer=tokenizer,
            prompt=ablated_prompt, shortcut_neurons=neurons,
            token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
        )
        shortcut_after = _answer_matches(pred_after, ex.answer)
        shortcut_active_before += shortcut_before
        shortcut_active_after  += shortcut_after
        details.append({
            "id": ex.id, "ablated_prompt": ablated_prompt,
            "answer": ex.answer,
            "prediction_before": pred_before, "prediction_after": pred_after,
            "shortcut_active_before": shortcut_before,
            "shortcut_active_after":  shortcut_after,
        })

    n = len(examples)
    fr_before = shortcut_active_before / n if n else 0.0
    fr_after  = shortcut_active_after  / n if n else 0.0
    return {
        "failure_rate_before":  fr_before,
        "failure_rate_after":   fr_after,
        "shortcut_suppression": fr_before - fr_after,
        "n":                    n,
        "details":              details,
    }


# ---------------------------------------------------------------------------
# Metric 3: Single-hop retention
# ---------------------------------------------------------------------------

def evaluate_single_hop_retention(
    manager: QwenMLPHookManager,
    tokenizer,
    single_hop_examples: List[Dict[str, Any]],
    neurons_to_erase: NeuronList,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Test whether erasing shortcut neurons disrupts unrelated single-hop facts.

    Parameters
    ----------
    single_hop_examples:
        List of {"question": ..., "answer": ...} dicts.
    neurons_to_erase:
        The global (or representative) shortcut neuron set.
    """
    if device is None:
        device = str(next(manager.model.parameters()).device)

    correct_before = correct_after = 0
    details = []

    for sh in single_hop_examples:
        q = sh["question"]
        a = str(sh["answer"])
        pred_before = manager.generate_text(
            tokenizer, q, max_new_tokens=max_new_tokens, device=device,
        )
        pred_after = erase_neurons_inference(
            manager=manager, tokenizer=tokenizer,
            prompt=q, shortcut_neurons=neurons_to_erase,
            token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
        )
        cb = _answer_matches(pred_before, a)
        ca = _answer_matches(pred_after,  a)
        correct_before += cb
        correct_after  += ca
        details.append({
            "question": q, "answer": a,
            "prediction_before": pred_before, "prediction_after": pred_after,
            "correct_before": cb, "correct_after": ca,
        })

    n = len(single_hop_examples)
    acc_before = correct_before / n if n else 0.0
    acc_after  = correct_after  / n if n else 0.0
    return {
        "before_accuracy": acc_before,
        "after_accuracy":  acc_after,
        "degradation":     acc_before - acc_after,
        "n":               n,
        "details":         details,
    }


# ---------------------------------------------------------------------------
# Metric 4: Bridge consistency (CoT)
# ---------------------------------------------------------------------------

def evaluate_bridge_consistency(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    locate_results: List[Dict[str, Any]],
    cot_template: str = "Let's think step by step. {question}",
    erase_mode: EraseMode = EraseMode.LOCAL,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 100,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check whether the model correctly identifies the bridge entity e2 in a
    chain-of-thought response.  Skips examples where bridge is None.
    """
    if device is None:
        device = str(next(manager.model.parameters()).device)

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    bridge_before = bridge_after = evaluated = 0
    details = []

    for ex, loc in zip(examples, locate_results):
        if not ex.bridge:
            continue
        evaluated += 1
        cot_prompt_str = cot_template.format(question=ex.multi_hop_prompt)
        pred_before = manager.generate_text(
            tokenizer, cot_prompt_str, max_new_tokens=max_new_tokens, device=device,
        )
        neurons = (
            [(int(l), int(j)) for l, j in loc.get("shortcut_neurons", [])]
            if erase_mode == EraseMode.LOCAL else global_neurons
        )
        pred_after = erase_neurons_inference(
            manager=manager, tokenizer=tokenizer,
            prompt=cot_prompt_str, shortcut_neurons=neurons,
            token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
        )
        bb = _contains_entity(pred_before, ex.bridge)
        ba = _contains_entity(pred_after,  ex.bridge)
        bridge_before += bb
        bridge_after  += ba
        details.append({
            "id": ex.id, "bridge": ex.bridge,
            "cot_prompt": cot_prompt_str,
            "response_before": pred_before, "response_after": pred_after,
            "bridge_mentioned_before": bb, "bridge_mentioned_after": ba,
        })

    n = evaluated
    return {
        "bridge_accuracy_before": bridge_before / n if n else 0.0,
        "bridge_accuracy_after":  bridge_after  / n if n else 0.0,
        "n_evaluated":            n,
        "details":                details,
    }


# ---------------------------------------------------------------------------
# Metric 5: Ablated-prompt shortcut score
# ---------------------------------------------------------------------------

def evaluate_ablated_shortcut_score(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    locate_results: List[Dict[str, Any]],
    erase_mode: EraseMode = EraseMode.LOCAL,
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure shortcut score on both ablated prompt variants before and after erasing:
      - q(r2∘r1(∅)): subject masked
      - q(r2(∅)):    all entities masked
    """
    if device is None:
        device = str(next(manager.model.parameters()).device)

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    counts: Dict[str, int] = {
        "no_subject_before": 0, "no_subject_after": 0,
        "no_bridge_before":  0, "no_bridge_after":  0,
        "n": 0,
    }
    details = []

    for ex, loc in zip(examples, locate_results):
        counts["n"] += 1
        neurons = (
            [(int(l), int(j)) for l, j in loc.get("shortcut_neurons", [])]
            if erase_mode == EraseMode.LOCAL else global_neurons
        )
        row: Dict[str, Any] = {"id": ex.id}

        for field_name, count_key in [
            ("ablated_prompt_no_subject", "no_subject"),
            ("ablated_prompt_no_bridge",  "no_bridge"),
        ]:
            ablated = getattr(ex, field_name, None) or ex.multi_hop_prompt
            pred_before = manager.generate_text(
                tokenizer, ablated, max_new_tokens=max_new_tokens, device=device,
            )
            pred_after = erase_neurons_inference(
                manager=manager, tokenizer=tokenizer,
                prompt=ablated, shortcut_neurons=neurons,
                token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
            )
            b = _answer_matches(pred_before, ex.answer)
            a = _answer_matches(pred_after,  ex.answer)
            counts[f"{count_key}_before"] += b
            counts[f"{count_key}_after"]  += a
            row[f"{count_key}_pred_before"] = pred_before
            row[f"{count_key}_pred_after"]  = pred_after
            row[f"{count_key}_correct_before"] = b
            row[f"{count_key}_correct_after"]  = a

        details.append(row)

    n = counts["n"]
    return {
        "no_subject_score_before": counts["no_subject_before"] / n if n else 0.0,
        "no_subject_score_after":  counts["no_subject_after"]  / n if n else 0.0,
        "no_bridge_score_before":  counts["no_bridge_before"]  / n if n else 0.0,
        "no_bridge_score_after":   counts["no_bridge_after"]   / n if n else 0.0,
        "n":                       n,
        "details":                 details,
    }


# ---------------------------------------------------------------------------
# Combined pipeline evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    manager: QwenMLPHookManager,
    tokenizer,
    examples: List[MultiHopExample],
    locate_results: List[Dict[str, Any]],
    erase_mode: EraseMode = EraseMode.LOCAL,
    single_hop_examples: Optional[List[Dict[str, Any]]] = None,
    cot_template: str = "Let's think step by step. {question}",
    token_pos: Optional[int] = -1,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all five evaluation metrics and return a combined report."""
    if device is None:
        device = str(next(manager.model.parameters()).device)

    if single_hop_examples is None:
        single_hop_examples = [
            {"question": ex.single_hop_prompts[0], "answer": ex.answer}
            for ex in examples if ex.single_hop_prompts
        ]

    global_neurons: NeuronList = []
    if erase_mode == EraseMode.GLOBAL:
        global_neurons = build_global_neuron_set(locate_results)

    logger.info("Evaluating multi-hop accuracy …")
    multihop = evaluate_multihop_accuracy(
        manager, tokenizer, examples, locate_results,
        erase_mode=erase_mode, token_pos=token_pos,
        max_new_tokens=max_new_tokens, device=device,
    )

    logger.info("Evaluating shortcut failure rate …")
    shortcut = evaluate_shortcut_failure_rate(
        manager, tokenizer, examples, locate_results,
        erase_mode=erase_mode, token_pos=token_pos,
        max_new_tokens=max_new_tokens, device=device,
    )

    logger.info("Evaluating single-hop retention …")
    neurons_for_retention = global_neurons if erase_mode == EraseMode.GLOBAL else (
        [(int(l), int(j)) for l, j in locate_results[0].get("shortcut_neurons", [])]
        if locate_results else []
    )
    retention = evaluate_single_hop_retention(
        manager, tokenizer, single_hop_examples,
        neurons_to_erase=neurons_for_retention,
        token_pos=token_pos, max_new_tokens=max_new_tokens, device=device,
    )

    logger.info("Evaluating bridge consistency …")
    bridge = evaluate_bridge_consistency(
        manager, tokenizer, examples, locate_results,
        cot_template=cot_template,
        erase_mode=erase_mode, token_pos=token_pos,
        max_new_tokens=max_new_tokens, device=device,
    )

    logger.info("Evaluating ablated-prompt shortcut score …")
    ablated = evaluate_ablated_shortcut_score(
        manager, tokenizer, examples, locate_results,
        erase_mode=erase_mode, token_pos=token_pos,
        max_new_tokens=max_new_tokens, device=device,
    )

    return {
        "erase_mode":             str(erase_mode),
        "multihop_accuracy":      multihop,
        "shortcut_failure":       shortcut,
        "single_hop_retention":   retention,
        "bridge_consistency":     bridge,
        "ablated_shortcut_score": ablated,
    }


def save_evaluation_results(report: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to %s", output_path)
