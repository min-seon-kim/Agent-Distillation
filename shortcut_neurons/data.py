"""
Data loading and multi-hop schema management.

Implements the JSONL schema from Ju et al. (2024) and converts the project's
existing QA datasets into that format.  Also supports the Yang et al.-style
diagnostic ablated prompts for shortcut detection.

Schema
------
{
  "id": "...",
  "multi_hop_prompt": "...",          # Full multi-hop question  (q(r2∘r1(e1)))
  "single_hop_prompts": ["...", ...], # Prompt variants / paraphrases
  "answer": "e3 string",              # Terminal entity answer
  "bridge": "e2 string | null",       # Bridge entity (if available)
  "head_entity": "e1 string | null",  # Initial subject entity (if available)
  "terminal_entity": "e3 string",     # Same as answer, kept for clarity
  "shortcut_frequency": int,          # Times model answers from shortcut (0 by default)
  "templates": {
    "few_shot": "... optional ...",
    "cot":      "... optional ..."
  },
  "relation_object_shortcut_prone": false,  # Set by Yang et al. diagnostic
  "dataset": "...",                   # Source dataset name
  "hop_type": "factual | math"
}
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-class for a single multi-hop example
# ---------------------------------------------------------------------------

@dataclass
class MultiHopExample:
    """
    One multi-hop factual example in the Ju et al. 2024 schema.

    The shortcut is:  e1 ──(shortcut)──> e3
    The correct path: e1 ──r1──> e2 ──r2──> e3
    """
    id: str
    multi_hop_prompt: str
    single_hop_prompts: List[str]
    answer: str
    bridge: Optional[str]           # e2 (bridge entity) – may be None
    head_entity: Optional[str]      # e1 (initial subject) – may be None
    terminal_entity: str            # e3 (= answer)
    shortcut_frequency: int = 0
    templates: Dict[str, str] = field(default_factory=dict)
    relation_object_shortcut_prone: bool = False
    dataset: str = "unknown"
    hop_type: str = "factual"       # "factual" | "math"

    # Optional: ablated-prompt variants for shortcut diagnostics
    ablated_prompt_no_subject: Optional[str] = None    # q(r2∘r1(∅)) – subject masked
    ablated_prompt_no_bridge: Optional[str] = None     # q(r2(∅))    – bridge context removed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiHopExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------

def _load_project_json(path: str | Path) -> List[Dict[str, Any]]:
    """Load the project's standard {metadata, examples} JSON format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    raise ValueError(f"Unrecognised JSON format in {path}")


def _make_ablated_prompts(question: str, head_entity: Optional[str]) -> tuple[str, str]:
    """
    Create Yang et al.-style diagnostic ablated prompts.

    q(r2∘r1(∅)) – replace the head entity with "[ENTITY]" to see whether
                  the model can still produce e3 without knowing e1.
    q(r2(∅))    – strip any named entity, leaving only the relational
                  skeleton so the model cannot rely on any entity signal.

    When head_entity is unknown we fall back to heuristic token masking.
    """
    if head_entity and head_entity in question:
        ablated_subject = question.replace(head_entity, "[ENTITY]", 1)
    else:
        # Heuristic: mask the first proper-noun-like token group (title-cased
        # multi-word noun phrase at the start of the question).
        ablated_subject = re.sub(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', "[ENTITY]", question, count=1
        )

    # For q(r2(∅)) replace ALL named entities and produce only the relational
    # skeleton.  A coarse approximation: mask all title-cased phrases.
    ablated_relation = re.sub(
        r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', "[ENTITY]", question
    )

    return ablated_subject, ablated_relation


def _extract_head_entity_heuristic(question: str) -> Optional[str]:
    """
    Heuristically extract the first named entity from a question.

    Finds the first capitalised noun phrase that is not a question word.
    Returns None if nothing plausible is found.
    """
    skip = {"Who", "What", "Where", "When", "Which", "How", "Is", "Are", "Was", "Were",
            "Did", "Does", "Do", "The", "A", "An"}
    candidates = re.findall(r'\b([A-Z][a-zA-Z\']+(?:\s+[A-Z][a-zA-Z\']+)*)\b', question)
    for c in candidates:
        if c not in skip and len(c) > 2:
            return c
    return None


def _factual_to_multihop(raw: Dict[str, Any], dataset_name: str) -> MultiHopExample:
    """
    Convert a project QA example to MultiHopExample.

    Since most of the project's datasets do not expose explicit entity
    annotations (bridge, head_entity), those fields are populated
    heuristically or left as None.  Shortcut frequency is initialised to 0
    and computed separately by :func:`compute_shortcut_frequencies`.
    """
    qid = str(raw.get("id", ""))
    question = raw.get("question", "").strip()
    answer = str(raw.get("answer", "")).strip()

    head_entity = raw.get("head_entity") or _extract_head_entity_heuristic(question)
    bridge = raw.get("bridge", None)

    ablated_subject, ablated_relation = _make_ablated_prompts(question, head_entity)

    single_hop_prompts: List[str] = [question]
    if "single_hop_prompts" in raw and raw["single_hop_prompts"]:
        single_hop_prompts = raw["single_hop_prompts"]
    elif "paraphrases" in raw and raw["paraphrases"]:
        single_hop_prompts = raw["paraphrases"]
    else:
        single_hop_prompts = [
            question,
            f"Answer this question: {question}",
            f"Please answer: {question}",
        ]

    templates: Dict[str, str] = {}
    if "few_shot" in raw:
        templates["few_shot"] = raw["few_shot"]
    if "cot" in raw:
        templates["cot"] = raw["cot"]

    return MultiHopExample(
        id=qid,
        multi_hop_prompt=question,
        single_hop_prompts=single_hop_prompts,
        answer=answer,
        bridge=bridge,
        head_entity=head_entity,
        terminal_entity=answer,
        shortcut_frequency=raw.get("shortcut_frequency", 0),
        templates=templates,
        relation_object_shortcut_prone=raw.get("relation_object_shortcut_prone", False),
        dataset=dataset_name,
        hop_type="factual",
        ablated_prompt_no_subject=ablated_subject,
        ablated_prompt_no_bridge=ablated_relation,
    )


def _math_to_multihop(raw: Dict[str, Any], dataset_name: str) -> MultiHopExample:
    """
    Adapt a math/arithmetic example to the multi-hop schema.

    Mathematical reasoning shortcuts exist when a model skips intermediate
    computation steps and produces a (possibly hallucinated) direct answer.
    Here:
      e1 = the given numerical quantities / conditions in the problem
      e2 = intermediate computation result (unknown without working)
      e3 = final answer
    """
    qid = str(raw.get("id", ""))
    question = raw.get("question", "").strip()
    answer = str(raw.get("answer", "")).strip()

    ablated_subject = re.sub(r'\b\d[\d,\.]*\b', "[NUM]", question)
    ablated_relation = re.sub(r'\b\d[\d,\.]*\b|\b[A-Z][a-zA-Z]*\b', "[MASK]", question)

    single_hop_prompts = [
        question,
        f"Solve step by step: {question}",
        f"Calculate: {question}",
    ]

    return MultiHopExample(
        id=qid,
        multi_hop_prompt=question,
        single_hop_prompts=single_hop_prompts,
        answer=answer,
        bridge=None,
        head_entity=None,
        terminal_entity=answer,
        shortcut_frequency=raw.get("shortcut_frequency", 0),
        templates={},
        relation_object_shortcut_prone=False,
        dataset=dataset_name,
        hop_type="math",
        ablated_prompt_no_subject=ablated_subject,
        ablated_prompt_no_bridge=ablated_relation,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

_DATASET_CONVERTERS = {
    "2wikimultihopqa":  (_factual_to_multihop, "2wikimultihopqa"),
    "hotpotqa":         (_factual_to_multihop, "hotpotqa"),
    "musique":          (_factual_to_multihop, "musique"),
    "bamboogle":        (_factual_to_multihop, "bamboogle"),
    "MATH":             (_math_to_multihop,    "MATH"),
    "GSM-hard":         (_math_to_multihop,    "GSM-hard"),
    "AIME":             (_math_to_multihop,    "AIME"),
    "olymath":          (_math_to_multihop,    "olymath"),
}


def load_dataset_as_multihop(
    path: str | Path,
    dataset_type: Optional[str] = None,
    max_examples: Optional[int] = None,
    factual_only: bool = False,
) -> List[MultiHopExample]:
    """
    Load one of the project's QA/math datasets and return a list of
    MultiHopExample objects in the Ju et al. schema.
    """
    raws = _load_project_json(path)

    if dataset_type is None and raws:
        dataset_type = raws[0].get("dataset_name", "unknown")
        for key in _DATASET_CONVERTERS:
            if key in str(dataset_type):
                dataset_type = key
                break

    converter, ds_name = _DATASET_CONVERTERS.get(
        dataset_type, (_factual_to_multihop, dataset_type or "unknown")
    )

    if factual_only and ds_name in {"MATH", "GSM-hard", "AIME", "olymath"}:
        logger.info("Skipping math dataset %s (factual_only=True)", ds_name)
        return []

    examples = [converter(r, ds_name) for r in raws]
    if max_examples is not None:
        examples = examples[:max_examples]

    logger.info("Loaded %d examples from %s (%s)", len(examples), path, ds_name)
    return examples


def filter_shortcut_prone(
    examples: List[MultiHopExample],
    min_shortcut_frequency: int = 10,
) -> List[MultiHopExample]:
    """
    Return only examples where shortcut_frequency > min_shortcut_frequency.

    Per Ju et al. 2024, only high-cooccurrence multi-hop facts are used in
    the erasing experiments (default threshold = 10).
    """
    kept = [e for e in examples if e.shortcut_frequency > min_shortcut_frequency]
    excluded = len(examples) - len(kept)
    if excluded:
        logger.info(
            "filter_shortcut_prone: kept %d / %d examples "
            "(excluded %d with shortcut_frequency <= %d)",
            len(kept), len(examples), excluded, min_shortcut_frequency,
        )
    return kept


def compute_shortcut_frequencies(
    examples: List[MultiHopExample],
    model,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 1,
) -> List[MultiHopExample]:
    """
    Probe the model with ablated prompts to estimate shortcut_frequency.

    For each example:
      1. Run the model on ablated_prompt_no_subject (q(r2∘r1(∅))).
      2. Run the model on ablated_prompt_no_bridge  (q(r2(∅))).
      3. If the model predicts the correct answer from either ablated prompt,
         increment shortcut_frequency and flag relation_object_shortcut_prone.

    Detected shortcuts are set to frequency=11 so they exceed the default
    erasing threshold of > 10.
    """
    import torch

    model.eval()

    def _greedy_answer(prompt: str, max_new_tokens: int = 30) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _answer_matches(generated: str, expected: str) -> bool:
        g = generated.lower().strip()
        e = expected.lower().strip()
        return e in g or g in e

    updated = []
    for ex in examples:
        shortcut_count = ex.shortcut_frequency
        shortcut_prone = ex.relation_object_shortcut_prone

        for ablated_field in ["ablated_prompt_no_subject", "ablated_prompt_no_bridge"]:
            ablated = getattr(ex, ablated_field, None)
            if not ablated:
                continue
            try:
                gen = _greedy_answer(ablated)
                if _answer_matches(gen, ex.answer):
                    shortcut_count += 1
                    shortcut_prone = True
            except Exception as exc:
                logger.warning("compute_shortcut_frequencies: %s on id=%s", exc, ex.id)

        if shortcut_prone and shortcut_count <= 10:
            shortcut_count = 11

        ex.shortcut_frequency = shortcut_count
        ex.relation_object_shortcut_prone = shortcut_prone
        updated.append(ex)

    return updated


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_multihop_jsonl(path: str | Path) -> List[MultiHopExample]:
    """Load a JSONL file produced by save_multihop_jsonl."""
    path = Path(path)
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(MultiHopExample.from_dict(json.loads(line)))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def save_multihop_jsonl(examples: List[MultiHopExample], path: str | Path) -> None:
    """Save a list of MultiHopExample objects to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(examples), path)


# ---------------------------------------------------------------------------
# Built-in demo dataset
# ---------------------------------------------------------------------------

def make_demo_dataset(path: Optional[str | Path] = None) -> List[MultiHopExample]:
    """
    Return (and optionally save) a small hand-crafted dataset with full
    e1/e2/e3 annotations for end-to-end pipeline testing.

    Each entry has shortcut_frequency=15 so it passes the default filter (> 10).
    """
    records = [
        {
            "id": "demo_001",
            "question": "What is the capital of the country where Albert Einstein was born?",
            "answer": "Berlin",
            "bridge": "Germany",
            "head_entity": "Albert Einstein",
            "shortcut_frequency": 15,
            "single_hop_prompts": [
                "What is the capital of the country where Albert Einstein was born?",
                "Albert Einstein's birth country — what is its capital?",
                "In which city is the government of Albert Einstein's birth country located?",
            ],
        },
        {
            "id": "demo_002",
            "question": "Who is the president of the country that hosts the Eiffel Tower?",
            "answer": "Emmanuel Macron",
            "bridge": "France",
            "head_entity": "Eiffel Tower",
            "shortcut_frequency": 20,
            "single_hop_prompts": [
                "Who is the president of the country that hosts the Eiffel Tower?",
                "The Eiffel Tower is in which country, and who leads that country?",
                "What is the name of the current leader of the Eiffel Tower's country?",
            ],
        },
        {
            "id": "demo_003",
            "question": "What language is spoken in the birthplace of Marie Curie?",
            "answer": "Polish",
            "bridge": "Poland",
            "head_entity": "Marie Curie",
            "shortcut_frequency": 18,
            "single_hop_prompts": [
                "What language is spoken in the birthplace of Marie Curie?",
                "Marie Curie was born in which country, and what language do they speak?",
                "What is the official language of Marie Curie's country of birth?",
            ],
        },
        {
            "id": "demo_004",
            "question": "What currency does the country of the Colosseum use?",
            "answer": "Euro",
            "bridge": "Italy",
            "head_entity": "Colosseum",
            "shortcut_frequency": 12,
            "single_hop_prompts": [
                "What currency does the country of the Colosseum use?",
                "The Colosseum is located in which country, and what is its currency?",
                "In what country is the Colosseum, and what money do they use?",
            ],
        },
        {
            "id": "demo_005",
            "question": "What is the official language of the country where the Amazon River originates?",
            "answer": "Spanish",
            "bridge": "Peru",
            "head_entity": "Amazon River",
            "shortcut_frequency": 14,
            "single_hop_prompts": [
                "What is the official language of the country where the Amazon River originates?",
                "The Amazon River begins in which country, and what language is spoken there?",
                "In which country does the Amazon start, and what is its official language?",
            ],
        },
    ]

    examples = []
    for r in records:
        q = r["question"]
        he = r.get("head_entity")
        ablated_subj, ablated_rel = _make_ablated_prompts(q, he)
        ex = MultiHopExample(
            id=r["id"],
            multi_hop_prompt=q,
            single_hop_prompts=r["single_hop_prompts"],
            answer=r["answer"],
            bridge=r.get("bridge"),
            head_entity=he,
            terminal_entity=r["answer"],
            shortcut_frequency=r["shortcut_frequency"],
            templates={},
            relation_object_shortcut_prone=False,
            dataset="demo",
            hop_type="factual",
            ablated_prompt_no_subject=ablated_subj,
            ablated_prompt_no_bridge=ablated_rel,
        )
        examples.append(ex)

    if path is not None:
        save_multihop_jsonl(examples, path)

    return examples
