"""
Qwen MLP hook manager for caching and ablating intermediate activations.

Supports Qwen2 / Qwen2.5 / Qwen3 causal language models whose MLP layers
have the SwiGLU structure:

    z_l = act_fn(gate_proj(x_l)) * up_proj(x_l)
    mlp_out_l = down_proj(z_l)

Each scalar dimension j of the intermediate tensor z_l is treated as one
FFN neuron (l, j) in the sense of Dai et al. (2022) "Knowledge Neurons in
Pretrained Transformers" and Ju et al. (2024).

Three operating modes
---------------------
CACHE   – forward pass runs normally; z_l is detach-copied into
          cached_activations[l] for every layer.

ABLATE  – same as CACHE but additionally zeroes out selected neurons
          (l, j) at a chosen token position before passing z_l into
          down_proj.  Used at inference time after neuron identification.

INJECT  – replaces z_l for selected layers with a caller-supplied tensor
          (requires_grad may be True).  Used during Integrated Gradients
          computation so gradients flow through the injected tensor.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

NeuronList = List[Tuple[int, int]]   # list of (layer_idx, neuron_idx)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_qwen_model(
    model_name: str = "Qwen/Qwen3-1.7B",
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """
    Load a Qwen causal language model and its tokenizer.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. "Qwen/Qwen3-1.7B" or
        "Qwen/Qwen2.5-1.5B-Instruct".
    device:
        "auto" lets HuggingFace pick the best device(s); pass "cuda" or
        "cpu" to override.
    torch_dtype:
        Weight dtype.  bfloat16 is recommended for Qwen3; float16 also works.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    logger.info(
        "Loaded %s  |  layers=%d  |  hidden=%d  |  intermediate=%d  |  dtype=%s",
        model_name,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.intermediate_size,
        torch_dtype,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class QwenMLPHookManager:
    """
    Wraps every MLP layer in a Qwen2/Qwen2.5/Qwen3 model with a lightweight
    monkey-patched forward that enables caching, ablation, and injection of
    the intermediate activation z_l.

    Usage
    -----
    ::

        manager = QwenMLPHookManager(model)
        manager.wrap_all_mlps()

        # Caching forward pass
        with torch.no_grad():
            out = model(input_ids)
        z3 = manager.cached_activations[3]   # [B, T, intermediate_size]

        # Erasing inference
        manager.set_ablate_neurons([(3, 42), (7, 100)], token_pos=-1)
        with torch.no_grad():
            out_erased = model(input_ids)
        manager.clear_ablations()

        manager.restore()
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self._patches: List[Tuple[int, nn.Module, object]] = []

        self.cached_activations: Dict[int, torch.Tensor] = {}

        self._ablate_neurons: Dict[int, Set[int]] = {}
        self._ablate_token_pos: Optional[int] = -1
        self._inject_activations: Dict[int, torch.Tensor] = {}

        self._is_wrapped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wrap_all_mlps(self) -> None:
        """Install wrapper forwards on every MLP layer (idempotent)."""
        if self._is_wrapped:
            return
        layers = self._get_layers()
        for layer_idx, layer in enumerate(layers):
            mlp = self._get_mlp(layer, layer_idx)
            self._verify_qwen_mlp(mlp, layer_idx)
            self._wrap_single_mlp(layer_idx, mlp)
        self._is_wrapped = True
        logger.debug("QwenMLPHookManager: wrapped %d MLP layers", len(layers))

    def restore(self) -> None:
        """Remove all wrappers and restore original forwards."""
        for _layer_idx, mlp, original_forward in self._patches:
            mlp.forward = original_forward
        self._patches.clear()
        self.cached_activations.clear()
        self._ablate_neurons.clear()
        self._inject_activations.clear()
        self._is_wrapped = False

    # ------------------------------------------------------------------
    # Ablation control
    # ------------------------------------------------------------------

    def set_ablate_neurons(
        self,
        neurons: NeuronList,
        token_pos: Optional[int] = -1,
    ) -> None:
        """
        Register shortcut neurons to zero out at the next forward pass.

        Parameters
        ----------
        neurons:
            List of (layer_idx, neuron_idx) pairs.
        token_pos:
            Token position to ablate. -1 = last prompt token (default,
            paper-compatible). None = all token positions.
        """
        self._ablate_neurons.clear()
        for layer_idx, neuron_idx in neurons:
            self._ablate_neurons.setdefault(layer_idx, set()).add(neuron_idx)
        self._ablate_token_pos = token_pos

    def clear_ablations(self) -> None:
        self._ablate_neurons.clear()
        self._ablate_token_pos = -1

    # ------------------------------------------------------------------
    # Injection control (Integrated Gradients)
    # ------------------------------------------------------------------

    def inject_activation(self, layer_idx: int, z: torch.Tensor) -> None:
        """
        Replace z_l for layer_idx with z during the next forward pass.
        If z.requires_grad is True the gradient of the output w.r.t. z
        can be computed via z.grad after calling output.backward().
        """
        self._inject_activations[layer_idx] = z

    def clear_injections(self) -> None:
        self._inject_activations.clear()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_text(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        device: Optional[str] = None,
    ) -> str:
        """Tokenise prompt, run greedy decoding, return the continuation."""
        if device is None:
            device = str(next(self.model.parameters()).device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError(
            "Cannot locate transformer layers. "
            "Expected model.model.layers (Qwen2/Qwen3) or model.transformer.h (GPT-style)."
        )

    @staticmethod
    def _get_mlp(layer, layer_idx: int) -> nn.Module:
        if hasattr(layer, "mlp"):
            return layer.mlp
        raise AttributeError(
            f"Layer {layer_idx} has no 'mlp' attribute. "
            f"Sub-modules: {list(dict(layer.named_children()).keys())}"
        )

    @staticmethod
    def _verify_qwen_mlp(mlp: nn.Module, layer_idx: int) -> None:
        """Assert the MLP follows the Qwen2/Qwen3 SwiGLU structure."""
        required = {"gate_proj", "up_proj", "down_proj", "act_fn"}
        actual = set(dict(mlp.named_children()).keys()) | set(dir(mlp))
        missing = required - actual
        if missing:
            raise AttributeError(
                f"Layer {layer_idx} MLP is missing attributes {missing}. "
                f"Found: {list(dict(mlp.named_children()).keys())}"
            )

    def _wrap_single_mlp(self, layer_idx: int, mlp: nn.Module) -> None:
        """Replace mlp.forward with a wrapper that caches / ablates / injects z_l."""
        original_forward = mlp.forward
        manager = self
        l_idx = layer_idx

        def wrapped_forward(x: torch.Tensor) -> torch.Tensor:
            # 1. Compute or inject z_l
            if l_idx in manager._inject_activations:
                # IG mode: use caller-supplied tensor; do NOT update cache
                z = manager._inject_activations[l_idx]
            else:
                # Normal mode: compute and cache
                z = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
                manager.cached_activations[l_idx] = z.detach().clone()

            # 2. Ablate selected neuron dimensions
            if l_idx in manager._ablate_neurons:
                neuron_indices = sorted(manager._ablate_neurons[l_idx])
                if neuron_indices:
                    z = z.clone()
                    tok_pos = manager._ablate_token_pos
                    if tok_pos is None:
                        z[..., neuron_indices] = 0.0
                    else:
                        seq_len = z.shape[1]
                        actual_pos = tok_pos if tok_pos >= 0 else seq_len + tok_pos
                        if 0 <= actual_pos < seq_len:
                            z[:, actual_pos, neuron_indices] = 0.0
                        else:
                            warnings.warn(
                                f"Ablation token_pos={tok_pos} (resolved={actual_pos}) "
                                f"out of range for seq_len={seq_len}; skipping.",
                                RuntimeWarning, stacklevel=2,
                            )

            # 3. Project back to hidden size
            return mlp.down_proj(z)

        mlp.forward = wrapped_forward
        self._patches.append((layer_idx, mlp, original_forward))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self._patches) if self._is_wrapped else len(self._get_layers())

    @property
    def intermediate_size(self) -> int:
        return self.model.config.intermediate_size

    def __repr__(self) -> str:
        return (
            f"QwenMLPHookManager(model={type(self.model).__name__}, "
            f"layers={self.num_layers}, "
            f"intermediate_size={self.intermediate_size}, "
            f"wrapped={self._is_wrapped})"
        )
