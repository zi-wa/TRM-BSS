from typing import Tuple, Dict, Literal
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn

from models.layers import CastedEmbedding
from models.losses import IGNORE_LABEL_ID, log_stablemax
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)

# TRM + discrete diffusion on the answer canvas
# uniform: DiffusionGemma's uniform state diffusion
# masked: baseline

# Output logit of a locked token
LOCKED_LOGIT = 100.0


@dataclass
class TinyRecursiveReasoningModel_ACTV1DiffusionCarry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    canvas: torch.Tensor  # B x L
    canvas_probs: torch.Tensor  # B x L x V

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1DiffusionConfig(TinyRecursiveReasoningModel_ACTV1Config):
    diffusion: Literal["uniform", "masked"]
    confidence_threshold: float


class TinyRecursiveReasoningModel_ACTV1Diffusion_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1DiffusionConfig) -> None:
        super().__init__(config)
        num_canvas_tokens = self.config.vocab_size + 1 if self.config.diffusion == "masked" else self.config.vocab_size
        self.embed_canvas = CastedEmbedding(num_canvas_tokens, self.config.hidden_size, init_std=1.0 / self.embed_scale, cast_to=self.forward_dtype)

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], canvas: torch.Tensor, canvas_probs: torch.Tensor) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,)

        canvas_weight = self.embed_canvas.embedding_weight.to(self.forward_dtype)
        canvas_embeddings = self.embed_canvas(canvas) + canvas_probs.to(self.forward_dtype) @ canvas_weight[:self.config.vocab_size]
        canvas_embeddings = F.pad(canvas_embeddings, (0, 0, self.puzzle_emb_len, 0))  # no canvas on puzzle emb positions

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]) + self.embed_scale * canvas_embeddings

        # Recursion unchanged from trm.py
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1Diffusion(nn.Module):
    """ACT wrapper with a diffusion canvas."""
    config_cls = TinyRecursiveReasoningModel_ACTV1DiffusionConfig
    inner_cls = TinyRecursiveReasoningModel_ACTV1Diffusion_Inner

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = self.config_cls(**config_dict)
        self.inner = self.inner_cls(self.config)
        self.mask_id = self.config.vocab_size

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size, seq_len = batch["inputs"].shape

        return TinyRecursiveReasoningModel_ACTV1DiffusionCarry(
            inner_carry=self.inner.empty_carry(batch_size),

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),

            canvas=torch.empty((batch_size, seq_len), dtype=torch.int32),
            canvas_probs=torch.empty((batch_size, seq_len, self.config.vocab_size), dtype=torch.float32),

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def initial_canvas(self, labels: torch.Tensor) -> torch.Tensor:
        if self.config.diffusion == "masked":
            noise = torch.full_like(labels, self.mask_id)
        else:
            noise = torch.randint_like(labels, self.config.vocab_size)
        if not self.training:
            return noise

        # train: noise a random fraction t ~ U(0, 1) of the answer
        noise_level = torch.rand((labels.shape[0], 1), device=labels.device)
        answer = torch.where(labels == IGNORE_LABEL_ID, 0, labels)
        return torch.where(torch.rand(labels.shape, device=labels.device) < noise_level, noise, answer)

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1DiffusionCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1DiffusionCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # fresh canvas for new sequences
        canvas = torch.where(carry.halted.view(-1, 1), self.initial_canvas(new_current_data["labels"]), carry.canvas)
        canvas_probs = torch.where(carry.halted.view(-1, 1, 1), 0, carry.canvas_probs)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data, canvas, canvas_probs)

        if self.config.diffusion == "masked":
            # carry-over unmasking: locked tokens are the output
            locked = canvas != self.mask_id
            locked_one_hot = F.one_hot(torch.where(locked, canvas, 0).long(), self.config.vocab_size).bool()
            locked_logits = torch.where(locked_one_hot, LOCKED_LOGIT, -LOCKED_LOGIT).to(logits.dtype)
            logits = torch.where(locked.unsqueeze(-1), locked_logits, logits)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # denoise with stablemax probs, same as the loss
            probs = torch.exp(log_stablemax(logits.to(torch.float32), dim=-1))
            confidence, prediction = probs.max(dim=-1)
            confident = confidence >= self.config.confidence_threshold
            if self.config.diffusion == "masked":
                new_canvas = torch.where((canvas == self.mask_id) & confident, prediction.to(canvas.dtype), canvas)
            else:
                new_canvas = torch.where(confident, prediction.to(canvas.dtype), torch.randint_like(canvas, self.config.vocab_size))

            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
            # no Q-continue branch (broken in trm.py)
            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return TinyRecursiveReasoningModel_ACTV1DiffusionCarry(new_inner_carry, new_steps, halted, new_canvas, probs, new_current_data), outputs
