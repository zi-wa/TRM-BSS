from typing import Tuple, Dict, Literal
import torch
import torch.nn.functional as F

from models.layers import CastedEmbedding
from models.recursive_reasoning.transformers_baseline import Model_ACTV2Config, Model_ACTV2InnerCarry, Model_ACTV2_Inner
from models.recursive_reasoning.trm_diffusion import TinyRecursiveReasoningModel_ACTV1Diffusion

# diffusion baseline without TRM: no recursion, no latent carry


class TransformerDiffusionConfig(Model_ACTV2Config):
    diffusion: Literal["uniform", "masked"]
    confidence_threshold: float


class TransformerDiffusion_Inner(Model_ACTV2_Inner):
    def __init__(self, config: TransformerDiffusionConfig) -> None:
        super().__init__(config)
        num_canvas_tokens = self.config.vocab_size + 1 if self.config.diffusion == "masked" else self.config.vocab_size
        self.embed_canvas = CastedEmbedding(num_canvas_tokens, self.config.hidden_size, init_std=1.0 / self.embed_scale, cast_to=self.forward_dtype)

    def forward(self, carry: Model_ACTV2InnerCarry, batch: Dict[str, torch.Tensor], canvas: torch.Tensor, canvas_probs: torch.Tensor) -> Tuple[Model_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)

        canvas_weight = self.embed_canvas.embedding_weight.to(self.forward_dtype)
        canvas_embeddings = self.embed_canvas(canvas) + canvas_probs.to(self.forward_dtype) @ canvas_weight[:self.config.vocab_size]
        canvas_embeddings = F.pad(canvas_embeddings, (0, 0, self.puzzle_emb_len, 0))

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"]) + self.embed_scale * canvas_embeddings

        # no latent carry, start from H_init
        z_H = self.H_level(self.H_init, input_embeddings, **seq_info)

        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return Model_ACTV2InnerCarry(z_H=z_H.detach()), output, (q_logits[..., 0], q_logits[..., 1])


class TransformerDiffusion(TinyRecursiveReasoningModel_ACTV1Diffusion):
    config_cls = TransformerDiffusionConfig
    inner_cls = TransformerDiffusion_Inner
