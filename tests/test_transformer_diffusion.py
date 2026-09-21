import torch

from models.losses import ACTLossHead
from models.recursive_reasoning.transformers_baseline import Model_ACTV2InnerCarry
from models.recursive_reasoning.transformer_diffusion import TransformerDiffusion

VOCAB_SIZE = 12
SEQ_LEN = 16
BATCH_SIZE = 4
HALT_MAX_STEPS = 4


def make_model():
    torch.manual_seed(0)
    return TransformerDiffusion(dict(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        num_puzzle_identifiers=3, puzzle_emb_ndim=32,
        H_cycles=1, H_layers=2, hidden_size=32, expansion=2, num_heads=2, pos_encodings="rope",
        halt_max_steps=HALT_MAX_STEPS, halt_exploration_prob=0.1,
        forward_dtype="float32", diffusion="uniform", confidence_threshold=0.5,
    ))


def make_batch():
    generator = torch.Generator().manual_seed(1)
    return {
        "inputs": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.int32),
        "labels": torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.int32),
        "puzzle_identifiers": torch.tensor([1, 2, 1, 2], dtype=torch.int32),
    }


def test_output_does_not_depend_on_previous_latent():
    model = make_model().eval()
    batch = make_batch()
    canvas = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.int32)
    canvas_probs = torch.zeros(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    carry = model.inner.reset_carry(torch.ones(BATCH_SIZE, dtype=torch.bool), model.inner.empty_carry(BATCH_SIZE))
    other_carry = Model_ACTV2InnerCarry(z_H=torch.randn_like(carry.z_H))
    _, logits, _ = model.inner(carry, batch, canvas, canvas_probs)
    _, other_logits, _ = model.inner(other_carry, batch, canvas, canvas_probs)

    assert torch.equal(logits, other_logits)


def test_canvas_embedding_receives_gradient():
    model = make_model().train()
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = make_batch()

    _, loss, _, _, _ = loss_head(carry=loss_head.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()

    assert model.inner.embed_canvas.embedding_weight.grad.abs().sum() > 0


def test_eval_denoises_for_halt_max_steps():
    model = make_model().eval()
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = make_batch()

    carry = loss_head.initial_carry(batch)
    inference_steps = 0
    with torch.inference_mode():
        while True:
            carry, _, _, preds, all_finish = loss_head(carry=carry, batch=batch, return_keys=["preds"])
            inference_steps += 1
            if all_finish:
                break

    assert inference_steps == HALT_MAX_STEPS
    assert preds["preds"].shape == (BATCH_SIZE, SEQ_LEN)
