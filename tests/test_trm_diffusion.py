import pytest
import torch

from models.losses import ACTLossHead, IGNORE_LABEL_ID, log_stablemax
from models.recursive_reasoning.trm_diffusion import TinyRecursiveReasoningModel_ACTV1Diffusion

VOCAB_SIZE = 12
MASK_ID = VOCAB_SIZE
SEQ_LEN = 16
BATCH_SIZE = 4
HALT_MAX_STEPS = 4

both_diffusions = pytest.mark.parametrize("diffusion", ["uniform", "masked"])


def make_model(diffusion, **overrides):
    config = dict(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
        num_puzzle_identifiers=3, puzzle_emb_ndim=32, puzzle_emb_len=2,
        H_cycles=2, L_cycles=2, H_layers=0, L_layers=1,
        hidden_size=32, expansion=2, num_heads=2, pos_encodings="rope",
        halt_max_steps=HALT_MAX_STEPS, halt_exploration_prob=0.1,
        forward_dtype="float32", diffusion=diffusion, confidence_threshold=0.5,
    )
    config.update(overrides)
    torch.manual_seed(0)
    return TinyRecursiveReasoningModel_ACTV1Diffusion(config)


def make_batch():
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.int32)
    labels = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.int32)
    labels[:, -3:] = IGNORE_LABEL_ID
    puzzle_identifiers = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
    return {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_identifiers}


@both_diffusions
def test_eval_canvas_does_not_read_labels(diffusion):
    model = make_model(diffusion).eval()
    labels = make_batch()["labels"]

    torch.manual_seed(0)
    canvas = model.initial_canvas(labels)
    torch.manual_seed(0)
    canvas_other_labels = model.initial_canvas(torch.zeros_like(labels))

    assert torch.equal(canvas, canvas_other_labels)
    if diffusion == "masked":
        assert (canvas == MASK_ID).all()
    else:
        assert canvas.min() >= 0 and canvas.max() < VOCAB_SIZE


def test_uniform_train_canvas_is_partially_noised_answer():
    model = make_model("uniform").train()
    labels = torch.randint(1, VOCAB_SIZE, (4096, SEQ_LEN), dtype=torch.int32)

    torch.manual_seed(0)
    canvas = model.initial_canvas(labels)

    # E[kept] = 0.5 + 0.5 / V for t ~ U(0, 1)
    kept = (canvas == labels).float().mean().item()
    assert abs(kept - (0.5 + 0.5 / VOCAB_SIZE)) < 0.02


def test_masked_train_canvas_is_answer_or_mask():
    model = make_model("masked").train()
    labels = torch.randint(1, VOCAB_SIZE, (4096, SEQ_LEN), dtype=torch.int32)

    torch.manual_seed(0)
    canvas = model.initial_canvas(labels)

    assert ((canvas == labels) | (canvas == MASK_ID)).all()
    assert abs((canvas == MASK_ID).float().mean().item() - 0.5) < 0.02


@both_diffusions
def test_zero_threshold_keeps_every_prediction(diffusion):
    model = make_model(diffusion, confidence_threshold=0.0).train()
    batch = make_batch()

    carry, outputs = model(model.initial_carry(batch), batch)

    assert torch.equal(carry.canvas, outputs["logits"].argmax(-1).to(carry.canvas.dtype))


def test_uniform_threshold_above_one_renoises_every_position():
    model = make_model("uniform", confidence_threshold=1.1).train()
    batch = make_batch()

    torch.manual_seed(0)
    carry, outputs = model(model.initial_carry(batch), batch)

    # noise matches only by chance (1 / V)
    matches = (carry.canvas == outputs["logits"].argmax(-1)).float().mean().item()
    assert matches < 0.3


def test_masked_threshold_above_one_unmasks_nothing():
    model = make_model("masked", confidence_threshold=1.1).eval()
    batch = make_batch()

    carry, _ = model(model.initial_carry(batch), batch)

    assert (carry.canvas == MASK_ID).all()


def test_masked_locked_tokens_stay_and_are_the_output():
    model = make_model("masked", confidence_threshold=0.0).eval()
    batch = make_batch()

    carry, _ = model(model.initial_carry(batch), batch)
    locked_canvas = carry.canvas.clone()
    assert (locked_canvas != MASK_ID).all()

    carry, outputs = model(carry, batch)

    assert torch.equal(carry.canvas, locked_canvas)
    assert torch.equal(outputs["logits"].argmax(-1).to(locked_canvas.dtype), locked_canvas)


def test_masked_loss_ignores_locked_positions():
    model = make_model("masked").train()
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = make_batch()

    carry, _, _, _, _ = loss_head(carry=loss_head.initial_carry(batch), batch=batch, return_keys=[])
    carry.canvas = torch.where(batch["labels"] == IGNORE_LABEL_ID, 0, batch["labels"])  # every position unmasked
    carry.halted = torch.zeros(BATCH_SIZE, dtype=torch.bool)

    _, loss, _, _, _ = loss_head(carry=carry, batch=batch, return_keys=[])
    loss.backward()

    assert model.inner.lm_head.weight.grad.abs().sum() == 0


@both_diffusions
def test_carry_keeps_stablemax_probs_for_self_conditioning(diffusion):
    model = make_model(diffusion).train()
    batch = make_batch()

    carry, outputs = model(model.initial_carry(batch), batch)

    expected_probs = torch.exp(log_stablemax(outputs["logits"].float(), dim=-1))
    assert torch.allclose(carry.canvas_probs, expected_probs)
    assert torch.allclose(carry.canvas_probs.sum(-1), torch.ones(BATCH_SIZE, SEQ_LEN))


@both_diffusions
def test_self_conditioning_changes_the_output(diffusion):
    model = make_model(diffusion).eval()
    batch = make_batch()
    inner_carry = model.inner.reset_carry(torch.ones(BATCH_SIZE, dtype=torch.bool), model.inner.empty_carry(BATCH_SIZE))
    canvas = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.int32)

    no_probs = torch.zeros(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    one_hot_probs = torch.nn.functional.one_hot(torch.full((BATCH_SIZE, SEQ_LEN), 5), VOCAB_SIZE).float()
    _, logits_without, _ = model.inner(inner_carry, batch, canvas, no_probs)
    _, logits_with, _ = model.inner(inner_carry, batch, canvas, one_hot_probs)

    assert not torch.allclose(logits_without, logits_with)


@both_diffusions
def test_canvas_embedding_receives_gradient(diffusion):
    model = make_model(diffusion).train()
    loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
    batch = make_batch()

    _, loss, _, _, _ = loss_head(carry=loss_head.initial_carry(batch), batch=batch, return_keys=[])
    loss.backward()

    assert model.inner.embed_canvas.embedding_weight.grad.abs().sum() > 0


@both_diffusions
def test_eval_denoises_for_halt_max_steps(diffusion):
    model = make_model(diffusion).eval()
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
