import torch
from adam_atan2_pytorch.foreach import AdamAtan2

LR = 1e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)


def original_adam_atan2_step(param, grad, exp_avg, exp_avg_sq, step):
    # fused adam-atan2 0.0.3 kernel (csrc/adam_atan2.cu)
    beta1, beta2 = BETAS
    param.mul_(1 - LR * WEIGHT_DECAY)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad * grad, 1 - beta2)
    denom = exp_avg_sq.sqrt() / (1 - beta2 ** step) ** 0.5
    param.sub_(LR / (1 - beta1 ** step) * torch.atan2(exp_avg, denom))


def make_optimizer(param, lr):
    # same args as pretrain.py; lr comes from the scheduler
    optimizer = AdamAtan2([param], lr=1e-8, weight_decay=WEIGHT_DECAY, betas=BETAS, a=1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer


def test_matches_original_adam_atan2_once_bias_correction_settles():
    torch.manual_seed(0)
    # float64 to keep rounding out
    param = torch.nn.Parameter(torch.randn(1024, dtype=torch.float64))
    exp_avg, exp_avg_sq = torch.zeros_like(param), torch.zeros_like(param)
    optimizer = make_optimizer(param, LR)

    for step in range(1, 151):
        grad = torch.randn(1024, dtype=torch.float64) * 0.01 + 0.002
        # same start param every step, compare the update rule only
        param_before = param.detach().clone()
        reference = param_before.clone()
        original_adam_atan2_step(reference, grad, exp_avg, exp_avg_sq, step)
        param.grad = grad
        optimizer.step()

    # bias correction differs (inside vs outside atan2) until beta1 ** step ~ 0
    # atol: weight decay and the atan2 term nearly cancel on some elements
    assert torch.allclose(param.detach() - param_before, reference - param_before, rtol=0, atol=1e-10)


def test_zero_lr_leaves_params_unchanged():
    param = torch.nn.Parameter(torch.randn(16))
    initial = param.detach().clone()
    optimizer = make_optimizer(param, 0.0)

    param.grad = torch.randn(16)
    optimizer.step()

    assert torch.equal(param.detach(), initial)
