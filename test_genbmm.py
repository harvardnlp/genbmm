import pytest
import genbmm
import torch


@pytest.fixture
def a():
    return torch.rand(10, 3, 4).cuda().requires_grad_(True)


@pytest.fixture
def b():
    return torch.rand(10, 4, 5).cuda().requires_grad_(True)


def test_logbmm(a, b):

    # Log-Sum-Exp
    c = genbmm.logbmm(a, b)

    # Equivalent
    a2 = a.unsqueeze(-1)
    b2 = b.unsqueeze(-3)
    c2 = (a2 + b2).logsumexp(-2)

    assert torch.isclose(c, c2).all()

    # Grad
    prob_a, prob_b = torch.autograd.grad(c.sum(), (a, b))
    prob_a2, prob_b2 = torch.autograd.grad(c2.sum(), (a, b))

    assert torch.isclose(prob_a, prob_a2).all()
    assert torch.isclose(prob_b, prob_b2).all()


def test_maxbmm(a, b):
    # Max
    c = genbmm.maxbmm(a, b)

    # Equivalent
    a2 = a.unsqueeze(-1)
    b2 = b.unsqueeze(-3)
    c2, _ = (a2 + b2).max(-2)

    assert torch.isclose(c, c2).all()

    # Grad
    argmax_a, argmax_b = torch.autograd.grad(c.sum(), (a, b))
    argmax_a2, argmax_b2 = torch.autograd.grad(c2.sum(), (a, b))

    assert torch.isclose(argmax_a, argmax_a2).all()
    assert torch.isclose(argmax_b, argmax_b2).all()


def test_samplebmm(a, b):
    # Sample
    c = genbmm.samplebmm(a, b)

    # Equivalent
    a2 = a.unsqueeze(-1)
    b2 = b.unsqueeze(-3)
    c2 = (a2 + b2).logsumexp(-2)

    assert torch.isclose(c, c2).all()

    # Grad
    sample_a, sample_b = torch.autograd.grad(c.sum(), (a, b))
    # c2 = (a + b).softmax(-2).sample(-2)


def test_prodmaxbmm(a, b):
    # Product-Max
    c = genbmm.prodmaxbmm(a, b)

    # Equivalent
    a2 = a.unsqueeze(-1)
    b2 = b.unsqueeze(-3)
    c2, _ = (a2 * b2).max(-2)

    assert torch.isclose(c, c2).all()

    # Grad
    grad_a, grad_b = torch.autograd.grad(c.sum(), (a, b))
    grad_a2, grad_b2 = torch.autograd.grad(c2.sum(), (a, b))

    assert torch.isclose(grad_a, grad_a2).all()
    assert torch.isclose(grad_b, grad_b2).all()
