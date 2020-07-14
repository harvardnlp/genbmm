import torch
from hypothesis import given
from hypothesis.strategies import integers

from .genmul import logbmm

mint = integers(min_value=6, max_value=10)
lint = integers(min_value=0, max_value=10)
sint = integers(min_value=3, max_value=5)


@given(sint, mint, mint, mint)
def test_logbmm(batch, row, inner, col):
    a = torch.rand(batch, row, inner, requires_grad=True).cuda()
    b = torch.rand(batch, inner, col, requires_grad=True).cuda()

    c = (a[:, :, :, None] + b[:, None, :, :]).logsumexp(-2)
    c2 = logbmm(a, b)

    assert(torch.isclose(c, c2).all())

    g = torch.autograd.grad(c, (a, b), torch.rand(batch, row, col).cuda(), create_graph=True)
    g2 = torch.autograd.grad(c, (a, b), torch.rand(batch, row, col).cuda(), create_graph=True)

    for v1, v2 in zip(g, g2):
        assert(torch.isclose(v1, v2).all())

    
    h = torch.autograd.grad((g[0], g[1]), (a, b), (torch.rand(batch, row, inner).cuda(),
                                                   torch.rand(batch, inner, col).cuda()))
    h2 = torch.autograd.grad((g2[0], g2[0]) , (a, b), (torch.rand(batch, row, inner).cuda(),
                                                       torch.rand(batch, inner, col).cuda()))

    for v1, v2 in zip(h, h2):
        assert(torch.isclose(v1, v2).all())

    