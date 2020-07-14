import torch
from hypothesis import given
from hypothesis.strategies import integers

from .genmul import logbmm

mint = integers(min_value=6, max_value=10)
lint = integers(min_value=0, max_value=10)
sint = integers(min_value=3, max_value=5)


@given(sint, mint, mint, mint)
def test_logbmm(batch, row, inner, col):
    a = torch.rand(batch, row, inner).cuda()
    b = torch.rand(batch, inner, col).cuda()

    c = (a[:, :, :, None] + b[:, None, :, :]).logsumexp(-2)
    c2 = logbmm(a, b)

    assert(torch.isclose(c, c2).all())
