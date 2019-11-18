import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from

from .sparse import banddiag, BandedMatrix

lint = integers(min_value=2, max_value=10)

@given(lint, lint, lint, lint)
def test_sparse(batch, n, lu, ld):
    start = torch.rand(batch, n, n)
    band, _ = banddiag(start, lu, ld)

    # Check construction
    band2 = torch.zeros(batch, n, lu + ld + 1)
    for r in range(n):
        for c in range(n):
            if c == r or (r < c and c - r  <= lu) or (r > c and r - c <= ld):

                band2[:, c, lu - c + r]  = start[:, r, c]
    assert((band2 == band).all())

    # Check undo
    mid = torch.zeros(batch, n, n)
    for r in range(n):
        for c in range(n):
            if c == r or (r < c and c - r  <= lu) or (r > c and r - c <= ld):
                mid[:, r, c]  = start[:, r, c]

    banded = BandedMatrix(band, lu, ld)
    x = banded.to_dense()
    assert((mid == x).all())


    # Check transpose:
    banded = BandedMatrix(band, lu, ld)
    assert (mid.transpose(-2, -1) == banded.transpose().to_dense()).all()

    banded = BandedMatrix(band, lu, ld)

    plus = banded.op(banded, lambda a, b: a + b)

    assert plus.lu == banded.lu
    assert plus.ld == banded.ld
    assert (plus.data == banded.data + banded.data).all()
    assert ((mid + mid) == plus.to_dense()).all()

    banded = BandedMatrix(band, lu, ld)
    plus = banded.op(banded.transpose(), lambda a, b: a + b)

    assert plus.lu == max(banded.lu, banded.ld)
    assert plus.ld == max(banded.lu, banded.ld)
    assert ((mid.transpose(-2, -1) + mid) == plus.to_dense()).all()


    # Check multiplication
    b2 = banded.multiply_simple(banded)
    assert torch.isclose(torch.bmm(mid.transpose(-2, -1), mid), b2.to_dense()).all()


    # torch.rand_like
    # m.data = torch._like(m.data)
    m = banded.multiply_simple(banded)
    dense_m = m.to_dense()

    a = mid.transpose(-2, -1).clone().contiguous().requires_grad_(True)
    b = mid.clone().contiguous().requires_grad_(True)
    grad, b_grad = torch.autograd.grad(torch.bmm(a, b), (a,b), dense_m)


    b1 = banded.multiply_back_simple(banded, m)
    # print(banddiag(grad, b1.lu, b1.ld)[0])
    print(b1.to_dense())
    print(grad)
    assert (torch.isclose(b1.data, banddiag(grad, b1.lu, b1.ld)[0]).all()),\
        "%s\n %s"%(b1.to_dense(),
                   grad)


    # print(torch.autograd.grad(b2.data, torch.ones_like(b2.data), banded.data))
    # assert(False)




@given(lint, lint, lint, lint)
def test_sparse2(batch, n, lu, ld):
    start = torch.rand(batch, n, n)
    band, _ = banddiag(start, lu, ld)
    banded_x = BandedMatrix(band, lu, ld)
    x = banded_x.to_dense()

    start = torch.rand(batch, n, n)
    band, _ = banddiag(start, lu, ld)
    banded_y = BandedMatrix(band, lu, ld)
    y = banded_y.to_dense()

    b2 = banded_x.multiply_simple(banded_y)
    assert torch.isclose(torch.bmm(y.transpose(-2, -1), x), b2.to_dense()).all()

    b2 = banded_x.multiply_simple(banded_y.transpose())
    assert torch.isclose(torch.bmm(y, x), b2.to_dense()).all()

    # Grads
    m = banded_x.multiply_simple(banded_y)
    dense_m = m.to_dense()

    a = y.transpose(-2, -1).clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    grad, b_grad = torch.autograd.grad(torch.bmm(a, b), (a,b), dense_m)


    b1 = banded_x.multiply_back_simple(banded_y, m).transpose()

    assert (torch.isclose(b1.data,
                          banddiag(b_grad,
                                   b1.lu, b1.ld)[0]).all()),\
        "%s\n %s"%(b1.to_dense(),
                   grad)

    b2 = banded_y.transpose().multiply_back_simple(banded_x, m.transpose())
    print(b2.to_dense())
    print(grad)
    assert (torch.isclose(b2.data,
                          banddiag(grad,
                                   b2.lu, b2.ld)[0]).all()),\
        "%s\n %s"%(b2.to_dense(),
                   grad)

# @given(lint, lint, lint, lint)
# def test_sparse(batch, n, lu, ld):

#     lu = 1
#     ld = 2
#     batch = 2
#     n = 5
#     start = torch.rand(batch, n, n)
#     band, _ = banddiag(start, lu, ld)

#     # Check construction
#     band2 = torch.zeros(batch, n, lu + ld + 1)
#     for r in range(n):
#         for c in range(n):
#             if c == r or (r < c and c - r  <= lu) or (r > c and r - c <= ld):

#                 band2[:, c, lu - c + r]  = start[:, r, c]
#     assert((band2 == band).all())

#     # Check undo
#     mid = torch.zeros(batch, n, n)
#     for r in range(n):
#         for c in range(n):
#             if c == r or (r < c and c - r  <= lu) or (r > c and r - c <= ld):
#                 mid[:, r, c]  = start[:, r, c]

#     banded = BandedMatrix(band, lu, ld)
#     x = banded.to_dense()
#     assert((mid == x).all())


#     # Check transpose:
#     banded = BandedMatrix(band, lu, ld)
#     assert (mid.transpose(-2, -1) == banded.transpose().to_dense()).all()

#     banded = BandedMatrix(band, lu, ld)

#     plus = banded.op(banded, lambda a, b: a + b)

#     assert plus.lu == banded.lu
#     assert plus.ld == banded.ld
#     assert (plus.data == banded.data + banded.data).all()
#     assert ((mid + mid) == plus.to_dense()).all()

#     banded = BandedMatrix(band, lu, ld)
#     plus = banded.op(banded.transpose(), lambda a, b: a + b)

#     assert plus.lu == max(banded.lu, banded.ld)
#     assert plus.ld == max(banded.lu, banded.ld)
#     assert ((mid.transpose(-2, -1) + mid) == plus.to_dense()).all()


    # Check multiplication
    # b2 = banded.multiply(banded)
    # assert (torch.bmm(mid.transpose(-2, -1), mid) == b2.to_dense()).all()
