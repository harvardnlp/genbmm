# genbmm

Cuda Kernels for generalized matrix multiplication in pytorch.

```python

import genbmm

a = torch.rand(10, 3, 4).cuda().requires_grad_(True)
b = torch.rand(10, 4, 5).cuda().requires_grad_(True)

# Log-Sum-Exp
c = genbmm.logbmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)
# Grad
prob_a, prob_b = torch.autograd.grad(c.sum(), (a, b))

# Max
c = genbmm.logbmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)
# Grad
argmax_a, argmax_b = torch.autograd.grad(c.sum(), (a, b))

# Sample
c = genbmm.logbmm(a, b)
# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)
# Grad
sample_a, sample_b = torch.autograd.grad(c.sum(), (a, b))

```

