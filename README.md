
Cuda Kernels for generalized matrix multiplies.

```python

import genbmm

# Genbmm
a = torch.rand(10, 3, 4)
b = torch.rand(10, 4, 5)
c = genbmm.logbmm(a, b)

# Equivalent
a = a.unsqueeze(-1)
b = b.unsqueeze(-3)
c2 = (a + b).logsumexp(-2)

```

TODO:

- cpu
- tests
