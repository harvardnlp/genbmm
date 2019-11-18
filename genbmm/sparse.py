import torch

def banddiag(orig_x, lu, ld, fill=0):
    s = list(orig_x.shape)
    x = orig_x
    # Upper pad
    if lu > 0:
        s[-2] = lu
        x = torch.cat([torch.zeros(*s, device=x.device, dtype=x.dtype), x], dim=-2)
    # Lower pad
    if ld > 0:
        s[-2] = ld
        x = torch.cat([x, torch.zeros(*s, device=x.device, dtype=x.dtype)], dim=-2)

    return torch.diagonal(x.unfold(-2, lu + ld+1, 1), 0, -3, -2).transpose(-2, -1), \
        x.narrow(-2, lu, orig_x.shape[-2])

def repdiag(x, lu, ld):
    s = list(x.shape)
    # Upper pad
    if lu > 0:
        s[-2] = ld
        x = torch.cat([torch.zeros(*s, device=x.device, dtype=x.dtype), x], dim=-2)
    # Lower pad
    if ld > 0:
        s[-2] = lu
        x = torch.cat([x, torch.zeros(*s, device=x.device, dtype=x.dtype)], dim=-2)
    return torch.diagonal(x.unfold(-2, lu +ld +1, 1), 0, -2, -1)

class BandedMatrix:
    def __init__(self, data, lu=0, ld=0):
        batch, n, off = data.shape
        assert off == lu + ld + 1, "Offsets need to add up."
        self.data = data
        self.lu, self.ld = lu, ld
        self.width = lu + ld + 1

    def _new(self, lu, ld):
        batch, n, off = self.data.shape
        data = torch.zeros(batch, n, ld + lu + 1,
                           dtype=self.data.dtype,
                           device=self.data.device)
        return data

    def to_dense(self):
        batch, n, off = self.data.shape
        full = torch.zeros(batch, n, n, dtype=self.data.dtype,
                           device=self.data.device)
        x2, x = banddiag(full, self.lu, self.ld)
        x2[:] = self.data
        return x

    def _expand(self, lu, ld):
        batch, n, off = self.data.shape
        data = self._new(lu, ld)
        s = lu - self.lu
        data[:, :, s: s+self.width] = self.data
        return BandedMatrix(data, lu, ld)


    def op(self, other, op, zero=0):
        batch, n, off = self.data.shape
        lu = max(self.lu, other.lu)
        ld = max(self.ld, other.ld)
        data = self._new(lu, ld).fill_(zero)

        s1 = lu - self.lu
        data[:, :, s1: s1+self.width] = self.data

        s2 = lu - other.lu
        data[:, :, s2: s2+other.width] = op(data[:, :, s2: s2+other.width],
                                            other.data)
        return BandedMatrix(data, lu, ld)

    def transpose(self):
        batch, n, off = self.data.shape
        y2 = repdiag(self.data.flip(-1), self.lu, self.ld)
        return BandedMatrix(y2, self.ld, self.lu)


    def multiply_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld)

        for i in range(n):
            for j in range(result.width):
                o = i + (j - result.lu)
                if o < 0 or o >= n:
                    continue

                val = torch.zeros(batch)
                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >=n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += self.data[:, i, k] * other.data[:, o, k2]
                data[:, i, j] = val
        return result



    def multiply_back_simple(self, other, grad_out):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        data = self._new(self.lu, self.ld)
        result = BandedMatrix(data, self.lu, self.ld)

        for i in range(n):
            for j in range(self.width):
                o = i + (j - self.lu)
                val = torch.zeros(batch)
                for k in range(grad_out.width):
                    pos = i + (k - grad_out.lu)
                    if pos < 0 or pos >= n:
                        continue
                    k2 = (o - pos) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += other.data[:, pos, k2] * grad_out.data[:, i, k]
                data[:, i, j] = val
        return result.transpose()
