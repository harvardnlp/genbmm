import torch
try:
    import _genbmm
except:
    pass

def banddiag(orig_x, lu, ld, fill=0):
    s = list(orig_x.shape)
    x = orig_x
    # Upper pad
    if lu >= 0:
        s[-2] = lu
        x = torch.cat([torch.zeros(*s, device=x.device, dtype=x.dtype), x], dim=-2)
    # Lower pad
    if ld >= 0:
        s[-2] = ld
        x = torch.cat([x, torch.zeros(*s, device=x.device, dtype=x.dtype)], dim=-2)
    unf = x.unfold(-2, lu + ld+1, 1)
    return torch.diagonal(unf, 0, -3, -2).transpose(-2, -1), \
        x.narrow(-2, lu, orig_x.shape[-2])

def repdiag(x, lu, ld):
    s = list(x.shape)
    # Upper pad
    if ld >= 0:
        s[-2] = ld
        x = torch.cat([torch.zeros(*s, device=x.device, dtype=x.dtype), x], dim=-2)
    # Lower pad
    if lu >= 0:
        s[-2] = lu
        x = torch.cat([x, torch.zeros(*s, device=x.device, dtype=x.dtype)], dim=-2)
    unf = x.unfold(-2, lu +ld +1, 1)
    return torch.diagonal(unf, 0, -2, -1)

class BandedMatrix:
    def __init__(self, data, lu, ld, fill=0):
        batch, n, off = data.shape
        assert off == lu + ld + 1, "Offsets need to add up."
        self.data = data
        self.fill = fill
        self.lu, self.ld = lu, ld
        self.width = lu + ld + 1

    def _new(self, lu, ld):
        batch, n, off = self.data.shape
        data = torch.zeros(batch, n, ld + lu + 1,
                           dtype=self.data.dtype,
                           device=self.data.device).fill_(self.fill)
        return data

    def band_shift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(torch.cat([self.data[:, :, 1:],
                                       torch.zeros(batch, n, 1).fill_(fill)], 2),
                            self.lu-1, self.ld+1, self.fill)

    def band_unshift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(torch.cat([
            torch.zeros(batch, n, 1).fill_(fill),
            self.data[:, :, :-1]
        ], 2),
                            self.lu-1, self.ld+1, self.fill)


    def col_shift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(torch.cat([self.data[:, 1:, :],
                                       torch.zeros(batch, 1, off).fill_(fill)], 1),
                            self.lu-1, self.ld+1, self.fill)

    def col_unshift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(torch.cat([
            torch.zeros(batch, 1, off).fill_(fill),
            self.data[:, :-1, :],
        ], 1),
                            self.lu+1, self.ld-1, self.fill)

    def to_dense(self):
        batch, n, off = self.data.shape
        full = torch.zeros(batch, n, n, dtype=self.data.dtype,
                           device=self.data.device)
        full.fill_(self.fill)
        x2, x = banddiag(full, self.lu, self.ld)
        x2[:] = self.data
        return x

    def _expand(self, lu, ld):
        batch, n, off = self.data.shape
        data = self._new(lu, ld)
        s = lu - self.lu
        data[:, :, s: s+self.width] = self.data
        return BandedMatrix(data, lu, ld, self.fill)


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
        return BandedMatrix(data, lu, ld, self.fill)

    def transpose(self):
        batch, n, off = self.data.shape
        y2 = repdiag(self.data.flip(-1), self.lu, self.ld)
        assert y2.shape[1] == n
        return BandedMatrix(y2, self.ld, self.lu, self.fill)


    # def multiply(self, other):
    #     batch, n, off = self.data.shape
    #     assert other.data.shape[1] == n
    #     lu = self.lu + other.ld
    #     ld = self.ld + other.lu
    #     out, = _genbmm.forward_band(self.data, self.lu, self.ld,
    #                                 other.data, other.lu, other.ld, 3)
    #     return BandedMatrix(out, lu, ld, self.fill)


    def multiply(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        lu = self.lu + other.ld
        ld = self.ld + other.lu
        out, = bandedbmm(self.data, self.lu, self.ld,
                         other.data, other.lu, other.ld)
        return BandedMatrix(out, lu, ld, self.fill)

    def multiply_log(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        lu = self.lu + other.ld
        ld = self.ld + other.lu
        out, = bandedlogbmm(self.data, self.lu, self.ld,
                            other.data, other.lu, other.ld)
        return BandedMatrix(out, lu, ld, self.fill)

    def multiply_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld, self.fill)

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

    def multiply_log_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld, self.fill)

        for i in range(n):
            for j in range(result.width):
                o = i + (j - result.lu)
                if o < 0 or o >= n:
                    continue

                val = torch.zeros(batch)
                m = torch.zeros(batch).fill_(-1e9)
                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >=n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    m = torch.max(m, self.data[:, i, k] + other.data[:, o, k2])

                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >=n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += torch.exp(self.data[:, i, k] + other.data[:, o, k2] - m)

                data[:, i, j] = torch.log(val) + m
        return result



    def multiply_back(self, other, out, grad_out):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        lu = self.lu + other.ld
        ld = self.ld + other.lu
        grad_a, = _genbmm.backward_band(self.data, self.lu, self.ld,
                                        other.data, other.lu, other.ld,
                                        grad_out, grad_out, 3)
        grad_a = BandedMatrix(grad_a, self.lu, self.ld, self.fill)
        return grad_a

    def multiply_back_simple(self, other, grad_out):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        data = self._new(self.lu, self.ld)
        result = BandedMatrix(data, self.lu, self.ld, self.fill)

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
