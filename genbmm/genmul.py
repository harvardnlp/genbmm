import torch
from .sparse import BandedMatrix
try:
    import _genbmm
except ImportError:
    pass


class LogMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, = _genbmm.forward(a, b, 0)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, out = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(a, b, grad_output.contiguous(), out, 0)
        return grad_a, grad_b


class MaxMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 1)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a, b, grad_output.contiguous(), switches.float(), 1
        )
        return grad_a, grad_b


class SampleMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 2)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a, b, grad_output.contiguous(), switches.float(), 2
        )
        return grad_a, grad_b


class BandedMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld):
        a = a.contiguous()
        b = b.contiguous()
        out, = _genbmm.forward_band(a, a, a_lu, a_ld,
                                    b, b_lu, b_ld, 3)
        ctx.save_for_backward(a, b, out, torch.LongTensor([a_lu, a_ld, b_lu, b_ld]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld = bands.tolist()
        grad_a, grad_b = _genbmm.backward_band(
            a, a_lu, a_ld,
            b, b_lu, b_ld, grad_output.contiguous(), switches.float(), 3
        )
        return grad_a, grad_b


class BandedLogMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
        a = a.contiguous()
        b = b.contiguous()
        out, = _genbmm.forward_band(a, a_lu, a_ld,
                                    b, b_lu, b_ld, 0)
        ctx.save_for_backward(a, b, out,
                              torch.LongTensor([a_lu, a_ld, b_lu, b_ld, o_lu, o_ld]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld, o_lu, o_ld = bands.tolist()
        a = BandedMatrix(a, a_lu, a_ld, -1e9)
        b = BandedMatrix(b, b_lu, b_ld, -1e9)
        grad_output = BandedMatrix(grad_output, o_lu, o_ld, -1e9)
        switches = BandedMatrix(switches.float(), o_lu, o_ld, -1e9)

        grad_a, = _genbmm.backward_band(
            a.data, a.lu, a.ld, b.data, b.lu, b.ld,
            grad_output.contiguous(), switches.data, 0
        )
        grad_a = BandedMatrix(grad_a, o_lu, o_ld).transpose().data
        b  = b.tranpose()
        grad_b, = _genbmm.backward_band(
            b.data, b.lu, b.ld,
            a.data, a.lu, a.ld,
            grad_output.transpose().data.contiguous(),
            switches.transpose().data, 0
        )
        return grad_a, grad_b


logbmm = LogMatMul.apply
maxbmm = MaxMatMul.apply
samplebmm = SampleMatMul.apply
bandedbmm = BandedMul.apply
bandedlogbmm = BandedLogMul.apply
