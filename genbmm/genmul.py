import torch

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
        out, = _genbmm.forward_band(a, a, a_lu, a_ld, b, b_lu, b_ld, 3)
        ctx.save_for_backward(a, b, out, torch.LongTensor([a_lu, a_ld, b_lu, b_ld]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld = bands.tolist()
        grad_a, grad_b = _genbmm.backward_band(
            a, a_lu, a_ld, b, b_lu, b_ld, grad_output.contiguous(), switches.float(), 3
        )
        return grad_a, grad_b


class BandedLogMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld):
        out, = _genbmm.forward_band(a, a_lu, a_ld, b, b_lu, b_ld, 0)
        ctx.save_for_backward(a, b, out, torch.LongTensor([a_lu, a_ld, b_lu, b_ld]))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld = bands.tolist()
        grad_a, grad_b = _genbmm.backward_band(
            a, a_lu, a_ld, b, b_lu, b_ld, grad_output.contiguous(), switches.float(), 0
        )
        return grad_a, grad_b


logbmm = LogMatMul.apply
maxbmm = MaxMatMul.apply
samplebmm = SampleMatMul.apply
bandedbmm = BandedMul.apply
bandedlogbmm = BandedLogMul.apply
