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
        grad_a2 = (a +
                   LogMatMul.apply((grad_output.log()-out), b.transpose(2, 1).contiguous())).exp()
        grad_b2 =  (b +
                    LogMatMul.apply(a.transpose(2,1).contiguous(), (grad_output.log()-out))).exp()
        return grad_a2, grad_b2


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


logbmm = LogMatMul.apply
maxbmm = MaxMatMul.apply
samplebmm = SampleMatMul.apply
