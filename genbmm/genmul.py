import torch

try:
    import _genbmm
except ImportError:
    pass

def trans(s):
    return s.transpose(-2, -1).contiguous()

class LogMatMulBack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, grad_out, part, maxes):
        ctx.save_for_backward(a, b, grad_out, part, maxes)
        grad_a, _ = _genbmm.backward(a, b, grad_out, part, maxes, 0)
        return grad_a

    @staticmethod
    def backward(ctx, grad_output):
        a, b, grad_out, part, maxes = ctx.saved_tensors
        grad_a, grad_b, grad_grad = _genbmm.backbackward(a, b, grad_out.contiguous(),
                                                         part, maxes, grad_output.contiguous(), 0)

        return grad_a, grad_b, grad_grad, None, None


class LogMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, maxes = _genbmm.forward(a, b, 0)
        ctx.save_for_backward(a, b, out, maxes)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, out, maxes = ctx.saved_tensors
        grad_a = LogMatMulBack.apply(a, b, grad_output.contiguous(), out, maxes)
        grad_b = LogMatMulBack.apply(trans(b), trans(a),
                                     trans(grad_output), trans(out), trans(maxes))

        return grad_a, trans(grad_b)

    # grad_a = LogMatMulBack.apply(a, b, grad_output, out, maxes)
        # grad_b = LogMatMulBack.apply(b, a, grad_output.transpose(-2, -1).contiguous(), out.transpose(-2, -1).contiguous(), maxes.transpose(-2, -1).contiguous())
        # # grad_a = torch.einsum("brc,bck,brk->brc", a.exp(),
        # #                       b.exp(),
        # #                       grad_output.contiguous() / (out - maxes).exp())

        # return grad_a, grad_b


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
            a.float(), b.float(), grad_output.contiguous().float(), switches.float(), switches.float(), 1
        )
        return grad_a.to(a.dtype), grad_b.to(b.dtype)


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
            a.float(), b.float(), grad_output.contiguous().float(), switches.float(), switches.float(), 2
        )
        return grad_a.to(a.dtype), grad_b.to(b.dtype)


logbmm = LogMatMul.apply
maxbmm = MaxMatMul.apply
samplebmm = SampleMatMul.apply
