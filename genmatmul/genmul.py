import torch
import struct_lib

class LogMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, = struct_lib.forward(a, b, 0)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, out = ctx.saved_tensors
        return struct_lib.backward(a, b, grad_output.contiguous(), out, 0)

class MaxMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = struct_lib.forward(a, b, 1)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        return struct_lib.backward(a, b, grad_output.contiguous(), switches.float(), 1)

class SampleMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = struct_lib.forward(a, b, 2)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        return struct_lib.backward(a, b, grad_output.contiguous(), switches.float(), 2)

logmatmul = LogMatMul.apply
maxmatmul = MaxMatMul.apply
samplematmul = SampleMatMul.apply
