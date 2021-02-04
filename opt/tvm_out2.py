import sys
import os
import time
import torch
import numpy as np
import logging

sys.path.append('/tvm/python')
sys.path.append('/tvm/topi/python')
sys.path.append('/tvm/vta/python')
os.environ['TVM_HOME'] = '/tvm'

import tvm
from tvm import autotvm
from tvm import te
import math

@autotvm.template("logsummulouter")
def logsummul_outer(n, l, m, dtype):
    nn = n
    n = nn
    bb = 1
    b = bb

    m_2 = int(math.sqrt(m))

    # matrices
    A = te.placeholder((bb, nn, l), name='A', dtype=dtype)
    B1 = te.placeholder((bb, m_2, l), name='B1', dtype=dtype)
    B2 = te.placeholder((bb, m_2, l), name='B2', dtype=dtype)

    def B_fun(bb, ii, jj):
        m1, m2 = ii // m_2 , ii % m_2
        return B1[bb, m1, jj] * B2[bb, m2, jj]
        
    B = te.compute(
        (bb, m, l),
        B_fun, name='B'
        )

    k = te.reduce_axis((0, l), name='k')
    k2 = te.reduce_axis((0, l), name='k2')

    M = te.compute(
        (b, m, n),
        lambda bb, ii, jj: te.max(A[bb, jj, k] + B[bb, ii, k], axis=k),
        name='M'
    )
    M2 = te.compute(
        (b, m, n),
        lambda bb, ii, jj: te.sum(te.exp(A[bb, jj, k2] + B[bb, ii, k2]- M[bb, ii, jj]), axis=k2),
        #lambda bb, ii, jj: te.sum(te.exp(A[bb, jj, k2] + B[bb, ii, k2]- M[bb, ii, jj]), axis=k2),
        name='M2')


    C = te.compute(
        (b, m, n),
        lambda bb, ii, jj: te.log(M2[bb, ii, jj]) + M[bb, ii, jj],
        name='C')

    s = te.create_schedule(C.op)

    # BB1S = s.cache_read(B1, "shared", [B])
    # BB1L = s.cache_read(BB1S, "local", [B])
    # BB2S = s.cache_read(B2, "shared", [B])
    # BB2L = s.cache_read(BB2S, "local", [B])
    
    AA = s.cache_read(A, "shared", [M])
    AL = s.cache_read(AA, "local", [M])
    BB = s.cache_read(B, "shared", [M])
    BL = s.cache_read(BB, "local", [M])

    AA2 = s.cache_read(A, "shared", [M2])
    AL2 = s.cache_read(AA2, "local", [M2])
    BB2 = s.cache_read(B, "shared", [M2])
    BL2 = s.cache_read(BB2, "local", [M2])

    cfg = autotvm.get_config()
    cfg.define_knob("y_bn", [32, 64, 128])
    cfg.define_knob("x_bn", [32, 64, 128])
    cfg.define_knob("y_t", [8, 32, 64])
    cfg.define_knob("x_t", [2, 4, 8, 32])
    cfg.define_knob("k_split", [1, 2, 8, 16])
    unroll = False

    b, y, x = s[C].op.axis
    y_bn = cfg["y_bn"].val
    x_bn = cfg["x_bn"].val
    by, y = s[C].split(y, y_bn)
    bx, x = s[C].split(x, x_bn)

    y_nthreads = cfg["y_t"].val
    x_nthreads = cfg["x_t"].val
    ty, yi = s[C].split(y, nparts=y_nthreads)
    tx, xi = s[C].split(x, nparts=x_nthreads)
    thread_x = te.thread_axis((0, x_nthreads), "threadIdx.x")
    thread_y = te.thread_axis((0, y_nthreads), "threadIdx.y")

    s[C].reorder(b, by, bx, ty, tx, yi, xi)
    s[C].bind(b, te.thread_axis("blockIdx.z"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    if unroll:
        s[C].pragma(yi, "auto_unroll_max_step", 16)

    def cache_split(shared):
        s[shared].compute_at(s[C], tx)
        _, yi, xi = s[shared].op.axis
        k, = s[shared].op.reduce_axis
        ko, ki = s[shared].split(k, cfg["k_split"].val)
        s[shared].reorder(ko, ki, yi, xi)
        if unroll:
            s[shared].pragma(ki, "auto_unroll_max_step", 16)
        return ko, ki
    ko, ki = cache_split(M)
    ko2, ki2 = cache_split(M2)

    s[B].compute_at(s[C], tx)
    # s[BB1S].compute_at(s[C], tx)
    # s[BB1L].compute_at(s[C], tx)
    # s[BB2S].compute_at(s[C], tx)
    # s[BB2L].compute_at(s[C], tx)

    def cache_read(shared, AA, AL, BB, BL, ko, ki):
        s[AA].compute_at(s[shared], ko)
        s[AL].compute_at(s[shared], ki)
        s[BB].compute_at(s[shared], ko)
        s[BL].compute_at(s[shared], ki)

        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=y_nthreads)
        tx, ki = s[AA].split(k, nparts=x_nthreads)
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        if unroll:
            s[AA].pragma(yi, "auto_unroll_max_step", 16)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=y_nthreads)
        tx, ki = s[BB].split(k, nparts=x_nthreads)
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        if unroll:
            s[BB].pragma(xi, "auto_unroll_max_step", 16)

        
    cache_read(M, AA, AL, BB, BL, ko, ki)
    cache_read(M2, AA2, AL2, BB2, BL2, ko2, ki2)

    return s, [A, B1, B2, C, M]


S1, S2, S3 = 16, 16, 16
print(tvm.lower(*logsummul_outer(S1, S2, S3, 'float32'), simple_mode=True))
from tvm.contrib.dlpack import to_pytorch_func
task = autotvm.task.create("logsummulouter", args=(S1, S2, S3, 'float32',), target='cuda', target_host="llvm")
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.Target("cuda"):
        s_mult, arg_bufs = logsummul_outer(S1, S2, S3, 'float32')
        mod = tvm.build(s_mult, arg_bufs)
logsum_pytorch = to_pytorch_func(mod)



@autotvm.template("logsummulback")
def logsummulback(n, l, m, dtype):
    nn = n
    n = nn
    bb = 1
    b = bb

    m_2 = int(math.sqrt(m))

    # matrices
    A = te.placeholder((bb, nn, l), name='A', dtype=dtype)
    B1 = te.placeholder((bb, m_2, l), name='B1', dtype=dtype)
    B2 = te.placeholder((bb, m_2, l), name='B2', dtype=dtype)
    C = te.placeholder((b, m, n), name='C')
    grad_C = te.placeholder((bb, m, n), name='grad_C', dtype=dtype)
    M = te.placeholder((b, m, n), name='M')

    def B_fun(bb, ii, jj):
        m1, m2 = ii // m_2 , ii % m_2
        return B1[bb, m1, jj] * B2[bb, m2, jj]

    B = te.compute(
        (bb, m, l),
        B_fun, name='B'
        )
    
    k = te.reduce_axis((0, n), name='k')
    def grad_fn(bb, m1, m2, ll):
        mm = m_2 * m1 + m2
        return te.sum((te.exp(A[bb, k, ll] + B[bb, mm, ll] - M[bb, mm, k]) /
                te.exp(C[bb, mm, k] - M[bb, mm, k])) * grad_C[bb, mm, k], axis=k)
        
    grad_B = te.compute((bb, m_2, m_2, l),
                        grad_fn,
                        name='grad_B')

    m1 = te.reduce_axis((0, m_2), name='m1')
    m2 = te.reduce_axis((0, m_2), name='m2')
    def grad_output1(bb, m1, ll):
        return te.sum(grad_B[bb, m1, m2, ll] * B2[bb, m2, ll], axis=m2)
    def grad_output2(bb, m2, ll):
        return te.sum(grad_B[bb, m1, m2, ll] * B1[bb, m1, ll], axis=m1)
    output = grad_B
    output1 = te.compute((bb, m_2, l),
                         grad_output1,
                         name="output1"
    )
    output2 = te.compute((bb, m_2, l),
                         grad_output2,
                         name="output2"
    )

    s = te.create_schedule([output1.op, output2.op])

    AA = s.cache_read(A, "shared", [output])
    AL = s.cache_read(AA, "local", [output])
    BB = s.cache_read(B, "shared", [output])
    BL = s.cache_read(BB, "local", [output])
    MM = s.cache_read(M, "shared", [output])
    ML = s.cache_read(MM, "local", [output])
    CC = s.cache_read(C, "shared", [output])
    CL = s.cache_read(CC, "local", [output])
    grad_CC = s.cache_read(grad_C, "shared", [output])
    grad_CL = s.cache_read(grad_CC, "local", [output])


    cfg = autotvm.get_config()
    cfg.define_knob("y_bn", [32, 64, 128])
    cfg.define_knob("x_bn", [32, 64, 128])
    cfg.define_knob("y_t", [8, 32, 64])
    cfg.define_knob("x_t", [2, 4, 8, 32])
    cfg.define_knob("k_split", [1, 2, 8, 16])
    unroll = False


    b_o1, y_o1, x_o1 = s[output1].op.axis
    b_o2, y_o2, x_o2 = s[output2].op.axis
    b, y1, y2, x = s[output].op.axis

    y_bn = cfg["y_bn"].val
    x_bn = cfg["x_bn"].val
    by1, y1 = s[output].split(y1, y_bn)
    
    bx, x = s[output].split(x, x_bn)


    y_nthreads = cfg["y_t"].val
    x_nthreads = cfg["x_t"].val

    ty1, yi1 = s[output].split(y1, nparts=y_nthreads)
    ty1_o1, yi1_o1 = s[output1].split(y_o1, nparts=y_nthreads)
    ty1_o2, yi1_o2 = s[output2].split(y_o2, nparts=y_nthreads)

    tx, xi = s[output].split(x, nparts=x_nthreads)
    tx_o1, xi_o1 = s[output1].split(x_o1, nparts=x_nthreads)
    tx_o2, xi_o2 = s[output2].split(x_o2, nparts=x_nthreads)

    
    thread_x = te.thread_axis((0, x_nthreads), "threadIdx.x")
    thread_y = te.thread_axis((0, y_nthreads), "threadIdx.y")

    s[output].reorder(b, by1, bx, ty1, tx, yi1, y2, xi)
    s[output1].reorder(b_o1, ty1_o1, tx_o1, yi1_o1, xi_o1)
    s[output2].reorder(b_o2, ty1_o2, tx_o2, yi1_o2, xi_o2)

    
    s[output].bind(b, te.thread_axis("blockIdx.z"))
    s[output].bind(by1, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(ty1, thread_y)
    s[output].bind(tx, thread_x)


    s[output1].bind(b_o1, te.thread_axis("blockIdx.z"))
    # s[output1].bind(bx_o1, te.thread_axis("blockIdx.x"))
    s[output1].bind(ty1_o1, thread_y)
    s[output1].bind(tx_o1, thread_x)
    
    s[output2].bind(b_o2, te.thread_axis("blockIdx.z"))
    # s[output1].bind(bx_o1, te.thread_axis("blockIdx.x"))
    s[output2].bind(ty1_o2, thread_y)
    s[output2].bind(tx_o2, thread_x)


    if unroll:
        s[output].pragma(yi1, "auto_unroll_max_step", 16)

    def cache_split(shared):
        k, = s[shared].op.reduce_axis
        ko, ki = s[shared].split(k, cfg["k_split"].val)
        s[shared].reorder(ko, ki, yi1, xi)
        if unroll:
            s[shared].pragma(ki, "auto_unroll_max_step", 16)

        return ko, ki
    ko, ki = cache_split(output)
    s[B].compute_at(s[output], tx)
    
    def cache_read(shared, AA, AL, ko, ki):
        s[AA].compute_at(s[shared], ko)
        s[AL].compute_at(s[shared], ki)

        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=y_nthreads)
        tx, ki = s[AA].split(k, nparts=x_nthreads)
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        if unroll:
            s[AA].pragma(yi, "auto_unroll_max_step", 16)

    cache_read(output, AA, AL, ko, ki)
    cache_read(output, BB, BL, ko, ki)
    cache_read(output, CC, CL, ko, ki)
    cache_read(output, MM, ML, ko, ki)
    cache_read(output, grad_CC, grad_CL, ko, ki)
    
    return s, [A, B1, B2, C, M, grad_C, output1, output2]



S1, S2, S3 = 16, 16, 16
# print(tvm.lower(*logsummulback(S1, S2, S3, 'float32'), simple_mode=True))
from tvm.contrib.dlpack import to_pytorch_func
task = autotvm.task.create("logsummulback", args=(S1, S2, S3, 'float32',), target='cuda', target_host="llvm")
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.Target("cuda"):
        s_mult, arg_bufs = logsummulback(S1, S2, S3, 'float32')
        mod = tvm.build(s_mult, arg_bufs)
logsumback_pytorch = to_pytorch_func(mod)


X = torch.rand(1, S1, S2).cuda()
Y1 = torch.rand(1, int(math.sqrt(S3)), S2).cuda()
Y2 = torch.rand(1, int(math.sqrt(S3)), S2).cuda()
Y1.requires_grad_(True)
Y2.requires_grad_(True)
out = torch.rand(1, S3, S1).cuda()
M = torch.rand(1, S3, S1).cuda()

logsum_pytorch(X, Y1, Y2, out, M)
b = ((Y1[:, :, None, :] * Y2[:, None, :, :]).view(1, 1, S3, S2) + X.view(1, S1, 1, S2)).logsumexp(-1)



grad_C = torch.rand(1, S3, S1).cuda()
grad_B1 = torch.rand(1, int(math.sqrt(S3)), S2).cuda()
grad_B2 = torch.rand(1, int(math.sqrt(S3)), S2).cuda()
logsumback_pytorch(X, Y1, Y2, out, M, grad_C, grad_B1, grad_B2)

print((torch.isclose(out, b.transpose(1, 2))).all())


(grad_B1_, grad_B2_) = torch.autograd.grad(b.transpose(1, 2), [Y1, Y2], [grad_C])

print(grad_B1_)
print(grad_B1)
print(grad_B2_)
print(grad_B2)

