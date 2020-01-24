import sys
import os
import time
import torch
import numpy as np

sys.path.append('/tvm/python')
sys.path.append('/tvm/topi/python')
sys.path.append('/tvm/vta/python')
os.environ['TVM_HOME'] = '/tvm'

import tvm
from tvm import autotvm

@autotvm.template
def hmm_runner(dtype):
    #n_num_step = 128
    #n_num_hidden = 1152
    #n_batch_size = 4
    #num_step = tvm.var("num_step")

    nn = 1152
    bb = 32
    tt = 128
    n = tvm.convert(nn)
    m = n
    b = tvm.var("batch")
    t = tvm.var("num_step")
    l = n
    k = tvm.reduce_axis((0, l), name='k')
    k2 = tvm.reduce_axis((0, l), name='k2')
    X = tvm.placeholder((t-1, b, n, m), name="X", dtype=dtype)

    s_state = tvm.placeholder((t, b, n))
    s_init = tvm.compute((1, b, n), lambda a, b, c: 0.0)
    M = tvm.compute(
        (t, b, n),
        lambda t, bb, ii: tvm.max(s_state[t-1, bb, k] + X[t-1, bb, k, ii], axis=k),
        name="M")

    M2 = tvm.compute(
        (t, b, n),
        lambda t, bb, ii: tvm.sum(tvm.exp(s_state[t-1, bb, k2] + X[t-1, bb, k2, ii]
                                          - M[t, bb, ii]), axis=k2),
        name="M2")
    C = tvm.compute(
        (t, b, n),
        #lambda t, bb, ii: M[t, bb, ii] + M2[t, bb,ii],
        lambda t, bb, ii: tvm.log(M2[t, bb, ii]) + M[t, bb, ii],
        name='C')

    s_scan = tvm.scan(s_init, C, s_state, inputs=[X])

    s = tvm.create_schedule(s_scan.op)
    #tvm.lower(s, [X], simple_mode=True )

    cfg = autotvm.get_config()
    cfg.define_knob("y_t", [8])
    cfg.define_knob("x_t", [16])
    cfg.define_knob("sm", [24])
    cfg.add_flop(1)

    num_thread_y = cfg["y_t"].val
    num_thread_x = cfg["x_t"].val * 3
    num_sm = cfg["sm"].val

    PERSIST_KERNEL = False
    DETECT_GLOBAL_BARRIER = False
    detect_global_barrier = DETECT_GLOBAL_BARRIER

    s = tvm.create_schedule(s_scan.op)
    CL = C
    SS = s.cache_read(s_state, "shared", [M])
    SL = s.cache_read(SS, "local", [M])
    SS2 = s.cache_read(s_state, "shared", [M2])
    SL2 = s.cache_read(SS2, "local", [M2])

    WhhL = s.cache_read(X, "local", [M])
    WhhL2 = s.cache_read(X, "local", [M2])

    #First
    ko, ki = s[M].split(s[M].op.reduce_axis[0], nparts=num_thread_y)
    MLF = s.rfactor(M, ko)
    #Second
    ko2, ki2 = s[M2].split(s[M2].op.reduce_axis[0], nparts=num_thread_y)
    MLF2 = s.rfactor(M2, ko2)

    block_x = tvm.thread_axis((0, num_sm), "blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
    if PERSIST_KERNEL:
        s[s_scan.op].env_threads([block_x, thread_y, thread_x])

    bx, xi = s[s_init].split(s_init.op.axis[2], nparts=num_sm)
    tx, xi = s[s_init].split(xi, nparts=num_thread_x)
    s[s_init].bind(bx, block_x)
    s[s_init].bind(tx, thread_x)

    bx, xi = s[CL].split(s[CL].op.axis[2], nparts=num_sm)
    tx, xi = s[CL].split(xi, nparts=num_thread_x)
    s[CL].bind(bx, block_x)
    s[CL].bind(tx, thread_x)
    #s[CL].bind(s[M2].op.reduce_axis[0], thread_y)

    s[M].compute_at(s[CL], tx)
    s[M].bind(s[M].op.reduce_axis[0], thread_y)
    #s[M].bind(s[M].op.axis[2], thread_x)
    s[MLF].compute_at(s[M], s[M].op.reduce_axis[0])
    #s[MLF].compute_at(s[CL], s[CL].op.reduce_axis[0])

    #s[M].set_store_predicate(thread_y.equal(0))

    # Repeat
    s[M2].compute_at(s[CL], tx)
    #s[M2].bind(s[M2].op.axis[2], thread_x)
    s[M2].bind(s[M2].op.reduce_axis[0], thread_y)
    s[MLF2].compute_at(s[M2], s[M2].op.reduce_axis[0])
    #s[M2].set_store_predicate(thread_y.equal(0))

    #if PERSIST_KERNEL:
    #    s[WhhL].compute_at(s[s_scan], thread_x)
    #    s[WhhL].unroll(WhhL.op.axis[0])
    #else:
    s[WhhL].compute_at(s[MLF], MLF.op.axis[3])
    s[WhhL2].compute_at(s[MLF2], MLF2.op.axis[3])

    kr, ki = s[MLF].split(MLF.op.reduce_axis[0], nparts=1)
    ko, ki = s[MLF].split(ki, factor=4)
    s[SS].compute_at(s[MLF], kr)
    s[SL].compute_at(s[MLF], ko)

    xo, xi = s[SS].split(SS.op.axis[2], factor=num_thread_x * num_thread_y * 3)
    ty, xi = s[SS].split(xi, nparts=num_thread_y)
    tx, xi = s[SS].split(xi, nparts=num_thread_x)
    #s[SS].bind(ty, thread_y)
    s[SS].bind(tx, thread_x)

    # Second
    kr, ki = s[MLF2].split(MLF2.op.reduce_axis[0], nparts=1)
    ko, ki = s[MLF2].split(ki, factor=4)
    s[SS2].compute_at(s[MLF2], kr)
    s[SL2].compute_at(s[MLF2], ko)

    xo, xi = s[SS2].split(SS2.op.axis[2], factor=num_thread_x * num_thread_y * 3)
    ty, xi = s[SS2].split(xi, nparts=num_thread_y)
    tx, xi = s[SS2].split(xi, nparts=num_thread_x)
    #s[SS2].bind(ty, thread_y)
    s[SS2].bind(tx, thread_x)
    return s, [X, s_scan]

#task = autotvm.task.create(hmm, args=('float32',), target='cuda', target_host="llvm")
with autotvm.apply_history_best('best.log'):
    with tvm.target.create("cuda"):
        s_mult, arg_bufs = hmm_runner('float32')
        mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")

from tvm.contrib.dlpack import to_pytorch_func
hmm_pytorch = to_pytorch_func(mod)

def fb(x):
    time, batch, size, _ = x.shape
    out = torch.zeros(time+1, batch, size).cuda()
    hmm_pytorch(x, out)

    out2 = torch.zeros(time+1, batch, size).cuda()
    hmm_pytorch(x.flip(0).transpose(-2, -1).contiguous(), out2)

    marginals = torch.exp(out[:-1].view(time, batch, 1, size) +
                          out2[:-1].flip(0).view(time, batch, size, 1) +
                          x.view(time, batch, size, size).transpose(-2, -1)
                          - out[-1].logsumexp(-1).view(1, -1, 1, 1))
    return out, marginals

if __name__ == "__main__":
    from tvm import autotvm
    import logging
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    task = autotvm.task.create(hmm_runner, args=('float32',),
                               target='cuda', target_host="llvm")

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=5),
        runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=100,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('hmm.log')])

    autotvm.record.pick_best("hmm.log", "best.log")
