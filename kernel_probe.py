"""Synthetic zero-input kernel diagnostic, NOT model inference or a speed score."""
import json
from pathlib import Path
import resource
import signal
import time
import torch
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

START = time.monotonic()
OUT = Path('/cache/kernel-probe-result.json')
ROWS = []


def record(stage, **extra):
    info = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    row = {'stage': stage, 'elapsed_s': time.monotonic()-START,
           'gpu_allocated': torch.cuda.memory_allocated(),
           'gpu_reserved': torch.cuda.memory_reserved(),
           'process_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'host_available_bytes': info['MemAvailable'], **extra}
    ROWS.append(row)
    OUT.write_text(json.dumps({'kind': 'synthetic_kernel_diagnostic_not_inference', 'rows': ROWS}, indent=2))
    print(json.dumps(row), flush=True)


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Kernel probe 15-minute deadline')))
signal.alarm(900)
try:
    record('imports_complete')
    scale = torch.ones(1, device='cuda', dtype=torch.float32)
    for m, n, k in [(128, 12288, 4096), (128, 4096, 6144), (1, 12288, 4096)]:
        record('begin_shape', shape=[m,n,k])
        x = torch.zeros((m,k), device='cuda', dtype=torch.bfloat16)
        w = torch.zeros((n,k), device='cuda', dtype=torch.bfloat16)
        a, sa = scaled_fp4_quant(x, scale, is_sf_swizzled_layout=True, backend='flashinfer-cutlass')
        b, sb = scaled_fp4_quant(w, scale, is_sf_swizzled_layout=True, backend='flashinfer-cutlass')
        torch.cuda.synchronize()
        record('quantization_complete', shape=[m,n,k])
        y = flashinfer_scaled_fp4_mm(a,b,sa,sb,scale,torch.bfloat16,backend='cutlass')
        torch.cuda.synchronize()
        assert tuple(y.shape)==(m,n)
        assert torch.isfinite(y).all().item() and torch.count_nonzero(y).item()==0
        record('gemm_zero_fixture_passed', shape=[m,n,k])
        del x,w,a,b,sa,sb,y
    record('complete')
except BaseException as exc:
    record('failed', error=str(exc))
    raise
