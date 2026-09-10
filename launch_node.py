"""Scoped, reproducible TP2 launcher. Never stops unrelated containers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

BASE = Path('/home/raulwesche/nvidia-glm53-tp2')
MODEL = Path('/home/raulwesche/models/nvidia-GLM-5.3-Flash-NVFP4')
DRAFT_REPO = Path('/home/raulwesche/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2')
DRAFT_REV = 'bf582e4eacc1810f76656d1811693ff6c6737d2a'
REV = '423acf37583782c51c142d145aef733d72943d93'
IMAGE = 'ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6'
NAME = 'nvidia-glm53-tp2'
MODEL_ID = 'nvidia/GLM-5.3-Flash-NVFP4'
HOSTS = {'spark-78f1': (0, '10.10.10.1'), 'spark-366f': (1, '10.10.10.5')}

def command(rank, ip, speculative):
    env = {
        'VLLM_HOST_IP': ip, 'HF_HOME':'/cache/huggingface',
        'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
        'VLLM_ENGINE_READY_TIMEOUT_S':'1800',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
        'PYTHONPATH':'/transformers-overlay',
        'TORCH_CUDA_ARCH_LIST':'12.1a', 'FLASHINFER_CUDA_ARCH_LIST':'12.1a',
        'FLASHINFER_DISABLE_VERSION_CHECK':'1',
        'MAX_JOBS':'1','FLASHINFER_NVCC_THREADS':'1',
        'FLASHINFER_WORKSPACE_BASE':'/cache','TRITON_CACHE_DIR':'/cache/triton',
        'NCCL_NET':'IB','NCCL_IB_DISABLE':'0','NCCL_IB_HCA':'rocep1s0f0',
        'NCCL_IB_GID_INDEX':'3','NCCL_IB_ROCE_VERSION_NUM':'2',
        'NCCL_IB_ADDR_FAMILY':'AF_INET','NCCL_IB_ADDR_RANGE':'10.10.10.0/24',
        'NCCL_SOCKET_IFNAME':'enp1s0f0np0','GLOO_SOCKET_IFNAME':'enp1s0f0np0',
        'TP_SOCKET_IFNAME':'enp1s0f0np0','MN_IF_NAME':'enp1s0f0np0',
        'NCCL_NVLS_ENABLE':'0','NCCL_CROSS_NIC':'0','NCCL_IB_MERGE_NICS':'0',
        'NCCL_CUMEM_ENABLE':'0','NCCL_IGNORE_CPU_AFFINITY':'1','NCCL_DEBUG':'INFO',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
    }
    cmd=['docker','run','--gpus','all','-d','--name',NAME,'--restart','no',
         '--label','wesche.task=nvidia-glm53-tp2','--network','host','--ipc','host',
         '--shm-size','32g','--ulimit','memlock=-1:-1','--cap-add','IPC_LOCK',
         '--device','/dev/infiniband:/dev/infiniband']
    mounts=[(MODEL,'/model'),(BASE/'cache','/cache'),
            (BASE/'patches/model_runner_profile_trace.py','/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py'),
            (BASE/'transformers-overlay','/transformers-overlay'),
            (BASE/'chat_template.jinja','/recipe/chat_template.jinja'),
            (BASE/'patches/sparse_attn_indexer_kpool.py','/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py'),
            (DRAFT_REPO,'/draft-repo')]
    for src,dst in mounts: cmd+=['-v',f'{src}:{dst}'+('' if dst=='/cache' else ':ro')]
    for k,v in env.items(): cmd+=['-e',k+'='+v]
    cmd += [IMAGE,'/model','--served-model-name',MODEL_ID,'--host','0.0.0.0','--port','8910',
            '--trust-remote-code','--generation-config','vllm','--tensor-parallel-size','2','--gpu-memory-utilization','0.88',
            '--max-model-len','131072','--max-num-seqs','1','--block-size','2304',
            '--language-model-only','--limit-mm-per-prompt','{"image":0,"video":0}',
            '--moe-backend','marlin','--kv-cache-dtype','fp8_e4m3','--kv-cache-memory','6442450944',
            '--enforce-eager','--max-num-batched-tokens','128','--no-enable-prefix-caching',
            '--tool-call-parser','glm47','--enable-auto-tool-choice','--reasoning-parser','glm45',
            '--default-chat-template-kwargs','{"enable_thinking":false}',
            '--chat-template','/recipe/chat_template.jinja',
            '--distributed-executor-backend','mp','--nnodes','2','--node-rank',str(rank),
            '--master-addr','10.10.10.1','--master-port','29531']
    if rank: cmd+=['--headless']
    if speculative: cmd+=['--speculative-config',json.dumps({'method':'dflash','model':'/draft-repo/snapshots/'+DRAFT_REV,'num_speculative_tokens':speculative})]
    return cmd

def run(cmd,check=True,timeout=60):
    return subprocess.run(cmd,capture_output=True,text=True,check=check,timeout=timeout)

def check_bindable(port):
    # Match server reuse semantics: TIME_WAIT is not a live port owner.
    # SO_REUSEPORT is deliberately not enabled; live listeners still conflict.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))


def main():
    p=argparse.ArgumentParser();p.add_argument('--speculative',type=int,choices=[0,3,7],default=0);p.add_argument('--dry-run',action='store_true');p.add_argument('--stop',action='store_true');a=p.parse_args()
    hostname=socket.gethostname();assert hostname in HOSTS,hostname
    rank,ip=HOSTS[hostname]
    if a.stop:
        x=run(['docker','inspect',NAME],check=False)
        if x.returncode:print('Named test container is absent.');return
        c=json.loads(x.stdout)[0]
        assert c['Config']['Labels'].get('wesche.task')=='nvidia-glm53-tp2'
        print(run(['docker','stop','-t','15',NAME],timeout=30).stdout)
        print(run(['docker','rm',NAME],timeout=20).stdout);return
    cmd=command(rank,ip,a.speculative)
    if a.dry_run: print(shlex.join(cmd));return
    s=json.loads(Path('/home/raulwesche/nvidia-glm53-download-status.json').read_text())
    assert s['phase']=='complete_verified' and s['revision']==REV,s
    assert s['verified_bytes']==s['expected_bytes']
    assert int(Path('/proc/sys/vm/swappiness').read_text())==0
    others=run(['docker','ps','--format','{{.Names}}']).stdout.splitlines()
    assert not others, f'Another container is active: {others}'
    gpu_jobs=run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip()
    assert not gpu_jobs, f'Another GPU workload is active: {gpu_jobs}'
    run(['sync'],timeout=30)
    run(['sudo','-n','sysctl','-w','vm.drop_caches=3'],timeout=20)
    check_bindable(8910 if rank==0 else 29532)
    assert (DRAFT_REPO/'snapshots'/DRAFT_REV/'config.json').is_file()
    metadata={'hostname':hostname,'rank':rank,'revision':REV,'speculative_tokens':a.speculative,
              'image':IMAGE,'command':cmd,'started_epoch':time.time(),
              'template_sha256':hashlib.sha256((BASE/'chat_template.jinja').read_bytes()).hexdigest(),
              'kernel_patch_sha256':hashlib.sha256((BASE/'patches/sparse_attn_indexer_kpool.py').read_bytes()).hexdigest()}
    BASE.mkdir(parents=True,exist_ok=True)
    tag=str(int(time.time()))
    (BASE/('launch-'+tag+'.json')).write_text(json.dumps(metadata,indent=2))
    out=run(cmd,timeout=100);print(out.stdout,flush=True)
    state=json.loads(run(['docker','inspect',NAME]).stdout)[0]
    metadata['container_id']=state['Id'];metadata['state']=state['State']
    (BASE/'active-launch.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
