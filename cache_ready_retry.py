"""Precompile without target weights, verify cache transfer, then retry TP2."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from prepare import ROOT, REMOTE, IMAGE, SSH

SOURCE = '10.0.0.109'
DEST = '10.0.0.183'
NAME = 'nvidia-glm53-kernel-probe'
OUT = ROOT/'evidence/text128-precompiled'
OUT.mkdir(parents=True,exist_ok=True)
STATE = ROOT/'cache-retry-status.json'
REL = '.cache/flashinfer/0.6.18.dev20260819/121a'


def status(phase,**extra):
    data={'phase':phase,'pid':os.getpid(),'epoch':time.time(),**extra}
    STATE.write_text(json.dumps(data,indent=2)); print(json.dumps(data),flush=True)


def remote(host,command,timeout=25):
    return subprocess.run(SSH+['raulwesche@'+host,command],capture_output=True,text=True,check=True,timeout=timeout).stdout


def py(host,code,timeout=25):
    return remote(host,'timeout '+str(timeout-3)+'s python3 -c '+shlex.quote(code),timeout)


def copy(local,host,dest):
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(local),'raulwesche@'+host+':'+dest],check=True,timeout=110)


def result(host):
    code="from pathlib import Path; p=Path('"+REMOTE+"/cache/kernel-probe-result.json'); print(p.read_text() if p.exists() else '{\"rows\":[]}')"
    return json.loads(py(host,code))


def stop_probe(host):
    code="""import subprocess,json
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=8)
if r.returncode==0:
 c=json.loads(r.stdout)[0]
 assert c['Config']['Labels'].get('wesche.task')==NAME
 subprocess.run(['docker','rm','-f',c['Id']],check=True,timeout=15)
 r=subprocess.run(['docker','inspect',c['Id']],capture_output=True,timeout=8)
 assert r.returncode!=0
print('scoped_probe_cleanup_verified')
""".replace('NAME',repr(NAME))
    print(host,py(host,code,40),flush=True)


def main():
    status('waiting_for_single_job_kernel_build')
    deadline=time.monotonic()+950
    while time.monotonic()<deadline:
        receipt=result(SOURCE)
        stage=receipt['rows'][-1]['stage'] if receipt['rows'] else 'initializing'
        if stage=='complete':break
        if stage=='failed':raise RuntimeError(receipt)
        live=remote(SOURCE,'timeout 8s docker ps --filter name='+NAME+' --format "{{.Names}}"',15)
        assert NAME in live.splitlines(), 'Kernel diagnostic exited without a passing receipt'
        time.sleep(10)
    else:raise TimeoutError('Kernel build did not finish within deadline')
    (OUT/'source-kernel-probe.json').write_text(json.dumps(receipt,indent=2))
    # Require the producer to exit before hashing/copying its cache.
    for _ in range(10):
        live=remote(SOURCE,'timeout 8s docker ps --filter name='+NAME+' --format "{{.Names}}"',15)
        if NAME not in live.splitlines():break
        time.sleep(1)
    else:raise RuntimeError('Passing diagnostic did not exit')
    assert not remote(DEST,'timeout 8s docker ps --format "{{.Names}}"',15).strip()
    assert not remote(DEST,'timeout 8s nvidia-smi --query-compute-apps=pid --format=csv,noheader',15).strip()
    status('hashing_and_transferring_compiled_cache')
    inventory="""from pathlib import Path
import hashlib,json
root=Path(BASE); rows=[]
for p in sorted((root/RELATIVE).rglob('*')):
 if p.is_file():rows.append({'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
print(json.dumps(rows))
""".replace('BASE',repr(REMOTE+'/cache')).replace('RELATIVE',repr(REL))
    manifest=json.loads(py(SOURCE,inventory,90));assert manifest and any(x['path'].endswith('.so') for x in manifest)
    manifest_path=OUT/'cache-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=OUT/'flashinfer-cache.tar'
    with archive.open('wb') as stream:
        subprocess.run(SSH+['raulwesche@'+SOURCE,'timeout 90s tar -C '+REMOTE+'/cache -cf - '+REL],stdout=stream,check=True,timeout=100)
    copy(archive,DEST,REMOTE+'/flashinfer-cache.tar')
    remote(DEST,'timeout 90s tar -C '+REMOTE+'/cache -xf '+REMOTE+'/flashinfer-cache.tar',100)
    observed=json.loads(py(DEST,inventory,90))
    index={x['path']:x for x in observed}
    assert all(index.get(x['path'])==x for x in manifest),'Transferred cache differs'
    (OUT/'cache-transfer-verified.json').write_text(json.dumps({'file_count':len(manifest),'bytes':sum(x['size'] for x in manifest),'all_source_hashes_match':True},indent=2))
    status('validating_cached_kernel_on_second_node')
    copy(ROOT/'kernel_probe.py',DEST,REMOTE+'/kernel_probe.py')
    digest=py(DEST,"from pathlib import Path; import hashlib; print(hashlib.sha256(Path('"+REMOTE+"/kernel_probe.py').read_bytes()).hexdigest())").strip()
    assert digest==hashlib.sha256((ROOT/'kernel_probe.py').read_bytes()).hexdigest()
    cmd=['docker','run','--rm','--name',NAME,'--label','wesche.task='+NAME,'--gpus','all','--network','none','--memory','32g','--memory-swap','32g','--cpus','4','--pids-limit','256']
    for key,value in {'MAX_JOBS':'1','FLASHINFER_NVCC_THREADS':'1','FLASHINFER_WORKSPACE_BASE':'/cache','TRITON_CACHE_DIR':'/cache/triton','TORCH_CUDA_ARCH_LIST':'12.1a','FLASHINFER_CUDA_ARCH_LIST':'12.1a','FLASHINFER_DISABLE_VERSION_CHECK':'1','PYTHONPATH':'/transformers-overlay'}.items():cmd+=['-e',key+'='+value]
    for src,dst,mode in [(REMOTE+'/cache','/cache','rw'),(REMOTE+'/transformers-overlay','/transformers-overlay','ro'),(REMOTE+'/kernel_probe.py','/probe.py','ro')]:cmd+=['-v',src+':'+dst+':'+mode]
    cmd+=['--entrypoint','python3',IMAGE,'-u','/probe.py']
    log=remote(DEST,'timeout --kill-after=5s 100s '+shlex.join(cmd),110)
    (OUT/'destination-kernel-probe.log').write_text(log)
    receipt=result(DEST);assert receipt['rows'][-1]['stage']=='complete',receipt
    (OUT/'destination-kernel-probe.json').write_text(json.dumps(receipt,indent=2))
    status('kernel_checks_passed_launching_guarded_tp2')
    os.execv(os.sys.executable,[os.sys.executable,'-u',str(ROOT/'retry_text128.py')])


if __name__=='__main__':
    lock=(ROOT/'cache-retry.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Cache preparation deadline')))
    signal.alarm(1500)
    try:main()
    except BaseException as exc:
        status('failed',error=str(exc))
        for host in [SOURCE,DEST]:
            try:stop_probe(host)
            except Exception as cleanup:print('CLEANUP_ERROR',host,str(cleanup),flush=True)
        raise
