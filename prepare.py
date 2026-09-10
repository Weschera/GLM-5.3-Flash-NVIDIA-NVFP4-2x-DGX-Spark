"""Prepare isolated TP2 runtime only on the two test Sparks."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
HOSTS = ['10.0.0.109', '10.0.0.183']
IMAGE = 'ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6'
REMOTE = '/home/raulwesche/nvidia-glm53-tp2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']

def run(host, cmd, timeout=110):
    return subprocess.run(SSH + ['raulwesche@' + host, cmd], capture_output=True, text=True, timeout=timeout, check=True).stdout

def prepare(host):
    run(host, 'mkdir -p ' + REMOTE + '/patches ' + REMOTE + '/cache ' + REMOTE + '/transformers-overlay')
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', str(ROOT/'docker/sparse_attn_indexer_kpool_sm121.py'), 'raulwesche@'+host+':'+REMOTE+'/patches/sparse_attn_indexer_kpool.py'], check=True, timeout=30)
    try:
        before=run(host, 'docker image inspect '+IMAGE+' --format "{{.Id}}"')
    except subprocess.CalledProcessError:
        with (ROOT/(host+'-image-pull.log')).open('w') as log:
            p=subprocess.Popen(SSH+['raulwesche@'+host,'timeout --kill-after=10s 1800s docker pull '+IMAGE],stdout=log,stderr=subprocess.STDOUT)
            p.wait(timeout=1830)
            if p.returncode: raise RuntimeError('image pull failed; '+str(log.name))
        before=run(host, 'docker image inspect '+IMAGE+' --format "{{.Id}}"')
    install='docker run --rm --network host --entrypoint python3 -v '+REMOTE+'/transformers-overlay:/overlay '+IMAGE+' -m pip install --disable-pip-version-check --timeout 25 --retries 2 --no-deps --upgrade --target /overlay transformers==5.16.1 tokenizers==0.23.1'
    result=run(host,'timeout --kill-after=5s 100s '+install)
    (ROOT/(host+'-transformers-install.log')).write_text(result)
    check="import transformers; from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig; import huggingface_hub,tokenizers; print(transformers.__version__,huggingface_hub.__version__,tokenizers.__version__,Glm5NextConfig.model_type)"
    out=run(host,'timeout 60s docker run --rm --entrypoint python3 -e PYTHONPATH=/overlay -v '+REMOTE+'/transformers-overlay:/overlay:ro '+IMAGE+' -c '+shlex.quote(check))
    record={'host':host,'image':IMAGE,'image_id':before.strip(),'transformers_check':out,'ready_epoch':time.time()}
    (ROOT/(host+'-prepared.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    return record

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results=list(ex.map(prepare,HOSTS))
    (ROOT/'prepared.json').write_text(json.dumps(results,indent=2))
    print('Runtime prepared on both test Sparks.',flush=True)
