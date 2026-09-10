"""Own both download jobs, bounded transport, logs, and final readback."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
HOSTS=['10.0.0.109','10.0.0.183']
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4']
REMOTE='/home/raulwesche/nvidia_glm53_download.py'


def run(host):
    target='raulwesche@'+host
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(ROOT/'download_node.py'),target+':'+REMOTE],check=True,timeout=30)
    command='timeout --signal=TERM --kill-after=20s 10800s python3 -u '+REMOTE
    with (ROOT/(host+'.log')).open('w') as log:
        child=subprocess.Popen(SSH+[target,command],stdout=log,stderr=subprocess.STDOUT)
        (ROOT/(host+'-process.json')).write_text(json.dumps(dict(host=host,pid=child.pid,
                                     command=command,started_epoch=time.time()),indent=2))
        try: code=child.wait(timeout=10840)
        except subprocess.TimeoutExpired:
            child.kill();child.wait(timeout=10);raise
    record=dict(host=host,exit_code=code,log=str(ROOT/(host+'.log')))
    if code==0:
        # Separate fresh SSH readback, not reliance on downloader stdout alone.
        result=subprocess.run(SSH+[target,"python3 -c \"from pathlib import Path; print(Path('/home/raulwesche/nvidia-glm53-download-status.json').read_text())\""],capture_output=True,text=True,check=True,timeout=20)
        status=json.loads(result.stdout)
        assert status['phase']=='complete_verified',status
        assert status['verified_bytes']==status['expected_bytes'],status
        record['verified_status']=status
    (ROOT/(host+'-result.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    return record


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results=list(ex.map(run,HOSTS))
    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
    if any(r['exit_code'] for r in results): raise SystemExit(1)
    print('Both official NVIDIA checkpoint copies downloaded and hash-verified.',flush=True)
