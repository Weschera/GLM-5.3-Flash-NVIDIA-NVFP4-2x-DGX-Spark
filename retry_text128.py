"""Guarded 128K text-only retry, followed by qualified DFlash2 trials."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback

import workflow as w

ROOT = Path(__file__).resolve().parent
w.OUT = ROOT / 'evidence' / 'text128-dflash-resume'
w.OUT.mkdir(parents=True, exist_ok=True)
w.STATE = ROOT / 'text128-status.json'
ARMED = []


def guard_state(host):
    return json.loads(w.py(host, "from pathlib import Path; p=Path('" + w.REMOTE + "/memory-guard-state.json'); print(p.read_text() if p.exists() else '{}')", 20))


def prepare_guard(host):
    state = w.inspect(host)
    assert not state or not state['State']['Running'], 'Existing test workload is still active'
    if state:
        assert state['Config']['Labels'].get('wesche.task') == w.NAME
        result = w.remote(host, 'timeout 30s python3 '+w.REMOTE+'/launch_node.py --stop', timeout=40)
        assert w.inspect(host) is None
    # Restore the previously authorized setting lost on reboot.
    w.remote(host, 'timeout 10s sudo -n sysctl -w vm.swappiness=0', timeout=15)
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(ROOT/'memory_guard.py'),
                    'raulwesche@'+host+':'+w.REMOTE+'/memory_guard.py'],check=True,timeout=25)
    code = """import hashlib,json,os,pathlib,subprocess,time
p=pathlib.Path(REMOTE)
s=p/'memory-guard-state.json'
if s.exists():
 old=json.loads(s.read_text())
 if old.get('phase')=='armed':
  try:os.kill(old['pid'],0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Another live guard already exists')
(p/'memory-guard.done').unlink(missing_ok=True)
f=(p/'memory-guard.log').open('a')
c=subprocess.Popen(['python3','-u',str(p/'memory_guard.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
print(json.dumps({'pid':c.pid,'sha256':hashlib.sha256((p/'memory_guard.py').read_bytes()).hexdigest()}))
""".replace('REMOTE', repr(w.REMOTE))
    launched=json.loads(w.py(host, code, 20))
    assert launched['sha256']==hashlib.sha256((ROOT/'memory_guard.py').read_bytes()).hexdigest()
    ARMED.append(host)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        state=guard_state(host)
        if state.get('pid')==launched['pid'] and state['phase']=='armed':
            return state
        time.sleep(0.3)
    raise RuntimeError('Guard did not arm on '+host)


original_inspect=w.inspect

def guarded_inspect(host):
    state=guard_state(host)
    assert state['phase']=='armed', state
    assert time.time()-state['epoch']<20, 'Guard heartbeat stale on '+host
    return original_inspect(host)


def disarm():
    for host in ARMED:
        try:
            result=w.py(host, "from pathlib import Path; import time; p=Path('"+w.REMOTE+"'); (p/'memory-guard.done').touch(); time.sleep(1); print((p/'memory-guard-state.json').read_text())", 15)
            (w.OUT/(host+'-guard-final.json')).write_text(result)
        except Exception as exc:
            print('GUARD_DISARM_ERROR',host,str(exc),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-baseline',type=Path)
    args=parser.parse_args()
    lock=(ROOT/'workflow.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:
        signal.signal(sig,w.interrupted)
    signal.alarm(17400)
    try:
        w.status('arming_local_memory_guards',context=131072,text_only=True)
        trace=ROOT/'docker/model_runner_profile_trace.py'
        for host in w.HOSTS:
            subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(trace),
                            'raulwesche@'+host+':'+w.REMOTE+'/patches/model_runner_profile_trace.py'],check=True,timeout=25)
            observed=w.py(host, "from pathlib import Path; import hashlib; print(hashlib.sha256(Path('"+w.REMOTE+"/patches/model_runner_profile_trace.py').read_bytes()).hexdigest())",20).strip()
            assert observed==hashlib.sha256(trace.read_bytes()).hexdigest()
        guards={host:prepare_guard(host) for host in w.HOSTS}
        (w.OUT/'guards-armed.json').write_text(json.dumps(guards,indent=2))
        # No trial starts without both local guards running.
        w.inspect=guarded_inspect
        w.main(resume_baseline=args.resume_baseline)
    except BaseException as exc:
        w.inspect=original_inspect
        w.status('failed_or_interrupted',error=str(exc),traceback=traceback.format_exc())
        w.collect(w.OUT/'failure')
        try:w.stop_pair()
        finally:disarm()
        raise
    else:
        disarm()
