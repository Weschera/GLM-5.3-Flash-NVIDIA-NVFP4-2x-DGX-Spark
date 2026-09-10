"""Best-effort local RAM guard, scoped strictly to this experiment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/raulwesche/nvidia-glm53-tp2')
NAME = 'nvidia-glm53-tp2'
STATE = ROOT / 'memory-guard-state.json'
DONE = ROOT / 'memory-guard.done'
FLOOR = 6 * 1024**3
DEADLINE = time.monotonic() + 18000


def save(phase, **extra):
    data = {'phase': phase, 'pid': os.getpid(), 'epoch': time.time(), **extra}
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(STATE)


def stop_owned(reason, available):
    # Capture compiler/worker RSS before stopping, with only a one-second budget.
    try:
        ps = subprocess.run(['ps','-eo','pid,ppid,rss,comm,args','--sort=-rss'],
                            capture_output=True,text=True,timeout=1)
        (ROOT/'memory-guard-trigger-diagnostics.json').write_text(json.dumps({
            'epoch':time.time(), 'reason':reason,
            'meminfo':Path('/proc/meminfo').read_text(),
            'top_processes':ps.stdout.splitlines()[:25]},indent=2))
    except Exception:
        pass
    # Never kill a process or container based only on its name.
    result = subprocess.run(['docker', 'inspect', NAME], capture_output=True,
                            text=True, timeout=8)
    if result.returncode:
        save('triggered_container_absent', reason=reason, available_bytes=available)
        return
    container = json.loads(result.stdout)[0]
    assert container['Config']['Labels'].get('wesche.task') == NAME
    if container['State']['Running']:
        result = subprocess.run(['docker', 'kill', container['Id']],
                                capture_output=True, text=True, timeout=12)
        observed = subprocess.run(['docker', 'inspect', container['Id']],
                                  capture_output=True, text=True, timeout=8)
        after = json.loads(observed.stdout)[0]['State']
        save('triggered', reason=reason, available_bytes=available,
             kill_returncode=result.returncode, container_state=after)
        assert not after['Running'], after
    else:
        save('triggered_already_stopped', reason=reason, available_bytes=available)


def main():
    save('armed', floor_bytes=FLOOR)
    last_save = 0
    while not DONE.exists():
        info = {line.split(':')[0]: int(line.split()[1]) * 1024
                for line in Path('/proc/meminfo').read_text().splitlines()}
        available = info['MemAvailable']
        if available < FLOOR:
            stop_owned('host_memory_floor', available)
            return
        if time.monotonic() >= DEADLINE:
            stop_owned('campaign_deadline', available)
            return
        if time.monotonic() - last_save >= 3:
            save('armed', floor_bytes=FLOOR, available_bytes=available)
            last_save = time.monotonic()
        time.sleep(0.5)
    save('disarmed')


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Guard deadline')))
    signal.alarm(18100)
    try:
        main()
    except BaseException as exc:
        save('error', error=str(exc))
        raise
