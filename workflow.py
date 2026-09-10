"""Durable download-gated baseline + DFlash sweep on the idle pair only."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time
import traceback
import urllib.request

from prepare import ROOT, HOSTS, IMAGE, REMOTE, SSH
import probes

OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
STATE=ROOT/'workflow-status.json'
START=time.time()
NAME='nvidia-glm53-tp2'


def status(phase,**extra):
    data={'phase':phase,'pid':os.getpid(),'started_epoch':START,'updated_epoch':time.time(),**extra}
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(STATE)
    print(json.dumps(data),flush=True)


def remote(host,cmd,timeout=110,check=True):
    r=subprocess.run(SSH+['raulwesche@'+host,cmd],capture_output=True,text=True,timeout=timeout)
    if check and r.returncode:raise RuntimeError(f'{host}: {cmd}\n{r.stdout}\n{r.stderr}')
    return r


def py(host,code,timeout=110):
    return remote(host,'timeout '+str(timeout-5)+'s python3 -c '+shlex.quote(code),timeout=timeout).stdout


def inspect(host):
    r=remote(host,'timeout 15s docker inspect '+NAME,timeout=20,check=False)
    return json.loads(r.stdout)[0] if r.returncode==0 else None


def stop_pair():
    for host in HOSTS:
        r=remote(host,'timeout 45s python3 '+REMOTE+'/launch_node.py --stop',timeout=55,check=False)
        print('STOP',host,r.returncode,r.stdout,r.stderr,flush=True)


def collect(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for host in HOSTS:
        try:
            r=remote(host,'timeout 20s docker logs --tail 3000 '+NAME,timeout=30,check=False)
            (dest/(host+'-server.log')).write_text(r.stdout+r.stderr)
            s=inspect(host);(dest/(host+'-container.json')).write_text(json.dumps(s,indent=2))
            r=remote(host,'timeout 15s nvidia-smi; free -b; sysctl vm.swappiness',timeout=25,check=False)
            (dest/(host+'-memory.txt')).write_text(r.stdout+r.stderr)
        except Exception as exc:(dest/(host+'-collection-error.txt')).write_text(str(exc))
    try:
        with urllib.request.urlopen(probes.URL+'/metrics',timeout=10) as r:(dest/'metrics.txt').write_bytes(r.read())
    except Exception as exc:(dest/'metrics-error.txt').write_text(str(exc))


def wait_downloads():
    deadline=time.monotonic()+10800
    code="""import json,pathlib
h=pathlib.Path('/home/raulwesche');s=json.loads((h/'nvidia-glm53-download-status.json').read_text());m=json.loads((h/'nvidia-glm53-download-manifest.json').read_text());d=pathlib.Path(s['destination']);s['complete_file_bytes']=sum(f['size'] for f in m['siblings'] if (d/f['rfilename']).is_file() and (d/f['rfilename']).stat().st_size==f['size']);print(json.dumps(s))"""
    while time.monotonic()<deadline:
        states={host:json.loads(py(host,code,25)) for host in HOSTS}
        status('waiting_for_hash_verified_downloads',nodes=states)
        assert all(s['phase']!='failed' for s in states.values()),states
        if all(s['phase']=='complete_verified' for s in states.values()):
            assert len({s['revision'] for s in states.values()})==1
            assert all(s['verified_bytes']==s['expected_bytes'] for s in states.values())
            (OUT/'target-download-readback.json').write_text(json.dumps(states,indent=2));return
        time.sleep(45)
    raise TimeoutError('Download-gated workflow deadline exceeded')


def preflight():
    status('preflight')
    assert (ROOT/'prepared.json').is_file(),'Both runtime dependency checks must pass'
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['launch_node.py','chat_template.jinja','docker/sparse_attn_indexer_kpool_sm121.py']}
    for host in HOSTS:
        for name in ['launch_node.py','chat_template.jinja']:
            subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(ROOT/name),'raulwesche@'+host+':'+REMOTE+'/'+name],check=True,timeout=25)
        code="import hashlib,json,pathlib; p=pathlib.Path('"+REMOTE+"'); print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['launch_node.py','chat_template.jinja']}))"
        observed=json.loads(py(host,code,20));assert all(observed[n]==hashes[n] for n in observed)
        # Verify the target's own tokenizer renders genuinely different ON/OFF suffixes.
        test="from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/model',trust_remote_code=True); s=open('/recipe/chat_template.jinja').read(); msgs=[{'role':'user','content':'Hello'}]; off=t.apply_chat_template(msgs,chat_template=s,tokenize=False,add_generation_prompt=True,enable_thinking=False); on=t.apply_chat_template(msgs,chat_template=s,tokenize=False,add_generation_prompt=True,enable_thinking=True); assert off.endswith('<think></think>') and on.endswith('<think>'); print(repr(off)); print(repr(on))"
        cmd='timeout 70s docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/overlay -e HF_HUB_OFFLINE=1 -v '+REMOTE+'/transformers-overlay:/overlay:ro -v /home/raulwesche/models/nvidia-GLM-5.3-Flash-NVFP4:/model:ro -v '+REMOTE+':/recipe:ro '+IMAGE+' -c '+shlex.quote(test)
        r=remote(host,cmd,timeout=80);(OUT/(host+'-template-proof.txt')).write_text(r.stdout+r.stderr)
    (OUT/'recipe-hashes.json').write_text(json.dumps(hashes,indent=2))
    # Read-only health check; never reconfigure or restart the Hermes pair.
    with urllib.request.urlopen('http://10.0.0.229:8900/v1/models',timeout=15) as r:(OUT/'hermes-before.json').write_bytes(r.read())


def launch(spec,dest):
    status('launching',speculative_tokens=spec)
    for host in reversed(HOSTS):
        r=remote(host,'timeout 100s python3 '+REMOTE+'/launch_node.py --speculative '+str(spec),timeout=110)
        (dest/(host+'-launch.txt')).write_text(r.stdout+r.stderr)
        state=inspect(host);assert state and state['State']['Running'],state
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        for host in HOSTS:
            s=inspect(host)
            if not s or not s['State']['Running']:
                collect(dest);raise RuntimeError('Server exited at startup: '+host)
        try:
            models=probes.api('/v1/models',timeout=5)
            assert any(m['id']==probes.MODEL for m in models['data'])
            (dest/'models-readback.json').write_text(json.dumps(models,indent=2));return
        except (OSError,TimeoutError):pass
        status('loading_weights_or_initializing',speculative_tokens=spec)
        time.sleep(12)
    collect(dest);raise TimeoutError('Engine startup exceeded 30-minute deadline')


def validate_code(path,outdir):
    host=HOSTS[0];dest=REMOTE+'/generated_probe.py'
    tests='''\nimport random,copy
assert merge_intervals([])==[]
assert merge_intervals([[1,3],[2,6],[8,10],[10,18]])==[[1,6],[8,18]]
assert merge_intervals([[5,5],[-4,-1],[-2,3],[4,6]])==[[-4,3],[4,6]]
def oracle(rows):
    answer=[]
    for a,b in sorted(rows):
        if answer and a<=answer[-1][1]:answer[-1][1]=max(answer[-1][1],b)
        else:answer.append([a,b])
    return answer
rng=random.Random(827)
for _ in range(200):
    values=[sorted([rng.randint(-100,100),rng.randint(-100,100)]) for _ in range(rng.randrange(30))]
    original=copy.deepcopy(values)
    assert merge_intervals(values)==oracle(values)
    assert values==original
print('PASS: fixed cases, 200 generated cases, nonmutation')
'''
    testpath=Path(outdir)/'code_with_oracles.py';testpath.write_text(Path(path).read_text()+tests)
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=5',str(testpath),'raulwesche@'+host+':'+dest],check=True,timeout=25)
    cmd='timeout 30s docker run --rm --network none --memory 512m --cpus 1 --pids-limit 64 --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --tmpfs /tmp:rw,noexec,nosuid,size=16m -v '+dest+':/probe.py:ro --entrypoint python3 '+IMAGE+' -I /probe.py'
    r=remote(host,cmd,timeout=40,check=False)
    (Path(outdir)/'code-oracle-result.txt').write_text(r.stdout+r.stderr)
    if r.returncode in [1,124,137]:
        raise probes.QualityFailure('Generated code failed isolated execution/oracles: '+r.stdout+r.stderr)
    if r.returncode:
        raise RuntimeError('Code-test infrastructure failed: '+str(r.returncode)+' '+r.stderr)
    probes.require_quality('PASS:' in r.stdout, 'Code oracle did not report PASS')


def evaluate_quality(dest):
    """Retain answer failures; runtime/transport failures still propagate."""
    failures=[]
    try:
        probes.long_context(dest)
        long_passed=True
    except probes.QualityFailure as exc:
        long_passed=False
        failures.append({'probe':'long-context-120k','category':'semantic','passed':False,'error':str(exc)})
    try:
        probes.quality(dest,validate_code)
        report=json.loads((dest/'quality-report.json').read_text())
    except probes.QualityFailure as exc:
        report={'runtime_checks_passed':False,'all_checks_passed':False,'strict_failures':[],
                'remaining_short_checks_not_completed':True}
        failures.append({'probe':'short-quality','category':'semantic','passed':False,'error':str(exc)})
    report['strict_failures'].extend(failures)
    report['long_context_passed']=long_passed
    report['semantic_checks_passed']=long_passed and report['runtime_checks_passed']
    report['all_checks_passed']=report['all_checks_passed'] and long_passed
    report['status']='passed' if report['all_checks_passed'] else 'completed_with_quality_failures'
    (dest/'quality-report.json').write_text(json.dumps(report,indent=2))
    if not report['all_checks_passed']:
        (dest/'quality-passed.json').unlink(missing_ok=True)
    return report


def load_saved_baseline(receipt):
    """Reuse real completed measurements only when inference inputs match."""
    import shutil
    import launch_node
    receipt=Path(receipt).resolve();source=receipt.parent
    saved=json.loads(receipt.read_text())
    candidates=[r for r in saved if r.get('variant')=='baseline' and r.get('speculative_tokens')==0]
    assert len(candidates)==1, 'Expected exactly one saved baseline'
    baseline=candidates[0]
    assert baseline['runtime_checks_passed'] and baseline['long_context_passed']
    assert len(baseline['rows'])==4 and baseline['median_decode_est_tps']>0
    assert hashlib.sha256((source/'source/probes.py').read_bytes()).digest()==hashlib.sha256((ROOT/'probes.py').read_bytes()).digest(), 'Probe code changed: fresh baseline required'
    fabric=sorted(launch_node.HOSTS.values())
    for host,(rank,ip) in zip(HOSTS,fabric):
        metadata=json.loads((source/'baseline'/(host+'-launch.txt')).read_text().splitlines()[-1])
        assert metadata['command']==launch_node.command(rank,ip,0), 'Inference command changed'
        assert metadata['revision']==launch_node.REV
        assert metadata['template_sha256']==hashlib.sha256((ROOT/'chat_template.jinja').read_bytes()).hexdigest()
    shutil.copytree(source/'baseline',OUT/'baseline')
    (OUT/'baseline-reuse.json').write_text(json.dumps({'source':str(receipt),'source_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'same_inference_commands':True,'same_probe_code':True,'quality_passed':baseline['quality_passed'],'note':'Reuse measurements, not the unsuccessful speculative attempts. No quality failure was rescored.'},indent=2))
    return baseline


def main(resume_baseline=None):
    wait_downloads();preflight();runs=[]
    if resume_baseline:
        runs=[load_saved_baseline(resume_baseline)]
        (OUT/'comparison.json').write_text(json.dumps(runs,indent=2))
    for spec in ([7,3] if resume_baseline else [0,7,3]):
        name='baseline' if not spec else 'dflash2-'+str(spec)
        dest=OUT/name;dest.mkdir(exist_ok=True)
        if runs:stop_pair()
        try:
            launch(spec,dest);status('throughput_probes',variant=name)
            probes.stream(dest,'startup-warmup','Reply with exactly READY.')
            rows=probes.speed(dest)
            status('context_and_quality_probes',variant=name)
            quality_report=evaluate_quality(dest)
            collect(dest)
            counters={}
            if spec:
                metrics=(dest/'metrics.txt').read_text()
                for suffix in ['num_drafts','num_draft_tokens','num_accepted_tokens']:
                    metric='vllm:spec_decode_'+suffix+'_total'
                    counters[suffix]=sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines() if line.startswith(metric+'{') or line.startswith(metric+' '))
                assert counters['num_drafts']>0 and counters['num_draft_tokens']>0,'No measured speculative decoding activity'
                counters['acceptance_rate']=counters['num_accepted_tokens']/counters['num_draft_tokens']
                counters['mean_acceptance_length']=1+counters['num_accepted_tokens']/counters['num_drafts']
                (dest/'speculative-counters.json').write_text(json.dumps(counters,indent=2))
            speed=statistics.median(r['decode_est_tps'] for r in rows)
            record={'variant':name,'speculative_tokens':spec,'quality_passed':quality_report['all_checks_passed'],'runtime_checks_passed':True,'semantic_checks_passed':quality_report['semantic_checks_passed'],'long_context_passed':quality_report['long_context_passed'],'strict_failures':quality_report['strict_failures'],'speculative_counters':counters,'median_decode_est_tps':speed,'median_end_to_end_tps':statistics.median(r['end_to_end_tps'] for r in rows),'rows':[{k:r[k] for k in ['label','usage','decode_est_tps','end_to_end_tps','ttft_s','wall_s']} for r in rows]}
        except Exception as exc:
            collect(dest);record={'variant':name,'speculative_tokens':spec,'quality_passed':False,'error':str(exc),'traceback':traceback.format_exc()}
            (dest/'failed.json').write_text(json.dumps(record,indent=2))
            if spec==0:raise
        runs.append(record);(OUT/'comparison.json').write_text(json.dumps(runs,indent=2))
    passed=[r for r in runs if r.get('runtime_checks_passed')]
    # A semantic failure can be measured, but cannot qualify a draft for selection.
    qualified=[r for r in passed if r.get('semantic_checks_passed')]
    best_failure_count=min((len(r.get('strict_failures',[])) for r in qualified),default=0)
    eligible=[r for r in qualified if len(r.get('strict_failures',[]))==best_failure_count]
    winner=max(eligible,key=lambda r:r['median_decode_est_tps']) if eligible else runs[0]
    fastest_measured=max(passed,key=lambda r:r['median_decode_est_tps'])
    if winner['speculative_tokens']!=runs[-1]['speculative_tokens'] or not runs[-1].get('runtime_checks_passed'):
        stop_pair();dest=OUT/'selected-relaunch';dest.mkdir(exist_ok=True);launch(winner['speculative_tokens'],dest)
        r=probes.stream(dest,'final-health','Reply with exactly READY.');assert r['content'].strip()=='READY'
    baseline=runs[0]
    (OUT/'diagnostic-fastest.json').write_text(json.dumps({'variant':fastest_measured['variant'],'median_decode_est_tps':fastest_measured['median_decode_est_tps'],'speedup_vs_baseline':fastest_measured['median_decode_est_tps']/baseline['median_decode_est_tps'],'quality_passed':fastest_measured['quality_passed'],'semantic_checks_passed':fastest_measured['semantic_checks_passed'],'note':'Measured speed only; does not override quality failures or selection.'},indent=2))
    summary={'status':'complete' if winner['quality_passed'] else 'complete_with_quality_caveats','endpoint':probes.URL+'/v1','model':probes.MODEL,'context':probes.WINDOW,'mode':'text-only','winner':winner,'baseline':baseline,'speedup_vs_baseline':winner['median_decode_est_tps']/baseline['median_decode_est_tps'],'runs':runs,'caveat':'Small synthetic C=1 microbench; decode rate is client streaming estimate, not a quality leaderboard or proof of exact losslessness.'}
    with urllib.request.urlopen('http://10.0.0.229:8900/v1/models',timeout=15) as r:(OUT/'hermes-after.json').write_bytes(r.read())
    summary['selected_models_readback']=probes.api('/v1/models',timeout=10)
    with urllib.request.urlopen(probes.URL+'/health',timeout=10) as response:
        assert response.status==200
        summary['health_http_status']=response.status
    summary['selected_container_readback']={host:inspect(host) for host in HOSTS}
    assert all(s and s['State']['Running'] for s in summary['selected_container_readback'].values())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));status('complete',summary_path=str(OUT/'summary.json'),winner=winner['variant'],speedup=summary['speedup_vs_baseline'])


def interrupted(signum,frame):raise InterruptedError('Workflow interrupted by signal '+str(signum))

if __name__=='__main__':
    lock=(ROOT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    signal.signal(signal.SIGALRM,interrupted);signal.alarm(18000)
    try:main()
    except BaseException as exc:
        status('failed_or_interrupted',error=str(exc),traceback=traceback.format_exc())
        collect(OUT/'failure');stop_pair();raise
