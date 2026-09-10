"""Real streaming probes; no output truncation or synthetic inference results."""
import json
from pathlib import Path
import re
import time
import urllib.request
import signal
from contextlib import contextmanager


@contextmanager
def request_deadline(seconds):
    """Hard wall deadline while preserving an earlier campaign deadline."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    duration = min(seconds, previous_timer[0]) if previous_timer[0] else seconds
    def expired(*_):
        raise TimeoutError('Inference request or campaign wall deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        remaining = previous_timer[0] - (time.monotonic() - started)
        if previous_timer[0] and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])

URL='http://10.0.0.109:8910'
MODEL='nvidia/GLM-5.3-Flash-NVFP4'
WINDOW=131072


def api(path,payload=None,timeout=110):
    request=urllib.request.Request(URL+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)


def stream(outdir,label,prompt,thinking=False,extra=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    base={'model':MODEL,'messages':[{'role':'user','content':prompt}],
          'temperature':0,'top_p':1,'seed':37,'repetition_penalty':1,
          'chat_template_kwargs':{'enable_thinking':thinking}}
    if extra:base.update(extra)
    tok_payload={k:v for k,v in base.items() if k in ['model','messages','tools','chat_template_kwargs']}
    tok_payload['add_generation_prompt']=True
    tok=api('/tokenize',tok_payload)
    prompt_tokens=tok['count'];assert 0<prompt_tokens<WINDOW,tok
    # Use the entire remaining serving window, not an arbitrary generation cap.
    base.update(max_tokens=WINDOW-prompt_tokens,stream=True,stream_options={'include_usage':True})
    (outdir/(label+'-request.json')).write_text(json.dumps(base,ensure_ascii=False,indent=2))
    request=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(base).encode(),headers={'Content-Type':'application/json'})
    begin=time.monotonic();first=None;last=None;content='';reasoning='';usage={};finish=None;tools={};done=False
    with request_deadline(3600), (outdir/(label+'-stream.jsonl')).open('w') as raw:
        # Near-window prefill may legitimately take longer than a health check.
        with urllib.request.urlopen(request,timeout=1800 if prompt_tokens>=100000 else 110) as response:
            for line in response:
                if time.monotonic()-begin>3600:raise TimeoutError('Request exceeded one-hour wall deadline')
                if not line.startswith(b'data:'):continue
                text=line[5:].strip()
                now=time.monotonic()
                if text==b'[DONE]':done=True;break
                data=json.loads(text)
                raw.write(json.dumps({'elapsed_s':now-begin,'event':data},ensure_ascii=False)+'\n');raw.flush()
                if data.get('error'):raise RuntimeError(data['error'])
                if data.get('usage'):usage=data['usage']
                for choice in data.get('choices',[]):
                    delta=choice.get('delta',{});c=delta.get('content') or '';r=delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if c or r or delta.get('tool_calls'):
                        first=first if first is not None else now;last=now
                    content+=c;reasoning+=r
                    for t in delta.get('tool_calls',[]):
                        entry=tools.setdefault(t['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                        if t.get('id'):entry['id']=t['id']
                        for k,v in t.get('function',{}).items():entry['function'][k]+=v or ''
                    if choice.get('finish_reason'):finish=choice['finish_reason']
    end=time.monotonic();n=usage.get('completion_tokens',0)
    record={'label':label,'prompt':prompt,'thinking':thinking,'prompt_tokens_tokenize':prompt_tokens,
            'usage':usage,'finish_reason':finish,'content':content,'reasoning':reasoning,
            'tool_calls':list(tools.values()),'sse_done':done,'wall_s':end-begin,
            'ttft_s':None if first is None else first-begin,
            'decode_est_tps':None if first is None or n<2 else (n-1)/max(end-first,1e-9),
            'end_to_end_tps':n/(end-begin),'replacement_characters':(content+reasoning).count('\ufffd')}
    (outdir/(label+'-result.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    assert done and finish not in [None,'length'] and n>0,record
    assert not record['replacement_characters'], 'Corrupted UTF-8 token output; see '+label
    if not thinking:assert not reasoning,'Thinking OFF leaked reasoning: '+label
    assert '<think>' not in content and '</think>' not in content,'Unparsed thinking markers'
    print(json.dumps({k:record[k] for k in ['label','usage','finish_reason','wall_s','ttft_s','decode_est_tps','replacement_characters']}),flush=True)
    return record

JSON_PROMPT='Return only a JSON object with keys total and label. total must be the sum of 137, 286, and 409. label must be exactly NVIDIA.'
CODE_PROMPT='Write only Python code defining merge_intervals(intervals). Input is a list of closed integer intervals [start,end] with start <= end. Return a list of lists sorted by start, merging overlapping or touching intervals. Empty input returns []. Do not mutate input. Include type hints, a clear docstring, and comments explaining the algorithm, but no imports or top-level example execution.'
KOREAN_PROMPT='인공지능의 역사를 1950년대부터 현재까지 시대별로 나누어 한국어로 자세히 설명해 주세요. 각 시대의 주요 접근 방법, 성과와 한계를 설명하고 약 900단어 분량의 완결된 글로 작성하세요.'
PROSE_PROMPT='Write a complete technical tutorial of about 1000 words explaining tensor parallel inference across two machines. Cover sharding, all-reduce, prefill versus decode, KV memory, bandwidth versus latency, why quantized MoE inference can remain memory-bound, and a practical debugging checklist. Use concrete examples and distinguish estimates from measurements.'


class QualityFailure(Exception):
    """A generated answer failed grading; not proof of a server failure."""


def require_quality(condition, message):
    if not condition:
        raise QualityFailure(message)


def quality_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise QualityFailure('Generated answer is not valid JSON: '+str(exc)) from exc


def quality(outdir,code_validator):
    results=[];format_failures=[]
    r=stream(outdir,'warmup','Reply with exactly READY.');require_quality(r['content'].strip()=='READY', 'warmup: expected exactly READY');results.append(r)
    r=stream(outdir,'json',JSON_PROMPT);require_quality(quality_json(r['content'])=={'total':sum([137,286,409]),'label':'NVIDIA'}, 'JSON arithmetic contract failed');results.append(r)
    r=stream(outdir,'code',CODE_PROMPT)
    source=r['content'].strip()
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',source,re.S)
    if blocks:
        require_quality(len(blocks)==1, 'Expected one code block')
        format_failures.append({'probe':'code','requirement':'Only Python code, no Markdown fence','passed':False})
        source=blocks[0]
    source_path=Path(outdir)/'generated_merge_intervals.py';source_path.write_text(source)
    code_validator(source_path,outdir);results.append(r)
    tool={'type':'function','function':{'name':'lookup_inventory','description':'Look up inventory by SKU.','parameters':{'type':'object','properties':{'sku':{'type':'string'}},'required':['sku'],'additionalProperties':False}}}
    r=stream(outdir,'tool','Use lookup_inventory to look up SKU NV-42. Do not guess the inventory.',extra={'tools':[tool],'tool_choice':'auto'})
    require_quality(r['finish_reason']=='tool_calls' and len(r['tool_calls'])==1, 'Expected one tool call')
    call=r['tool_calls'][0];require_quality(call['function']['name']=='lookup_inventory' and quality_json(call['function']['arguments'])=={'sku':'NV-42'}, 'Tool name or arguments failed');results.append(r)
    messages=[{'role':'user','content':'Use lookup_inventory to look up SKU NV-42. Then return only the stock count as an integer.'},{'role':'assistant','content':None,'tool_calls':[call]},{'role':'tool','tool_call_id':call['id'],'content':'{"sku":"NV-42","stock":17}'}]
    r=stream(outdir,'tool-roundtrip','',extra={'tools':[tool],'messages':messages,'tool_choice':'none'});require_quality(r['content'].strip()=='17', 'Tool roundtrip stock answer failed');results.append(r)
    r=stream(outdir,'thinking-on','Find the sum of the integers from 1 through 37. Give just the numeric answer after reasoning.',thinking=True)
    if r['content'].strip()!=str(sum(range(1,38))):
        format_failures.append({'probe':'thinking-on','requirement':'Final content exactly 703','passed':False,'observed':r['content']})
    require_quality(bool(r['reasoning']), 'Thinking ON did not expose reasoning')
    results.append(r)
    report={'probes':[r['label'] for r in results],'runtime_checks_passed':True,
            'all_checks_passed':not format_failures,'strict_failures':format_failures,
            'status':'passed' if not format_failures else 'completed_with_strict_failures'}
    (Path(outdir)/'quality-report.json').write_text(json.dumps(report,indent=2))
    if not format_failures:
        (Path(outdir)/'quality-passed.json').write_text(json.dumps(report,indent=2))
    return results


def long_context(outdir):
    # Exercise a genuinely near-128K context, not merely a 128K server flag.
    # Retain a substantial generation reserve and use the target tokenizer.
    def archive_prompt(n):
        lines=[f'Record {i:05d}: Routine inventory maintenance and warehouse procedures. No verification marker is present in this background record.' for i in range(n)]
        lines.insert(n//2, 'The middle archive marker is AMBER-5826.')
        return ('Read this archive as data. The first verification marker is VIOLET-7391.\n<archive>\n'
                +'\n'.join(lines)+'\n</archive>\nThe final marker is CYAN-1948.\n'
                +'Return only a JSON array of the first, middle, and final marker strings, in that order.')
    n=4000
    for _ in range(10):
        long_prompt=archive_prompt(n)
        count=api('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':long_prompt}],
                  'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['count']
        if 118000<=count<=122000:break
        n=max(1, int(n*120000/count))
    else:raise RuntimeError('Could not calibrate a near-128K prompt')
    r=stream(outdir,'long-context-120k',long_prompt)
    assert 118000<=r['prompt_tokens_tokenize']<=122000
    require_quality(quality_json(r['content'])==['VIOLET-7391','AMBER-5826','CYAN-1948'], 'Near-128K marker retrieval failed')
    (Path(outdir)/'long-context-passed.json').write_text(json.dumps({'prompt_tokens':r['prompt_tokens_tokenize'],'passed':True}))
    return r


def speed(outdir):
    rows=[]
    for repeat in range(2):
        for name,prompt in [('prose',PROSE_PROMPT),('korean',KOREAN_PROMPT)]:
            r=stream(outdir,f'speed-{name}-{repeat}',prompt)
            assert len(r['content'])>1000,'Unexpectedly short throughput answer'
            rows.append(r)
            (Path(outdir)/'speed-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    return rows
