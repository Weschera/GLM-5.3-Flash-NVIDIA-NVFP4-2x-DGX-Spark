"""Offline evidence verification; never sends traffic or changes run receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

LABELS = {f'speed-{language}-{repeat}' for language in ('prose', 'korean') for repeat in range(2)}
ARMS = ('baseline', 'dflash2-7', 'dflash2-3')


def load(path):
    return json.loads(path.read_text())


def verify_sample(directory, label, reference_request=None):
    record = load(directory / f'{label}-result.json')
    request = load(directory / f'{label}-request.json')
    if reference_request is not None:
        assert request == reference_request, f'{label}: request differs from baseline'
    content, reasoning, usage, finish, first = '', '', None, None, None
    previous = -1.0
    ids = set()
    raw_path = directory / f'{label}-stream.jsonl'
    for line in raw_path.read_text().splitlines():
        row = json.loads(line)
        elapsed, event = row['elapsed_s'], row['event']
        assert elapsed >= previous, f'{label}: nonmonotonic stream'
        previous = elapsed
        assert not event.get('error'), event
        ids.add(event['id'])
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            delta = choice.get('delta', {})
            c = delta.get('content') or ''
            r = delta.get('reasoning') or delta.get('reasoning_content') or ''
            if first is None and (c or r or delta.get('tool_calls')):
                first = elapsed
            content += c
            reasoning += r
            finish = choice.get('finish_reason') or finish
    assert len(ids) == 1, f'{label}: mixed or empty streams'
    assert content == record['content'], f'{label}: content mismatch'
    assert reasoning == record['reasoning'], f'{label}: reasoning mismatch'
    assert usage == record['usage'], f'{label}: usage mismatch'
    assert finish == record['finish_reason'] == 'stop', f'{label}: not a natural stop'
    assert record['sse_done'], f'{label}: parser did not record DONE'
    assert first == record['ttft_s'] and previous <= record['wall_s']
    assert usage['prompt_tokens'] == record['prompt_tokens_tokenize']
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    assert (content + reasoning).count('\ufffd') == record['replacement_characters'] == 0
    rate = (usage['completion_tokens'] - 1) / (record['wall_s'] - first)
    assert math.isclose(rate, record['decode_est_tps'], rel_tol=1e-9)
    assert math.isclose(usage['completion_tokens'] / record['wall_s'], record['end_to_end_tps'], rel_tol=1e-9)
    return {'label': label, 'decode_est_tps': rate, 'completion_tokens': usage['completion_tokens'],
            'stream_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}


def verify_campaign(root):
    arms = []
    missing = []
    for arm in ARMS:
        directory = root / arm
        actual = {p.name.removesuffix('-result.json') for p in directory.glob('speed-*-result.json')}
        assert actual <= LABELS, f'{arm}: unexpected speed samples {actual - LABELS}'
        missing.extend(f'{arm}/{label}' for label in sorted(LABELS - actual))
        rows = []
        for label in sorted(actual):
            reference = load(root / 'baseline' / f'{label}-request.json')
            rows.append(verify_sample(directory, label, reference))
        long_pass = None
        if (directory / 'long-context-passed.json').exists():
            verify_sample(directory, 'long-context-120k', load(root / 'baseline/long-context-120k-request.json'))
            long_result = load(directory / 'long-context-120k-result.json')
            assert json.loads(long_result['content']) == ['VIOLET-7391', 'AMBER-5826', 'CYAN-1948']
            assert 118000 <= long_result['usage']['prompt_tokens'] <= 122000
            long_pass = True
        arms.append({'arm': arm, 'verified_speed_count': len(rows),
                     'median_decode_est_tps': statistics.median(row['decode_est_tps'] for row in rows) if rows else None,
                     'long_context_verified': long_pass, 'rows': rows})
    return {'status': 'speed_matrix_verified' if not missing else 'partial',
            'expected_speed_samples': len(ARMS) * len(LABELS),
            'verified_speed_samples': sum(a['verified_speed_count'] for a in arms),
            'missing_speed_samples': missing, 'arms': arms,
            'scope': 'Raw SSE content, usage, timing, natural stop, exact request parity, and saved retrieval receipts. Not independent proof of quality or losslessness. DONE is trusted from the parser receipt because raw JSONL does not retain its literal marker.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = verify_campaign(args.root)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)
    if report['missing_speed_samples'] and not args.allow_partial:
        raise SystemExit(2)
