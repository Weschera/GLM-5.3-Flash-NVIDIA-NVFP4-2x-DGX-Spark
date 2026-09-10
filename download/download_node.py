"""Download one pinned official checkpoint and verify every Hub file."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '15'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
REPO = 'nvidia/GLM-5.3-Flash-NVFP4'
REVISION = '423acf37583782c51c142d145aef733d72943d93'
DEST = Path('/home/raulwesche/models/nvidia-GLM-5.3-Flash-NVFP4')
STATE = Path('/home/raulwesche/nvidia-glm53-download-status.json')
START = time.time()


def status(phase, **extra):
    record = dict(repo=REPO, revision=REVISION, destination=str(DEST),
                  host=os.uname().nodename, phase=phase, started_epoch=START,
                  updated_epoch=time.time(), **extra)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(STATE)
    print(json.dumps(record), flush=True)


def alarm(signum, frame):
    raise TimeoutError('Per-file verification exceeded 110 seconds')


def main():
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        manifest = json.load(response)
    assert manifest['sha'] == REVISION
    files = manifest['siblings']
    assert files and all(isinstance(f.get('size'), int) for f in files)
    expected = sum(f['size'] for f in files)
    DEST.mkdir(parents=True, exist_ok=True)
    if DEST.is_symlink() or str(DEST.resolve()) != str(DEST):
        raise RuntimeError('Unexpected symlink destination')
    fs = os.statvfs(DEST)
    existing = sum((DEST/f['rfilename']).stat().st_size for f in files
                   if (DEST/f['rfilename']).is_file())
    if fs.f_bavail * fs.f_frsize < max(0, expected-existing) + 20_000_000_000:
        raise RuntimeError('Insufficient local disk space plus staging reserve')
    STATE.with_name('nvidia-glm53-download-manifest.json').write_text(json.dumps(manifest, indent=2))
    status('downloading', expected_bytes=expected, expected_files=len(files))
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(DEST), max_workers=4)
    status('verifying', expected_bytes=expected, expected_files=len(files))
    receipts = []
    signal.signal(signal.SIGALRM, alarm)
    for f in sorted(files, key=lambda f: f['rfilename']):
        path = DEST / f['rfilename']
        if not path.is_relative_to(DEST) or path.is_symlink():
            raise RuntimeError('Unsafe file path')
        before = path.stat()
        if before.st_size != f['size']:
            raise RuntimeError('Size mismatch: ' + str(path))
        lfs = f.get('lfs')
        if lfs:
            hasher = hashlib.sha256()
            expected_hash = lfs['sha256']
            algorithm = 'sha256'
        else:
            hasher = hashlib.sha1()
            hasher.update(f'blob {before.st_size}\0'.encode())
            expected_hash = f['blobId']
            algorithm = 'git-blob-sha1'
        signal.alarm(110)
        try:
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                    hasher.update(chunk)
        finally:
            signal.alarm(0)
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if identity(before) != identity(after) or hasher.hexdigest() != expected_hash:
            raise RuntimeError('Hash or identity mismatch: ' + str(path))
        receipts.append(dict(path=f['rfilename'], bytes=before.st_size,
                             algorithm=algorithm, digest=expected_hash))
        print(json.dumps(dict(verified=f['rfilename'], count=len(receipts), total=len(files))), flush=True)
    receipt_path = STATE.with_name('nvidia-glm53-download-verified.json')
    receipt_path.write_text(json.dumps(dict(repo=REPO, revision=REVISION,
                                destination=str(DEST), verified_files=receipts), indent=2))
    status('complete_verified', expected_bytes=expected, verified_bytes=sum(r['bytes'] for r in receipts),
           verified_file_count=len(receipts), receipt=str(receipt_path))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
