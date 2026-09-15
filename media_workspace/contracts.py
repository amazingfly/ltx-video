"""Versioned artifact envelopes; existing component manifests remain payloads."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

KINDS = {'audio', 'image', 'video', 'story'}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths):
    """Hash declared inputs only, including directory content and relative names."""
    result = []
    for raw in paths:
        path = Path(raw).absolute()
        if not path.exists():
            raise FileNotFoundError(f'Artifact path not found: {path}')
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        result.append({'path': str(path), 'directory': path.is_dir(), 'files': [
            {'path': str(p), 'relative': str(p.relative_to(path)) if path.is_dir() else p.name,
             'bytes': p.stat().st_size, 'sha256': hash_file(p)} for p in files]})
    return result


def write_contract(path, *, kind, paths, producer):
    if kind not in KINDS:
        raise ValueError(f'Unknown artifact kind: {kind}')
    artifacts = [item for group in snapshot(paths) for item in group['files']]
    if not artifacts:
        raise ValueError('Artifact contract cannot be empty')
    result = {'schema': 'agentic-media/artifacts', 'version': 1, 'kind': kind,
              'producer': producer, 'artifacts': artifacts}
    validate_contract(result, verify=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(path)
    return result


def validate_contract(value, *, verify=False, expected_kind=None):
    if not isinstance(value, dict) or value.get('schema') != 'agentic-media/artifacts' or (type(value.get('version')) is not int or value['version'] != 1):
        raise ValueError('Unsupported artifact contract schema/version')
    if value.get('kind') not in KINDS or (expected_kind and value['kind'] != expected_kind):
        raise ValueError('Unexpected artifact kind')
    if not isinstance(value.get('producer'), dict) or not isinstance(value.get('artifacts'), list) or not value['artifacts']:
        raise ValueError('Artifact contract requires producer and nonempty artifacts')
    seen = set()
    for item in value['artifacts']:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not Path(item['path']).is_absolute():
            raise ValueError('Artifact requires an absolute path')
        digest = item.get('sha256', '')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Artifact requires a SHA-256 digest')
        if type(item.get('bytes')) is not int or item['bytes'] < 0 or item['path'] in seen:
            raise ValueError('Invalid artifact size or duplicate path')
        seen.add(item['path'])
        if verify:
            path = Path(item['path'])
            if not path.is_file() or path.stat().st_size != item['bytes'] or hash_file(path) != digest:
                raise ValueError(f'Artifact changed or missing: {path}')
    return value
