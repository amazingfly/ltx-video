"""Validate the maintained/experimental/archive workflow index."""
import json
from pathlib import Path


def validate_catalog(root):
    root = Path(root)
    value = json.loads((root/'workflows.json').read_text())
    if value.get('version') != 1 or not isinstance(value.get('workflows'), list):
        raise ValueError('Invalid workflow catalog version or entries')
    seen = set()
    for workflow in value['workflows']:
        name = workflow['id']
        if name in seen or workflow.get('status') not in {'supported','experimental','archived'}:
            raise ValueError(f'Invalid workflow id/status: {name}')
        seen.add(name)
        if not workflow.get('entrypoints'):
            raise ValueError(f'{name}: no entrypoints')
        for entry in workflow['entrypoints']:
            path = (root/entry).resolve()
            if not path.is_relative_to(root.resolve()) or not path.exists():
                raise ValueError(f'{name}: missing or external entrypoint {entry}')
    return value
