"""Inspect workspace paths or launch a component in its configured environment."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .config import load_workspace, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['show', 'doctor', 'run', 'verify', 'catalog'])
    parser.add_argument('--config')
    parser.add_argument('--component')
    # Command arguments must follow --; options before it are parsed normally.
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index('--') if '--' in raw else len(raw)
    command = raw[split+1:]
    args = parser.parse_args(raw[:split])
    try:
        if args.command == 'catalog':
            from .workflows import validate_catalog
            value = validate_catalog(Path(__file__).resolve().parent.parent)
            print(f"Workflow catalog verified: {len(value['workflows'])} entries"); return 0
        if args.command == 'verify':
            import hashlib
            package = Path(__file__).parent
            manifest = package / 'SOURCE.json'
            if not manifest.exists():
                print('Canonical shared package'); return 0
            hashes = json.loads(manifest.read_text())['sha256']
            if set(hashes) != {p.name for p in package.glob('*.py')}:
                raise ValueError('Shared package file set differs from SOURCE.json')
            for name, expected in hashes.items():
                if hashlib.sha256((package/name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Shared package modified: {name}')
            print('Shared package checksums verified'); return 0
        workspace = load_workspace(args.config)
        env = {**environment(workspace), **os.environ}
        if args.command == 'show':
            print(json.dumps(workspace, indent=2)); return 0
        selected = workspace['components']
        if args.component:
            if args.component not in selected:
                raise ValueError(f'Component not configured: {args.component}')
            selected = {args.component: selected[args.component]}
        if not selected:
            raise ValueError('No workspace components configured')
        for name, item in selected.items():
            if not Path(item['root']).is_dir() or not shutil.which(item['python']):
                raise ValueError(f'{name}: missing checkout or interpreter')
        if args.command == 'doctor':
            print(f'Workspace ready: {", ".join(selected)}'); return 0
        if not args.component or not command:
            raise ValueError('run requires --component NAME -- COMMAND [ARGS]')
        item = selected[args.component]
        python = shutil.which(item['python'])
        env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
        command = [item['python'] if value == '{python}' else value for value in command]
        return subprocess.run(command, cwd=item['root'], env=env, check=False).returncode
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr); return 1
