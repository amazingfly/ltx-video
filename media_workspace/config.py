"""One workspace configuration shared by standalone component commands."""
from __future__ import annotations
import json
import os
from pathlib import Path

COMPONENT_ENV = {'ltx': 'LTX', 'sa3': 'SA3', 'images': 'IMAGES', 'storybook': 'STORYBOOK', 'pipeline': 'PIPELINE'}
PATH_ENV = {'data': 'MEDIA_DATA_ROOT', 'models': 'MEDIA_MODEL_ROOT', 'datasets': 'MEDIA_DATASET_ROOT',
            'runs': 'MEDIA_RUN_ROOT', 'credentials': 'MEDIA_AUTH_ROOT', 'images': 'LTX_IMAGE_DIR',
            'music': 'LTX_MUSIC_DIR', 'ltx_outputs': 'LTX_OUTPUTS_DIR', 'sa3_outputs': 'SA3_OUTPUTS_DIR',
            'sa3_library': 'SA3_LIBRARY_DIR', 'sa3_web_runs': 'SA3_WEB_RUNS_DIR',
            'gemma_base': 'GEMMA_BASE_DIR', 'gemma_server': 'GEMMA_SERVER', 'gemma_model': 'GEMMA_MODEL',
            'gemma_mmproj': 'GEMMA_MMPROJ', 'piper': 'PIPER_BIN', 'voices': 'PIPER_VOICES',
            'ffmpeg': 'FFMPEG_BIN', 'ffprobe': 'FFPROBE_BIN',
            'sd15_model': 'SD15_MODEL', 'sdxl_binary': 'SDXL_BINARY', 'sdxl_model': 'SDXL_MODEL',
            'sdxl_loras': 'SDXL_LORA_DIR', 'qwen_server': 'QWEN_SERVER', 'qwen_model_dir': 'QWEN_MODEL_DIR'}


def config_path(explicit=None):
    return Path(explicit or os.environ.get('MEDIA_WORKSPACE_CONFIG') or
                Path.home() / '.config/agentic-media/workspace.json').expanduser().absolute()


def resolve(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError('Workspace paths must be nonempty strings')
    path = Path(os.path.expandvars(value)).expanduser()
    return str((base / path).absolute() if not path.is_absolute() else path.absolute())


def load_workspace(explicit=None):
    path = config_path(explicit)
    if not path.is_file():
        if explicit or os.environ.get('MEDIA_WORKSPACE_CONFIG'):
            raise FileNotFoundError(f'Workspace configuration not found: {path}')
        return {'version': 1, 'components': {}, 'paths': {}}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Workspace configuration requires version=1')
    if not isinstance(value.get('components', {}), dict) or not isinstance(value.get('paths', {}), dict):
        raise ValueError('Workspace components and paths must be objects')
    result = {'version': 1, 'components': {}, 'paths': {}}
    for name, item in value.get('components', {}).items():
        if name not in COMPONENT_ENV or not isinstance(item, dict):
            raise ValueError(f'Invalid workspace component: {name}')
        root = resolve(item.get('root'), path.parent)
        python = item.get('python', '.venv/bin/python')
        if not isinstance(python, str) or not python:
            raise ValueError(f'Invalid interpreter for {name}')
        result['components'][name] = {'root': root, 'python': resolve(python, Path(root)) if '/' in python else python}
    for name, value in value.get('paths', {}).items():
        if name not in PATH_ENV:
            raise ValueError(f'Unknown workspace path: {name}')
        result['paths'][name] = resolve(value, path.parent)
    return result


def environment(workspace):
    result = {}
    for name, item in workspace['components'].items():
        prefix = COMPONENT_ENV[name]
        result[prefix + '_REPO'] = item['root']
        result[prefix + '_PYTHON'] = item['python']
    for name, value in workspace['paths'].items():
        result[PATH_ENV[name]] = value
    if 'IMAGES_REPO' in result:
        result['SD15_REPO'] = result['IMAGES_REPO']
    return result


def apply_environment(explicit=None):
    """Explicit environment variables win over saved defaults."""
    values = environment(load_workspace(explicit))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values
