#!/usr/bin/env python3
"""Carry the exact legal texts of the two pinned public Flash build sources."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCES = {
    'BigMoeOnEdge': {
        'repository': 'https://github.com/Helldez/BigMoeOnEdge',
        'revision': '930b2c9ebd8c2cface6566bd9f11075ac07adfe0',
        'license': 'Apache-2.0',
        'sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
        'notice': None,
    },
    'llama.cpp': {
        'repository': 'https://github.com/Helldez/llama.cpp',
        'revision': '7ac98f4a8089a36c0581b7c6e8973417a8fa175c',
        'license': 'MIT',
        'sha256': '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d',
        'notice': None,
    },
}
MANIFEST = {'schema': 'flash-source-licenses-v1', 'sources': SOURCES}


def checked(path, pin):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError('missing or invalid Flash source licence: ' + path.name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('Flash source licence hash changed: ' + path.name)
    return raw


def collect(bigmoe, llama, destination):
    """Caller has verified both source commits and the patched llama tree."""
    texts = {}
    for name, source in zip(SOURCES, (Path(bigmoe), Path(llama))):
        # Neither exact pinned source has a root NOTICE. A new notice requires
        # explicit review rather than silently dropping it or inventing one.
        if any(p.name.upper().startswith('NOTICE') for p in source.iterdir()):
            raise ValueError('unexpected upstream NOTICE: ' + name)
        texts[name] = checked(source / 'LICENSE', SOURCES[name]['sha256'])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in texts.items():
        (destination / ('LICENSE-' + name)).write_bytes(raw)
    (destination / 'FLASH-SOURCE-LICENSES.json').write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + '\n')


def package(source, stage):
    source, stage = Path(source), Path(stage)
    metadata = source / 'FLASH-SOURCE-LICENSES.json'
    if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 16384:
        raise ValueError('Flash source licence manifest missing or invalid')
    if json.loads(metadata.read_text()) != MANIFEST:
        raise ValueError('Flash source licence pins changed')
    texts = {name: checked(source / ('LICENSE-' + name), pin['sha256']) for name, pin in SOURCES.items()}
    for name, raw in texts.items():
        (stage / 'licenses' / (name + '-LICENSE')).write_bytes(raw)
    shutil.copyfile(metadata, stage / 'FLASH-SOURCE-LICENSES.json')


if __name__ == '__main__':
    try:
        if len(sys.argv) == 5 and sys.argv[1] == 'collect':
            collect(*sys.argv[2:])
        elif len(sys.argv) == 4 and sys.argv[1] == 'package':
            package(*sys.argv[2:])
        else:
            raise ValueError('usage: flash_licenses.py collect BIGMOE LLAMA DEST | package SOURCE STAGE')
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
