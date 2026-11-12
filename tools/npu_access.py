#!/usr/bin/env python3
"""Stage the exact DMA-heap grant; apply and rollback require explicit switches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

RULE = Path('/etc/udev/rules.d/72-local-ai-dma-heap.rules')
HEAP = Path('/dev/dma_heap/system')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def rule_for(uid):
    if not isinstance(uid, int) or uid <= 0:
        raise ValueError('a non-root installing uid is required')
    return ('# Local AI: exact system heap, installing user only. O-51.\n'
            f'SUBSYSTEM=="dma_heap", KERNEL=="system", OWNER:="{uid}", '
            'MODE:="0400"\n')


def sudo(*args):
    subprocess.run(['sudo', '-n', *map(str, args)], check=True)


def refresh():
    sudo('udevadm', 'control', '--reload-rules')
    sudo('udevadm', 'trigger', '--action=change', '--subsystem-match=dma_heap',
         '--sysname-match=system')
    sudo('udevadm', 'settle', '--timeout=10')


def check_stage(stage):
    if stage.is_symlink() or stage.stat().st_uid != os.getuid() or stat.S_IMODE(stage.stat().st_mode) != 0o700:
        raise ValueError('stage must be an owner-only directory')
    m = json.loads((stage / 'manifest.json').read_text())
    if m['uid'] != os.getuid() or m['rule_path'] != str(RULE):
        raise ValueError('stage identity mismatch')
    if (stage / 'rule').read_text() != rule_for(m['uid']) or digest(stage / 'rule') != m['installed_sha256']:
        raise ValueError('staged rule changed')
    return m


def stage_at(stage):
    if not stat.S_ISCHR(HEAP.stat().st_mode):
        raise ValueError('system heap is not a character device')
    stage.mkdir(mode=0o700)
    m = {'uid': os.getuid(), 'rule_path': str(RULE), 'old_sha256': digest(RULE),
         'old_rule_mode': stat.S_IMODE(RULE.stat().st_mode) if RULE.exists() else None}
    if RULE.exists():
        (stage / 'rule.original').write_bytes(RULE.read_bytes())
    (stage / 'rule').write_text(rule_for(m['uid']))
    m['installed_sha256'] = digest(stage / 'rule')
    acl = subprocess.check_output(['getfacl', '-p', str(HEAP)])
    (stage / 'heap.original.acl').write_bytes(acl)
    m['original_acl_sha256'] = digest(stage / 'heap.original.acl')
    (stage / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(json.dumps({'staged': str(stage), 'applied': False, **m}))


def apply(stage):
    m = check_stage(stage)
    if digest(RULE) != m['old_sha256']:
        raise ValueError('live rule preimage changed')
    sudo('install', '-o', 'root', '-g', 'root', '-m', '0644', stage / 'rule', RULE)
    refresh()
    st = HEAP.stat()
    if st.st_uid != m['uid'] or stat.S_IMODE(st.st_mode) != 0o400:
        raise ValueError('udev did not apply the exact owner/read-only grant; rollback required')
    fd = os.open(HEAP, os.O_RDONLY | os.O_CLOEXEC)
    os.close(fd)  # permission QC only; no allocation or CDSP session
    if not os.access('/dev/fastrpc-cdsp', os.R_OK | os.W_OK):
        raise ValueError('existing active-seat CDSP access is missing; rollback required')
    print(json.dumps({'applied': True, 'heap_uid': st.st_uid, 'heap_mode': '0400',
                      'read_open': True, 'cdsp_access': True, 'device_session_opened': False}))


def rollback(stage):
    m = check_stage(stage)
    if digest(RULE) != m['installed_sha256']:
        raise ValueError('live rule is not this stage; refusing stale rollback')
    if digest(stage / 'heap.original.acl') != m['original_acl_sha256']:
        raise ValueError('original ACL changed')
    if m['old_sha256'] is None:
        sudo('rm', '--', RULE)
    else:
        if digest(stage / 'rule.original') != m['old_sha256']:
            raise ValueError('original rule changed')
        sudo('install', '-o', 'root', '-g', 'root', '-m', f"{m['old_rule_mode']:04o}", stage / 'rule.original', RULE)
    refresh()
    sudo('setfacl', '-P', '--restore=' + str(stage / 'heap.original.acl'))
    print(json.dumps({'rolled_back': True, 'heap_acl_restored': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    if args.apply:
        apply(args.stage)
    elif args.rollback:
        rollback(args.stage)
    else:
        stage_at(args.stage)
