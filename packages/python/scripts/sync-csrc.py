#!/usr/bin/env python3
"""
Sync C sources from the repo root into packages/python/csrc/ for sdist builds.
"""

import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.dirname(here)
repo_csrc = os.path.normpath(os.path.join(pkg_root, '..', '..', 'csrc'))
dst_csrc = os.path.join(pkg_root, 'csrc')

if not os.path.isdir(repo_csrc):
    raise FileNotFoundError(
        f'Cannot find repo csrc/ at {repo_csrc}. '
        'Run this script from the bo/packages/python/ directory.'
    )

if os.path.isdir(dst_csrc):
    shutil.rmtree(dst_csrc)

os.makedirs(dst_csrc)

for fname in sorted(os.listdir(repo_csrc)):
    src = os.path.join(repo_csrc, fname)
    if os.path.isfile(src) and (fname.endswith('.c') or fname.endswith('.h')):
        if 'test' in fname:
            continue
        shutil.copy2(src, os.path.join(dst_csrc, fname))
        print(f'  copied {fname}')

print(f'Synced {repo_csrc} -> {dst_csrc}')
