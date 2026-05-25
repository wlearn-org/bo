import os
import subprocess
import sys
from setuptools import setup, Extension

if sys.platform == 'win32':
    sys.exit(
        'wlearn-bo requires Linux. Windows is not supported.'
    )

here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')

if not os.path.isdir(csrc):
    sync_script = os.path.join(here, 'scripts', 'sync-csrc.py')
    if os.path.isfile(sync_script):
        print('setup.py: csrc/ not found, running sync-csrc.py...')
        subprocess.check_call([sys.executable, sync_script])
    else:
        sys.exit(
            'csrc/ directory not found and scripts/sync-csrc.py is missing.\n'
            'Run from the repo root: python scripts/sync-csrc.py'
        )

sources = []
for root, dirs, files in os.walk('csrc'):
    for f in sorted(files):
        if f.endswith('.c') and 'test' not in f:
            sources.append(os.path.join(root, f))
sources.append(os.path.join('wlearn_bo', '_native.c'))

setup(
    ext_modules=[
        Extension(
            'wlearn_bo._native',
            sources=sources,
            include_dirs=[csrc],
            extra_compile_args=['-std=c11', '-O2'],
            libraries=['m'],
        )
    ]
)
