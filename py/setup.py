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
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, 'src')


def _repo_source_files():
    if not os.path.isdir(src_dir):
        return []
    return [
        name for name in sorted(os.listdir(src_dir))
        if name.endswith(('.c', '.h'))
    ]


def _needs_sync(files):
    if not files:
        return False
    if not os.path.isdir(csrc):
        return True
    for name in files:
        src = os.path.join(src_dir, name)
        dst = os.path.join(csrc, name)
        if not os.path.isfile(dst):
            return True
        if os.path.getmtime(src) > os.path.getmtime(dst):
            return True
    return False


repo_files = _repo_source_files()
sync_script = os.path.join(here, 'scripts', 'sync-csrc.py')
if _needs_sync(repo_files):
    print('setup.py: syncing csrc/ from repo sources...')
    subprocess.check_call([sys.executable, sync_script])
elif not os.path.isdir(csrc):
    sys.exit(
        'csrc/ directory not found and repo src/ is unavailable.\n'
        'Run from the repo: python py/scripts/sync-csrc.py'
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
