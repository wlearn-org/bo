import os
import subprocess
import sys
from setuptools import setup, Extension, find_packages

if sys.platform == 'win32':
    sys.exit(
        'wlearn-bo requires Linux. Windows is not supported.'
    )

here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')
repo_root = os.path.dirname(here)
src_dir = os.path.join(repo_root, 'src')
with open(os.path.join(here, 'README.md'), encoding='utf-8') as handle:
    long_description = handle.read()


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
    name='wlearn-bo',
    version='0.1.0',
    description=(
        'Bayesian optimization with Gaussian processes for '
        'hyperparameter tuning'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anton Zemlyansky',
    license_files=['LICENSE', 'NOTICE'],
    url='https://wlearn.org',
    project_urls={
        'Repository': 'https://github.com/wlearn-org/bo',
        'Issues': 'https://github.com/wlearn-org/bo/issues',
    },
    python_requires='>=3.9',
    platforms=['Linux'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=(
        'bayesian-optimization gaussian-process '
        'hyperparameter-tuning automl wlearn'
    ),
    packages=find_packages(),
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
