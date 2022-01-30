import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = """
        StochasticMDD
        """

# Setup
setup(
    name='stochmdd',
    description=descr,
    long_description=open(src('README.md')).read(),
    long_description_content_type='text/markdown',
    author='mrava87',
    author_email='matteoravasi@gmail.com',
    install_requires=['numpy >= 1.15.0', 'scipy', 'pylops', 'torch', 'pylops-gpu'],
    extras_require={'advanced': ['llvmlite', 'numba']},
    packages=find_packages(exclude=['pytests']),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('stochmdd/version.py')),
    setup_requires=['pytest-runner', 'setuptools_scm'],
    test_suite='pytests',
    tests_require=['pytest'],
    zip_safe=True)
