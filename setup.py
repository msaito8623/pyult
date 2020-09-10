from setuptools import setup

pkg = __import__('pyult')
author = pkg.__author__
email = pkg.__author_email__
version = pkg.__version__
license = pkg.__license__
description = pkg.__description__
classifiers = pkg.__classifiers__

def load_requirements (fname):
    """
    Read a requirement.txt
    """
    with open(fname, 'r') as f:
        return [ i.rstrip() for i in list(f) if i and not i.startswith('#') ]

setup(
    name='pyult',
    version=version,
    license=license,
    description=description,
    long_description=open('README.rst').read(),
    author=author,
    author_email=email,
    url='https://github.com/msaito8623/pyult',
    classifiers=classifiers,
    platform='Linux',
    packages=['pyult'],
    install_requires=load_requirements('requirements.txt'),
    extras_require={
        'tests': ['pytest'],
        'docs': ['sphinx', 'sphinx_rtd_theme']
        },
    )
