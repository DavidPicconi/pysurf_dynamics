#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from os.path import abspath, join as joinpath

# Write the name of the local source directory in the package subfolder
cwd = abspath('.')
with open(joinpath('pysurf', 'source_path.txt'), 'w') as fh:
    fh.write(cwd)
#

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pycolt>=0.5.3', 'qctools>=0.3.0', 'netcdf4>=1.5', 'numpy>=1.21', 'scipy>=1.7', 'jinja2>=3.1']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Maximilian Menger, Johannes Ehrmaier",
    author_email='m.f.s.j.Menger@rug.nl',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License 2.0',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python Surface Hopping Code",
    entry_points={
        'console_scripts': [],
    },
    install_requires=requirements,
    license="Apache License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pysurf',
    name='pysurf',
    #packages=find_packages(include=['pysurf', 'pysurf.*'] + ['plugins']),
    packages=find_packages(include=['pysurf', 'pysurf.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mfsjmenger/pysurf',
    version='0.2.0',
    zip_safe=False,
)
