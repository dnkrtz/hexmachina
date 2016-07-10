# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='AllHex',
    version='0.0.1',
    description='Hexahedral meshing approach based on SRF-method.',
    long_description=readme,
    author='Aidan Kurtz',
    author_email='aidan@mail.mcgill.ca',
    url='https://github.com/dnkrtz/AllHex',
    license=license,
    packages=find_packages(exclude=('tests'))
)