#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='bnn_bo',
      version='0.1',
      packages=find_packages(include=['pybnn']),
      # package_dir = {'pybnn': 'pybnn'},
      # packages=['pybnn'],
      install_requires=[
            'botorch',
            'gpytorch'
      ]
)