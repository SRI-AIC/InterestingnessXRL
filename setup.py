#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='InterestingnessXRL',
      version='1.0',
      description='A python library for eXplainable Reinforcement Learning (XRL) based on the concept of interestingness elements.',
      author='Pedro Sequeira',
      author_email='pedrodbs@gmail.com',
      url='https://github.com/pedrodbs/InterestingnessXRL',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=['numpy', 'gym', 'pillow', 'jsonpickle', 'pygame', 'pandas', 'matplotlib', 'scipy', 'pyfpgrowth'],
      zip_safe=True
      )
