#!/usr/bin/env python3

from setuptools import setup

setup(name="coherent_point_drift",
      version="0.4.0",
      description="A library for aliging point clouds",
      packages=['coherent_point_drift'],
      requires=['numpy (>=1.10)'],
      entry_points={'console_scripts': ['cpd=coherent_point_drift.main:main']}
)
