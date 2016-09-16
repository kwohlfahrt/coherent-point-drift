# Coherent Point Drift

A python implementation of rigid and affine coherent point drift alignment.
Includes brute-force search for rigid alignment in global rotation space.

## Installation

The library can be installed by running `setup.py`.

## Demo

A demo of global alignment is included under the name `test_global.py`.

Usage is as follows:

    test_global.py gen N D R | ./test_global.py plot

The first generates and aligns a series of transformations of a random point
cloud. The second plots their locations and RMSDs.
