# Coherent Point Drift

[![Build Status](https://travis-ci.org/kwohlfahrt/coherent-point-drift.svg?branch=master)](https://travis-ci.org/kwohlfahrt/coherent-point-drift)
[![codecov](https://codecov.io/gh/kwohlfahrt/coherent-point-drift/branch/master/graph/badge.svg)](https://codecov.io/gh/kwohlfahrt/coherent-point-drift)

A python implementation of rigid and affine alignment using the
[coherent point drift][cpd] algorithm. Includes brute-force search for rigid
alignment in global rotation space.

# Installation

Installing the package via `setup.py` is the recommended usage. This provides
the executable `cpd`. Detailed instructions can be found in
the [official documentation][setuptools]. It can also be run from the source
directory with `python3 -m coherent_point_drift.main`.

## Dependencies

[Python 3][python] and [Numpy][numpy]. Numpy is available from [PyPI][pypi] and
can be installed as described in the [pip documentation][pip-install]. To
support plotting, [matplotlib][matplotlib] is required. To support MATLAB
compatible output, [scipy][scipy] is required. These are all available
from [PyPI][pypi].

# Usage

Detailed instructions and lists of supported formats are available with the
`--help` option, but general use is as follows:

    cpd align reference.csv points.csv
    
This will produce a transform to align the points in `points.csv` to the points
in `reference.csv` (other input formats are supported).

To save the resulting transform (e.g. for plotting), specify the `--format`
argument. Then, redirect the output into your target file:

    cpd align reference.csv points.csv --format pickle > xform.pickle

The alignment can be visualized with the `plot` command:

    cpd plot reference.csv points.csv xform.pickle
    
Or, it can be applied to an arbitrary set of points with the `xform` command:

    cpd xform points.csv xform --format txt
    
Note `--format` specifies the format of the output, the input format is guessed
from the file extension.

### Demo

Some sample points and a transform can be found in the `tests/fixtures`
directory, as `ref.pickle`, `deg.pickle` and `xform.pickle` respectively.

[setuptools]: https://docs.python.org/3.3/install/#the-new-standard-distutils
[Python]: https://python.org
[Numpy]: https://www.numpy.org
[matplotlib]: http://matplotlib.org
[cpd]: http://dx.doi.org/10.1109/TPAMI.2010.46
[pypi]: https://pypi.python.org/pypi
[pip-install]: https://pip.pypa.io/en/stable/user_guide/#installing-packages
[scipy]: https://scipy.org
