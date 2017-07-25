#!/usr/bin/env python3

from .align import driftRigid, driftAffine, globalAlignment
from .geometry import rigidXform, affineXform, RMSD
from .util import last
from itertools import islice, filterfalse
from operator import contains
from functools import partial
from pickle import load, dump, HIGHEST_PROTOCOL
dump = partial(dump, protocol=HIGHEST_PROTOCOL)
from pathlib import Path
from sys import stdout
import numpy as np


try:
    from scipy.io import savemat
except ImportError:
    savemat = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def loadPoints(path):
    if path.suffix == '.txt':
        return np.loadtxt(str(path))
    if path.suffix == '.csv':
        return np.loadtxt(str(path), delimiter=',')
    elif path.suffix == '.pickle':
        with path.open('rb') as f:
            return load(f)
    else:
        raise ValueError("File type '{}' not recognized (need '.csv' or '.pickle')"
                         .format(path.suffix))


def savePoints(f, points, fmt):
    delimiters = {'txt': ' ', 'csv': ','}
    if fmt == "pickle":
        dump(points, f)
    elif fmt in delimiters:
        delimiter = delimiters[fmt]
        lines = map(str.encode, map(delimiter.join, map(partial(map, str), points)))
        f.write(b'\n'.join(lines))
        f.write(b'\n')
    else:
        raise ValueError("Invalid format: {}".format(fmt))


def loadXform(path):
    if path.suffix == ".pickle":
        with path.open("rb") as f:
            return load(f)
    elif path.suffix == ".mat":
        if loadmat is None:
            raise RuntimeError("Loading .mat files not supported in SciPy")
        xform = loadmat(str(path))
        if set("Rts") <= set(xform.keys()):
            return xform['R'], xform['t'], xform['s']
        if set("Bt") <= set(xform.keys()):
            return xform['B'], xform['t']
        else:
            raise RuntimeError("Invalid transform format"
                               "(must contain [R, t, s] or [B, t]), not {}"
                               .format(list(xform.keys())))
    else:
        raise ValueError("Invalid transform file type (need .pickle or .mat)")


def saveXform(f, xform, fmt):
    if fmt == "pickle":
        dump(xform, f)
    elif fmt == "mat":
        if len(xform) == 3:
            labels = "R", "t", "s"
        elif len(xform) == 2:
            labels = "B", "t"
        else:
            raise ValueError("Invalid xform")
        savemat(f, dict(zip(labels, xform)))
    elif fmt == "print":
        lines = map(str.encode, map(' '.join, map(partial(map, str), xform[0])))
        f.write(b'\n'.join(lines))
        f.write(b'\n\n')
        f.write(' '.join(map(str, xform[1])).encode())
        f.write(b'\n\n')
        if len(xform) > 2:
            f.write(str(xform[2]).encode())
        f.write(b'\n')
    else:
        raise ValueError("Invalid format: {}".format(fmt))


def plot(args):
    from numpy import delete
    import matplotlib

    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = list(map(loadPoints, args.points))
    xform = loadXform(args.transform)

    if len(xform) == 2:
        transform = affineXform
    elif len(xform) == 3:
        transform = rigidXform

    proj_axes = tuple(filterfalse(partial(contains, args.axes), range(points[0].shape[1])))
    points = list(map(partial(delete, obj=proj_axes, axis=1),
                      points + [transform(points[1], *xform)]))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(*points[0].T[::-1], color='red')
    ax.scatter(*points[1].T[::-1], color='blue')
    ax.scatter(*points[2].T[::-1], marker='x', color='blue')
    if args.outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(args.outfile))


# TODO: Test this, should be straightforward!
def xform(args):
    points = loadPoints(args.points)
    xform = loadXform(args.transform)

    if len(xform) == 2:
        transformed = affineXform(points, *xform)
    elif len(xform) == 3:
        transformed = rigidXform(points, *xform)

    savePoints(stdout.buffer, transformed, args.format)


def align(args):
    if len(args.points) < 2:
        raise ValueError("Must provide at least 2 point sets")
    points = list(map(loadPoints, args.points))
    reference, target = points[0::2], points[1::2]

    reference_classes = np.repeat(np.arange(len(reference)), list(map(len, reference)))
    target_classes = np.repeat(np.arange(len(target)), list(map(len, target)))
    w = np.repeat(np.broadcast_to(args.w, len(target)), list(map(len, target)))

    prior = reference_classes.reshape(1, -1) == target_classes.reshape(-1, 1)
    prior = prior / prior.sum(axis=-1, keepdims=True) * w.reshape(-1, 1)

    reference, target = map(np.concatenate, [reference, target])

    if args.mode == "rigid":
        if args.scope == "global":
            xform = globalAlignment(
                reference, target, prior, mirror=True, maxiter=args.niter, processes=args.j
            )
        else:
            xform = last(islice(driftRigid(reference, target, prior), args.niter))
    elif args.mode == "affine":
        if args.scope == "global":
            raise NotImplementedError("Global affine alignment is not yet implemented.")
        else:
            xform = last(islice(driftAffine(reference, target, prior), args.niter))

    saveXform(stdout.buffer, xform, args.format)


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Align two sets of points.")
    subparsers = parser.add_subparsers()

    points_help = "The point sets to align (in pickle, csv or txt format)"
    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("points", nargs='+', type=Path, help=points_help)
    align_parser.add_argument("-w", type=float, nargs='+', default=[0.5],
                              help="The 'w' parameter for the CPD algorithm")
    align_parser.add_argument("--mode", type=str, choices={"rigid", "affine"},
                              default="rigid", help="The type of drift to use.")
    align_parser.add_argument("--niter", type=int, default=200,
                              help="The number of iterations to use.")
    align_parser.add_argument("--scope", type=str, choices={"global", "local"},
                              default="local", help="Use global alignment instead of local.")
    align_parser.add_argument("-j", type=int, default=None,
                              help="Number of processes to launch for global search")

    output_options = {"pickle", "print"}
    if savemat is not None:
        output_options.add("mat")
        align_parser.add_argument("--format", type=str, choices=output_options,
                                  default="print", help="Output format")
    align_parser.set_defaults(func=align)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("points", nargs=2, type=Path, help=points_help)
    plot_parser.add_argument("transform", type=Path,
                             help="The transform")
    plot_parser.add_argument("--axes", type=int, nargs=2, default=(0, 1),
                             help="The axes to plot")
    plot_parser.add_argument("--outfile", type=Path,
                             help="Where to save the plot (omit to display)")
    plot_parser.set_defaults(func=plot)

    xform_parser = subparsers.add_parser("transform", aliases=["xform"])
    xform_parser.add_argument("points", type=Path, help=points_help)
    xform_parser.add_argument("transform", type=Path, help="The transform")
    xform_parser.add_argument("--format", type=str, choices={"pickle", "txt", "csv"},
                              default="txt", help="Output format")
    xform_parser.set_defaults(func=xform)

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)


if __name__ == "__main__":
    main()
