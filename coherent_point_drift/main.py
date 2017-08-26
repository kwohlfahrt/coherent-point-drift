#!/usr/bin/env python3

from .align import driftRigid, driftAffine, globalAlignment
from .geometry import RigidXform, AffineXform, RMSD
from .util import last
from itertools import islice, filterfalse
import operator as op
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
            return RigidXform(xform['R'], xform['t'], xform['s'])
        if set("Bt") <= set(xform.keys()):
            return AffineXform(xform['B'], xform['t'])
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
        if isinstance(xform, RigidXform):
            savemat(f, {"R": xform.R, "t": xform.t, "s": xform.s})
        if isinstance(xform, AffineXform):
            savemat(f, {"B": xform.B, "t": xform.t})
        else:
            raise ValueError("Invalid xform")
    elif fmt == "print":
        f.write(str(xform).encode())
        f.write(b'\n')
    else:
        raise ValueError("Invalid format: {}".format(fmt))


def plot(args):
    from numpy import delete
    import matplotlib

    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(args.points) < 2:
        raise ValueError("Must provide at least 2 point sets")
    points = list(map(loadPoints, args.points))
    reference, target = points[0::2], points[1::2]
    ndim = points[0].shape[1]
    sizes = np.broadcast_to(args.sizes, len(reference))

    xformed = list(map(partial(op.matmul, loadXform(args.transform)), target))

    proj_axes = tuple(filterfalse(partial(op.contains, args.axes), range(ndim)))
    project = partial(delete, obj=proj_axes, axis=1)

    colors = list(map("C{}".format, range(10)))
    fig, axs = plt.subplots(1, 3, figsize=args.figsize, sharex=True, sharey=True)
    for i, (ax, pointss) in enumerate(zip(axs, [reference, target, xformed])):
        for color, size, points in zip(colors, sizes, map(project, pointss)):
            fc = ec = color
            if i == 2 and args.reference:
                fc = 'none'
            ax.scatter(*points.T[::-1], s=size, color=fc, edgecolor=ec, marker='o')
        ax.set_xticks([])
        ax.set_yticks([])
    if args.reference:
        for color, size, points in zip(colors, sizes, map(project, reference)):
            axs[-1].scatter(*points.T[::-1], s=size, color=color, marker='+')
    titles = ["Reference", "Data", "Transformed"]
    for ax, title in zip(axs, titles):
        ax.set_title(title)

    if args.outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(args.outfile))


def xform(args):
    points = loadPoints(args.points)
    transformed = loadXform(args.transform) @ points
    savePoints(stdout.buffer, transformed, args.format)


def align(args):
    if len(args.points) < 2:
        raise ValueError("Must provide at least 2 point sets")
    points = list(map(loadPoints, args.points))
    reference, target = points[0::2], points[1::2]

    reference_classes = np.repeat(np.arange(len(reference)), list(map(len, reference)))
    target_classes = np.repeat(np.arange(len(target)), list(map(len, target)))
    w = np.repeat(np.broadcast_to(args.w, len(target)), list(map(len, target)))

    prior = reference_classes.reshape(-1, 1) == target_classes.reshape(1, -1)
    prior = prior / prior.sum(axis=1, keepdims=True) * (1 - w).reshape(1, -1)

    reference, target = map(np.concatenate, [reference, target])

    if args.mode == "rigid":
        if args.scope == "global":
            P, xform = globalAlignment(
                reference, target, prior, mirror=False, maxiter=args.niter, processes=args.j
            )
        else:
            P, xform = last(islice(driftRigid(reference, target, prior), args.niter))
    elif args.mode == "affine":
        if args.scope == "global":
            raise NotImplementedError("Global affine alignment is not yet implemented.")
        else:
            P, xform = last(islice(driftAffine(reference, target, prior), args.niter))

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
                              help="The probability of points being outliers")
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
    plot_parser.add_argument("points", nargs='+', type=Path, help=points_help)
    plot_parser.add_argument("transform", type=Path,
                             help="The transform")
    plot_parser.add_argument("--figsize", type=float, nargs=2, default=(9, 3),
                             help="The size of the rsulting figure")
    plot_parser.add_argument("--axes", type=int, nargs=2, default=(0, 1),
                             help="The axes to plot")
    plot_parser.add_argument("--sizes", type=float, nargs='+', default=[0.5],
                             help="The size of markers (for each class)")
    plot_parser.add_argument("--reference", action="store_true",
                             help="Plot the reference in the alignment view")
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
