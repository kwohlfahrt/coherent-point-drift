#!/usr/bin/env python3

from coherent_point_drift.align import driftRigid, driftAffine, globalAlignment
from coherent_point_drift.geometry import rigidXform, affineXform, RMSD
from coherent_point_drift.util import last
from itertools import islice
from functools import partial
from pickle import load, dump
from pathlib import Path

try:
    from scipy.io import savemat
except ImportError:
    savemat = None

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

def loadPoints(path):
    if path.suffix == '.csv':
        from numpy import loadtxt
        return loadtxt(str(path))
    elif path.suffix == '.pickle':
        with path.open('rb') as f:
            return load(f)
    else:
        raise ValueError("File type '{}' not recognized (need '.csv' or '.pickle')"
                         .format(path.suffix))


def plot(args):
    import matplotlib.pyplot as plt
    from numpy import delete

    points = list(map(loadPoints, args.points))
    if args.transform.suffix == ".pickle":
        with args.transform.open("rb") as f:
            xform = load(f)
    elif args.transform.suffix == ".mat":
        if loadmat is None:
            raise RuntimeError("Loading .mat files not supported in SciPy")
        xform = loadmat(str(args.transform))
        if set(xform.keys()) == set("Rts"):
            xform = xform['R'], xform['t'], xform['s']
        if set(xform.keys()) == set("Bt"):
            xform = xform['B'], xform['t']
        else:
            raise RuntimeError("Invalid transform format"
                               "(must contain [R, t, s] or [B, t]), not {}"
                               .format(list(xform.keys())))
    else:
        raise ValueError("Invalid transform file type (need .pickle or .mat)")

    if len(xform) == 2:
        transform = affineXform
    elif len(xform) == 3:
        transform = rigidXform
    else:
        raise RuntimeError("Transform must have 2 or 3 elements, not {}"
                           .format(len(xform)))

    if args.axes is not None:
        points = list(map(partial(delete, obj=args.axes, axis=1),
                          points + [transform(points[1], *xform)]))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(*points[0].T[::-1], color='red')
    ax.scatter(*points[1].T[::-1], color='blue')
    ax.scatter(*points[2].T[::-1], marker='x', color='blue')
    plt.show()

def align(args):
    from sys import stdout

    points = list(map(loadPoints, args.points))

    if args.mode == "rigid":
        if args.scope == "global":
            xform = globalAlignment(*points, w=args.w, maxiter=args.niter, mirror=True,
                                    processes=args.j)
        else:
            xform = last(islice(driftRigid(*points, w=args.w), args.niter))
    elif args.mode == "affine":
        if args.scope == "global":
            raise NotImplementedError("Global affine alignment is not yet implemented.")
        else:
            xform = last(islice(driftAffine(*points, w=args.w), args.niter))

    if args.output == "pickle":
        dump(xform, stdout.buffer)
    elif args.output == "mat":
        if args.mode == "rigid":
            output = dict(zip(("R", "t", "s"), xform))
        elif args.mode == "affine":
            output = dict(zip(("B", "t"), xform))
        savemat(stdout.buffer, output)
    elif args.output == "print":
        print(*xform, sep='\n')


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Align two sets of points.")
    subparsers = parser.add_subparsers()

    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("points", nargs=2, type=Path,
                              help="The point sets to align (in pickle or csv format)")
    align_parser.add_argument("-w", type=float, default=0.5,
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
        align_parser.add_argument("--output", type=str, choices=output_options,
                                  default="print", help="Output format")
    align_parser.set_defaults(func=align)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("points", nargs=2, type=Path,
                             help="The points to plot (in pickle or csv format)")
    plot_parser.add_argument("transform", type=Path,
                             help="The transform")
    plot_parser.add_argument("--axes", type=int, nargs='+', default=None,
                             help="The axes to project")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)


if __name__ == "__main__":
    main()
