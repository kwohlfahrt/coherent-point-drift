#!/usr/bin/env python3

from coherent_point_drift.align import driftRigid, driftAffine, globalAlignment
from coherent_point_drift.geometry import rigidXform, affineXform, RMSD
from coherent_point_drift.util import last
from itertools import islice
from pickle import load, dump
from pathlib import Path

try:
    from scipy.io import savemat
except ImportError:
    savemat = None

def loadPoints(path):
    if path.suffix == '.csv':
        pass
    elif path.suffix == '.pickle':
        with path.open('rb') as f:
            return load(f)
    else:
        raise ValueError("File type '{}' not recognized (need '.csv' or '.pickle')"
                         .format(path.suffix))

def main(args=None):
    from argparse import ArgumentParser
    from sys import argv, stdout

    parser = ArgumentParser(description="Align two sets of points.")
    parser.add_argument("points", nargs=2, type=Path,
                        help="The point sets to align (in pickle format)")
    parser.add_argument("-w", type=float, default=0.5,
                        help="The 'w' parameter for the CPD algorithm")
    parser.add_argument("--mode", type=str, choices={"rigid", "affine"}, default="rigid",
                        help="The type of drift to use.")
    parser.add_argument("--niter", type=int, default=200,
                        help="The number of iterations to use.")
    parser.add_argument("--scope", type=str, choices={"global", "local"}, default="local",
                        help="Use global alignment instead of local.")
    output_options = {"pickle", "print"}
    if savemat is not None:
        output_options.add("mat")
    parser.add_argument("--output", type=str, choices=output_options, default="print",
                        help="Output format")

    args = parser.parse_args(argv[1:] if args is None else args)
    points = list(map(loadPoints, args.points))

    if args.mode == "rigid":
        if args.scope == "global":
            xform = globalAlignment(*points, w=args.w, maxiter=args.niter, mirror=True)
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
        savemat(stdout, output)
    elif args.output == "print":
        print(*xform, sep='\n')

if __name__ == "__main__":
    main()
