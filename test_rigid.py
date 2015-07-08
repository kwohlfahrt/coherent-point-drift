#!/usr/bin/env python3
from math import pi

def tryAlignment(reference, R, t, s):
    from geometry import rigidXform, rotationMatrix
    from align import globalAlignment

    moved = rigidXform(reference, rotationMatrix(*R), t, s)
    return globalAlignment(reference, moved)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from numpy.random import rand, seed, shuffle
    from multiprocessing import Pool
    from itertools import repeat, starmap

    parser = ArgumentParser("Test random data for 2D and 3D alignment")
    parser.add_argument('N', type=int, help='Number of points')
    parser.add_argument('D', type=int, choices=(2, 3), help='Number of dimensions')
    parser.add_argument('repeats', type=int, help='Number of trials to run')
    parser.add_argument('--drop', default=0, type=int,
                        help='number of points to exclude from the reference set')

    parser.add_argument('--rotate', nargs=2, type=float, default=(-pi, pi),
                        help='The range of rotations to test')
    parser.add_argument('--translate', nargs=2, type=float, default=(-1.0, 1.0),
                        help='The range of translations to test')
    parser.add_argument('--scale', nargs=2, type=float, default=(0.5, 1.5),
                        help='The range of scales to test')

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pickle', type=str)

    seed(4)

    args = parser.parse_args()
    reference = rand(args.N, args.D)

    # Could generate per-process if memory is an issue
    if args.D == 2:
        rotations = rand(args.repeats, 1) * (args.rotate[1] - args.rotate[0]) + args.rotate[0]
    if args.D == 3:
        angles = rand(args.repeats) * (args.rotate[1] - args.rotate[0]) + args.rotate[0]
        axes = rand(args.repeats, 3)
        rotations = list(zip(angles, axes)) # Used twice, for processing and pickling
    translations = (rand(args.repeats, args.D) * (args.translate[1] - args.translate[0])
                    + args.translate[0])
    scales = rand(args.repeats) * (args.scale[1] - args.scale[0]) + args.scale[0]

    with Pool() as p:
        # Pool only supports one argument for map, so use starmap + zip
        xforms = p.starmap(tryAlignment, zip(repeat(reference), rotations, translations, scales))

    if args.pickle:
        import pickle
        from pathlib import Path
        pickle_path = Path(args.pickle)
        with pickle_path.open('wb') as f:
            for data in (reference, xforms, rotations, translations, scales):
                pickle.dump(data, f)

    if args.plot:
        import matplotlib.pyplot as plt
        from geometry import rigidXform, rotationMatrix

        plt.scatter(reference[:, 0], reference[:, 1], marker='v', color='black')
        rotation_matrices = starmap(rotationMatrix, rotations)
        moveds = map(rigidXform, repeat(reference), rotation_matrices, translations, scales)
        for xform, moved in zip(xforms, moveds):
            color = rand(3)
            plt.scatter(moved[:, 0], moved[:, 1], marker='o', color=color, alpha=0.5)
            fitted = rigidXform(moved, *xform)
            plt.scatter(fitted[:, 0], fitted[:, 1], marker='+', color=color)
        plt.show()
