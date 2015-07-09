#!/usr/bin/env python3
from math import pi

def generate(N, D, ndrops=0, rotate_range=(-pi,pi),
             translate_range=(-1, 1), scale_range=(0.5, 1.5),
             custom_seed=None):
    # Only use one random number generator, so only one seed
    from numpy.random import choice, uniform, random, seed
    from math import sqrt
    from numpy.linalg import norm

    if custom_seed is not None:
        seed(custom_seed)

    if D == 2:
        rotation = (uniform(*rotate_range),)
    if D == 3:
        angle = uniform(*rotate_range)
        axis = random(3)
        axis = axis/norm(axis)
        rotation = angle, axis
    translation = uniform(*translate_range, size=D)
    scale = uniform(*scale_range)
    drops = choice(range(N), size=ndrops, replace=False)

    return rotation, translation, scale, drops

def degrade(reference, rotation, translation, scale, drop):
    from numpy import delete
    from geometry import rotationMatrix, rigidXform

    points = delete(reference, drop, axis=0)
    rotation_matrix = rotationMatrix(*rotation)
    return rigidXform(points, rotation_matrix, translation, scale)

if __name__ == '__main__':
    from argparse import ArgumentParser

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
    parser.add_argument('--load', type=str)

    args = parser.parse_args()
    if not (args.plot or args.pickle):
        from warnings import warn
        warn("No action specified, data will be discarded.")

    if args.load is None:
        from numpy.random import seed, rand
        from multiprocessing import Pool
        from itertools import starmap
        from functools import partial
        from align import globalAlignment

        seed(4)
        reference = rand(args.N, args.D)
        seeds = range(args.repeats)
        generator = partial(generate, args.N, args.D, args.drop,
                            args.rotate, args.translate, args.scale)
        transforms = list(map(generator, seeds))
        transformeds = starmap(partial(degrade, reference), transforms)
        with Pool() as p:
            # Pool only supports one argument for map, so use starmap + zip
            fits = p.map(partial(globalAlignment, reference), transformeds)
    else:
        from util import loadAll
        from pathlib import Path
        load_path = Path(args.load)
        with load_path.open('rb') as f:
            reference, transforms, fits = loadAll(f)

    if args.pickle:
        from pickle import dump
        from pathlib import Path
        pickle_path = Path(args.pickle)
        with pickle_path.open('wb') as f:
            for data in (reference, transforms, fits):
                dump(data, f)

    if args.plot:
        import matplotlib.pyplot as plt
        from itertools import starmap
        from numpy.random import seed, random
        from geometry import rigidXform, RMSD

        seed(4)
        plt.figure()
        colors = random((len(transforms), 3))
        moveds = list(starmap(partial(degrade, reference), transforms))
        for color, moved in zip(colors, moveds):
            plt.scatter(moved[:, 0], moved[:, 1], marker='o', color=color, alpha=0.2)
        fitteds = list(map(rigidXform, moveds, *zip(*fits)))
        plt.scatter(reference[:, 0], reference[:, 1], marker='v', color='black')
        for color, fitted in zip(colors, fitteds):
            plt.scatter(fitted[:, 0], fitted[:, 1], marker='+', color=color)
        plt.figure()
        rmsds = list(map(partial(RMSD, reference), fitteds))
        plt.violinplot(rmsds)
        plt.show()
