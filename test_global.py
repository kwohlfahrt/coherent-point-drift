#!/usr/bin/env python3
from math import pi

def degrade(reference, rotation, translation, scale, drop, duplications, noise):
    from numpy import delete
    from geometry import rotationMatrix, rigidXform
    from itertools import chain, repeat

    points = delete(reference, drop, axis=0)
    rotation_matrix = rotationMatrix(*rotation)
    indices = chain.from_iterable(repeat(i, n) for i, n in enumerate(duplications))
    return rigidXform(points, rotation_matrix, translation, scale)[list(indices)] + noise

def generateDegradation(args, custom_seed):
    # Only use one random number generator, so only one seed
    from numpy.random import choice, uniform, random, seed
    from numpy import array
    from math import sqrt
    from numpy.linalg import norm

    seed(custom_seed)

    if args.D == 2:
        rotation = (uniform(*args.rotate),)
    if args.D == 3:
        angle = uniform(*args.rotate)
        axis = random(3)
        axis = axis/norm(axis)
        rotation = angle, axis
    translation = uniform(*args.translate, size=args.D)
    scale = uniform(*args.scale)
    drops = choice(range(args.N), size=args.drop, replace=False)
    duplications = choice(range(args.duplicate[0], args.duplicate[1] + 1), size=args.N - args.drop)
    noise = args.noise * random((sum(duplications), args.D))

    return rotation, translation, scale, drops, duplications, noise

def generate(args):
    from functools import partial
    from itertools import starmap
    from numpy.random import seed, random, randint
    from numpy import iinfo
    from pickle import dumps
    from sys import stdout
    from align import globalAlignment

    seed(args.seed)
    reference= random((args.N, args.D))
    stdout.buffer.write(dumps(reference))
    seeds = randint(iinfo('int32').max, size=args.repeats)

    degradations = list(map(partial(generateDegradation, args), seeds))
    transformeds = starmap(partial(degrade, reference), degradations)
    # Pool only supports one argument for map, so use starmap + zip
    fits = map(partial(globalAlignment, reference), transformeds)
    for repeat in zip(degradations, fits):
        stdout.buffer.write(dumps(repeat))

def plot(args):
    from pickle import load
    from sys import stdin
    import matplotlib.pyplot as plt
    from itertools import starmap
    from numpy.random import seed, random
    from util import loadAll
    from geometry import rigidXform, RMSD

    seed(4) # For color choice
    reference = load(stdin.buffer)
    plt.figure(0)
    plt.scatter(reference[:, 0], reference[:, 1], marker='D', color='black')

    rmsds = []
    for degradation, fit in loadAll(stdin.buffer):
        plt.figure(0)
        color = random(3)
        degraded = degrade(reference, *degradation)
        plt.scatter(degraded[:, 0], degraded[:, 1], marker='o', color=color, alpha=0.2)
        fitted = rigidXform(degraded, *fit)
        plt.scatter(fitted[:, 0], fitted[:, 1], marker='+', color=color)
        rmsds.append(RMSD(reference, fitted))
    if len(rmsds) > 1:
        plt.figure(1)
        plt.violinplot(rmsds)
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("Test random data for 2D and 3D alignment")
    subparsers = parser.add_subparsers()

    parser_gen = subparsers.add_parser('generate', aliases=['gen'], help="Generate points and fits")
    parser_gen.set_defaults(func=generate)
    parser_gen.add_argument('N', type=int, help='Number of points')
    parser_gen.add_argument('D', type=int, choices=(2, 3), help='Number of dimensions')
    parser_gen.add_argument('repeats', type=int, help='Number of trials to run')
    parser_gen.add_argument('--drop', default=0, type=int,
                        help='number of points to exclude from the reference set')

    parser_gen.add_argument('--rotate', nargs=2, type=float, default=(-pi, pi),
                        help='The range of rotations to test')
    parser_gen.add_argument('--translate', nargs=2, type=float, default=(-1.0, 1.0),
                        help='The range of translations to test')
    parser_gen.add_argument('--scale', nargs=2, type=float, default=(0.5, 1.5),
                        help='The range of scales to test')
    parser_gen.add_argument('--noise', type=float, default=0.01,
                            help='The amount of noise to add')
    parser_gen.add_argument('--duplicate', nargs=2, type=int, default=(1, 1),
                            help='The range of multiples for each point in the degraded set')
    parser_gen.add_argument('--seed', type=int, default=4,
                            help='The random seed for generating a degradation')


    parser_plot = subparsers.add_parser('plot', help="Plot the generated points")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)