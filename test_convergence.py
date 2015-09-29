#!/usr/bin/env python3
from test_util import generateDegradation, degrade

def process(reference, transformed):
    from coherent_point_drift.align import driftRigid
    from coherent_point_drift.geometry import rigidXform, RMSD
    from itertools import starmap, islice
    from functools import partial

    fits = islice(driftRigid(reference, transformed), 200)
    fitteds = starmap(partial(rigidXform, transformed), fits)
    rmsds = map(partial(RMSD, reference), fitteds)
    return list(rmsds)

def generate(args):
    from multiprocessing import Pool
    from functools import partial
    from itertools import starmap
    from numpy.random import seed, random, randint
    from numpy import iinfo
    from pickle import dumps
    from sys import stdout

    seed(4)
    reference= random((args.N, args.D))
    stdout.buffer.write(dumps(reference))
    seeds = randint(iinfo('int32').max, size=args.repeats)

    degradations = list(map(partial(generateDegradation, args), seeds))
    transformeds = starmap(partial(degrade, reference), degradations)

    with Pool() as p:
        # Pool only supports one argument for map, so use starmap + zip
        rmsds = p.imap(partial(process, reference), transformeds)
        for repeat in zip(degradations, rmsds):
            stdout.buffer.write(dumps(repeat))

def plot(args):
    from pickle import load
    from sys import stdin
    import matplotlib.pyplot as plt
    from itertools import starmap
    from numpy.random import seed, random
    from coherent_point_drift.util import loadAll
    from coherent_point_drift.geometry import rigidXform, RMSD
    from math import degrees

    seed(4)
    reference = load(stdin.buffer)

    rmsds = []
    converged = 0
    # This is strict :(
    degradations, rmsds = zip(*loadAll(stdin.buffer))
    plt.figure(0)
    for rmsd in rmsds:
        color = 'red' if len(rmsd) < 100 else 'blue'
        plt.plot(rmsd, color=color, alpha=0.3)
    rotations = map(lambda x: degrees(x[0][0]), degradations)
    min_rmsds= map(min, rmsds)
    rotation_rmsds = sorted(zip(rotations, min_rmsds), key=lambda x: x[0])

    plt.figure(1)
    plt.plot(*zip(*rotation_rmsds))
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    from math import pi

    parser = ArgumentParser("Test random data for 2D and 3D local alignment convergence")
    subparsers = parser.add_subparsers()
    parser_gen = subparsers.add_parser('generate', aliases=['gen'],
                                       help="Generate points and RMSD-sequences")
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

    parser_plot = subparsers.add_parser('plot', help="Plot the genrated convergence rates")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
