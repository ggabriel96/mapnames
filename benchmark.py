import argparse
import json
from functools import partial
from random import sample
from time import time

from ortools.graph import pywrapgraph as ortg

from mapnames import graph
from mapnames import string


def report_flow(matcher, json_dict, errs):
    if matcher.solve_status == ortg.SimpleMinCostFlow.OPTIMAL:
        for arc in errs:
            l, r = matcher.endpoints(arc)
            print('----------------------> %s\n'
                  '  got mapped to       : %s\n'
                  '  but should have been: %s\n'
                  '  Cost = %d' % (
                      matcher.left[l].label,
                      matcher.right[r].label,
                      json_dict[matcher.left[l].label],
                      matcher.flow.UnitCost(arc)))
        print('Total cost:', matcher.flow.OptimalCost())

    print(f'Status: {matcher.solve_status} of\n'
          f'        {ortg.SimpleMinCostFlow.NOT_SOLVED}: not solved,\n'
          f'        {ortg.SimpleMinCostFlow.OPTIMAL}: optimal,\n'
          f'        {ortg.SimpleMinCostFlow.FEASIBLE}: feasible,\n'
          f'        {ortg.SimpleMinCostFlow.INFEASIBLE}: infeasible,\n'
          f'        {ortg.SimpleMinCostFlow.UNBALANCED}: unbalanced,\n'
          f'        {ortg.SimpleMinCostFlow.BAD_RESULT}: bad result,\n'
          f'        {ortg.SimpleMinCostFlow.BAD_COST_RANGE}: bad cost range')
    print('Preferences statistics:', matcher.prefs_min, matcher.prefs_max,
          matcher.prefs_mean, matcher.prefs_std, matcher.prefs_qtiles)


def run_benchmark(json_dict, matcher, string_metric):
    keys = list(json_dict.keys())
    values = list(json_dict.values())
    m = matcher(keys, values)

    time_init = time()
    m.set_prefs(string_metric)
    time_end_prefs = time()
    m.match()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init

    others = {}
    acc = m.accuracy(json_dict, opt_dict_out=others)
    errs = others.get('errors')

    if args.matcher == 'mcf':
        report_flow(m, json_dict, errs)

    return {
        'total_time': total_time,
        'preferences_time': prefs_time,
        'accuracy': acc,
        'errors': errs
    }


def output(output_path, x, y):
    with open(output_path, 'a') as out_file:
        print(f'{x}, {y}', file=out_file)


def main():
    print('Running benchmark with arguments:', args)

    with open(args.json, 'r') as input_json:
        input_dict = json.load(input_json)

    if args.size:
        tmp = sample(input_dict.items(), args.size)
        input_dict = dict(tmp)

    size = len(input_dict)
    matcher = selected_matcher()
    string_metric = selected_metric()

    results = run_benchmark(input_dict, matcher, string_metric)

    acc = results['accuracy']
    total_time = results['total_time']
    prefs_time = results['preferences_time']
    print('Accuracy:', acc)
    print(f'Preferences run time: {prefs_time} seconds'
          f' ({prefs_time / 60} minutes)')
    print(f'Total run time: {total_time} seconds'
          f' ({total_time / 60} minutes)')

    if args.outdir:
        filename = args.json.split('/')[-1]
        for label, value in zip(['sec', 'acc'], [total_time, acc]):
            output_path = f'{args.outdir}/{filename}_{label}.csv'
            if args.reset:
                with open(output_path, 'w') as out_file:
                    print('x, y', file=out_file)
            output(output_path, size, value)


def selected_matcher():
    if args.matcher == 'mcf':
        return graph.MinCostFlow
    elif args.matcher == 'gs':
        return graph.StableMarriage
    elif args.matcher == 'igs':
        return graph.SimpleMarriageAttempt
    elif args.matcher == 'lgm':
        return graph.LeftGreedyMarriage


def selected_metric():
    metric = None
    if args.distance == 'qg':
        metric = partial(string.qdistance, q=args.q)
    elif args.distance == 'ed':
        metric = partial(string.edit_distance, trim=args.trim)
    return partial(graph.vertex_diff, string_metric=metric)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('json', type=str, help='path to .json input file')
    argp.add_argument('-m', '--matcher', choices=['gs', 'igs', 'lgm', 'mcf'],
                      default='mcf',
                      help='select which bipartite matcher to use: gs for'
                           ' Gale-Shapley\'s stable marriage, igs for an'
                           ' adaptation of gs for incomplete preference lists,'
                           ' lgm for a greedy and even more left-biased'
                           ' adaptation of gs for incomplete preference lists'
                           ' or mcf for min-cost flow. Default: %(default)s')
    argp.add_argument('-d', '--distance', choices=['ed', 'qg'], default='qg',
                      help='select which string similarity metric to use:'
                           ' ed for edit distance and qg for q-gram distance.'
                           ' Default: %(default)s')
    argp.add_argument('-q', type=int, default=2,
                      help='the q if chosen metric is q-gram distance.'
                           ' Default: %(default)s')
    argp.add_argument('-t', '--trim', action='store_true',
                      help='set to trim strings if chosen metric is edit'
                           ' distance')
    argp.add_argument('-s', '--size', type=int,
                      help='randomly pick this many entries from input')
    argp.add_argument('-o', '--outdir', type=str,
                      help='directory to output time and accuracy as files')
    argp.add_argument('-r', '--reset', action='store_true',
                      help='reset file before output')
    args = argp.parse_args()

    main()
