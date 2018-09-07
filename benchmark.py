import argparse
import json
from time import time

from ortools.graph import pywrapgraph as ortg

import Graph
from SuffixArray import SuffixArray


def report_flow(matcher, json_dict):
    acc, errs = None, None
    if matcher.solve_status == ortg.SimpleMinCostFlow.OPTIMAL:
        acc, errs = matcher.accuracy(json_dict)
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
        print(f'{len(errs)} wrong assignments')
        print('Accuracy:', acc)
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
    return acc, errs


def assignment_bench(json_dict):
    keys = list(json_dict.keys())
    values = list(json_dict.values())

    matcher = Graph.BipartiteMatcher(keys, values, SuffixArray)

    time_init = time()
    matcher.set_prefs(Graph.vertex_diff)
    time_end_prefs = time()
    matcher.min_cost_flow()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init

    acc, errs = report_flow(matcher, json_dict)

    return {
        'total_time': total_time,
        'preferences_time': prefs_time,
        'accuracy': acc,
        'errors': errs
    }


def benchmark(json_dict):
    keys = list(json_dict.keys())
    values = list(json_dict.values())

    G = Graph.BipartiteMatcher(keys, values)

    time_init = time()
    G.set_prefs(Graph.vertex_diff)
    time_end_prefs = time()
    G.stable_match()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init
    acc, errs = G.accuracy(json_dict)

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

    size = len(input_dict)
    rslts = assignment_bench(input_dict)

    total_time = rslts['total_time']
    prefs_time = rslts['preferences_time']
    print(f'Preferences run time: {prefs_time} seconds'
          f' ({prefs_time / 60} minutes)')
    print(f'Total run time: {total_time} seconds'
          f' ({total_time / 60} minutes)')

    paths = args.json.split('/')
    filename = paths[-1]
    for label, value in \
            zip(['sec', 'acc'], [rslts['total_time'], rslts['accuracy']]):
        output_path = f'{args.outdir}/{filename}_{label}.csv'
        if args.reset:
            with open(output_path, 'w') as out_file:
                print('x, y', file=out_file)
        output(output_path, size, value)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('json', type=str, help='path to .json input file')
    argp.add_argument('outdir', type=str,
                      help='directory to output time and accuracy as files')
    argp.add_argument('-r', '--reset', action='store_true',
                      help='reset file before output')
    args = argp.parse_args()

    main()
