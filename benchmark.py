import argparse
import json
from functools import partial
from pathlib import Path
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


def run_benchmark(json_dict, matcher, filter_class, string_metric):
    keys = list(json_dict.keys())
    values = list(json_dict.values())
    m = matcher(keys, values, filter_class=filter_class)

    time_init = time()
    m.set_prefs(string_metric)
    time_end_prefs = time()
    m.match()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init

    others = {}
    acc = m.accuracy(json_dict, opt_dict_out=others)

    errs = others.get('errors', [])
    if isinstance(m, graph.MinCostFlow):
        report_flow(m, json_dict, errs)

    print('Dataset size:', m.n)

    if isinstance(m, graph.IncompleteBipartiteMatcher):
        print('Lengths of preferences:\n'
              f'     min: {m.prefs_min}\n'
              f'     max: {m.prefs_max}\n'
              f'    mean: {m.prefs_mean}\n'
              f'  stddev: {m.prefs_std}\n'
              f'  qtiles: {m.prefs_qtiles}')

    unmatched = others.get('unmatched', [])
    print('Accuracy:', acc)
    print(f'Unmatched elements: {len(unmatched)} ({len(unmatched) / m.n})')
    print(f'Mismatched elements: {len(errs)} ({len(errs) / m.n})')
    print(f'Preferences run time: {prefs_time} seconds'
          f' ({prefs_time / 60} minutes)')
    print(f'Total run time: {total_time} seconds'
          f' ({total_time / 60} minutes)')

    return {
        'total_time': total_time,
        'preferences_time': prefs_time,
        'accuracy': acc
    }


def main():
    print('Running benchmark with arguments:', args)

    with open(args.json, 'r') as input_json:
        input_dict = json.load(input_json)

    if args.size:
        tmp = sample(input_dict.items(), args.size)
        input_dict = dict(tmp)

    matcher = selected_matcher()
    filter_class = selected_filter()
    string_metric = selected_metric()

    # input_dict is not available in selected_matcher()
    # and I didn't want to put it there
    if matcher == graph.CheatingMatcher:
        matcher = partial(graph.CheatingMatcher, correct_mapping=input_dict)

    results = run_benchmark(input_dict, matcher, filter_class, string_metric)

    if args.outdir:
        acc = results['accuracy']
        total_time = results['total_time']

        outdir = Path(args.outdir)
        if not outdir.exists():
            outdir.mkdir(parents=True)

        filename_in = args.json.split('/')[-1]
        filename_out = args.matcher if args.comparison else filename_in
        outfile = outdir / f'{filename_out}.csv'

        if args.reset or not outfile.exists():
            with outfile.open('w') as f:
                if not args.comparison:
                    print('matcher,', end='', file=f)
                print('case,accuracy,time', file=f)
        with outfile.open('a') as f:
            if not args.comparison:
                print(f'{args.matcher},', end='', file=f)
            print(f'{filename_in},{acc},{total_time}', file=f)


def selected_matcher():
    if args.matcher == 'mcf':
        return graph.MinCostFlow
    elif args.matcher == 'gs':
        return graph.StableMarriage
    elif args.matcher == 'igs':
        return graph.SimpleMarriageAttempt
    elif args.matcher == 'lgm':
        return graph.LeftGreedyMarriage
    elif args.matcher == 'c':
        return graph.CheatingMatcher


def selected_filter():
    if args.filter == 'sa':
        return string.SuffixArray
    elif args.filter == 'qg':
        return partial(string.QGramIndex, q=args.q)


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
    argp.add_argument('-m', '--matcher', choices=['gs', 'igs', 'lgm',
                                                  'mcf', 'c'],
                      default='mcf',
                      help='select which bipartite matcher to use: gs for'
                           ' Gale-Shapley\'s stable marriage; igs for an'
                           ' adaptation of gs for incomplete preference lists;'
                           ' lgm for a greedy and even more left-biased'
                           ' adaptation of gs for incomplete preference lists;'
                           ' mcf for min-cost flow; or c for a cheating matcher'
                           ' that assigns the correct mapping if it is present'
                           ' in the preference lists.'
                           ' Default: %(default)s')
    argp.add_argument('-f', '--filter', choices=['sa', 'qg'], default='sa',
                      help='select which filter to use: sa for suffix array'
                           ' and qg for q-gram index. Default: %(default)s')
    argp.add_argument('-d', '--distance', choices=['ed', 'qg'], default='qg',
                      help='select which string similarity metric to use:'
                           ' ed for edit distance and qg for q-gram distance.'
                           ' Default: %(default)s')
    argp.add_argument('-q', type=int, default=2,
                      help='the q if chosen metric is q-gram distance or'
                           ' filter is q-gram index.'
                           ' Default: %(default)s')
    argp.add_argument('-t', '--trim', action='store_true',
                      help='set to trim strings if chosen metric is edit'
                           ' distance')
    argp.add_argument('-s', '--size', type=int,
                      help='randomly pick this many entries from input')
    argp.add_argument('-o', '--outdir', type=str,
                      help='directory to output time and accuracy as files')
    argp.add_argument('-c', '--comparison', action='store_true',
                      help='switch output of time and accuracy in a file'
                           ' dedicated to the selected matcher instead of'
                           ' given input. Good for comparing different'
                           ' matchers')
    argp.add_argument('-r', '--reset', action='store_true',
                      help='reset file before output')
    args = argp.parse_args()

    main()
