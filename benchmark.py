import argparse
import csv
import json
import pprint
from functools import partial
from pathlib import Path
from random import sample, seed
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


def run_benchmark(json_dict, matcher_cls, filter_cls, strdist_cls):
    keys = list(json_dict.keys())
    values = list(json_dict.values())
    matcher = matcher_cls(keys, values, filter_class=filter_cls)

    time_init = time()
    matcher.set_prefs(strdist_cls)
    time_end_prefs = time()
    matcher.match()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init

    acc_opt = {}
    acc = matcher.accuracy(json_dict, opt_dict_out=acc_opt)

    stats = {
        'size': len(keys),
        'accuracy': acc,
        'total_time': total_time,
        'prefs_time': prefs_time
    }

    if isinstance(matcher, graph.IncompleteBipartiteMatcher):
        q1, q2, q3 = matcher.prefs_qtiles
        stats['prefs_min'] = int(matcher.prefs_min)
        stats['prefs_max'] = int(matcher.prefs_max)
        stats['prefs_mean'] = float(matcher.prefs_mean)
        stats['prefs_std'] = float(matcher.prefs_std)
        stats['prefs_q1'] = float(q1)
        stats['prefs_q2'] = float(q2)
        stats['prefs_q3'] = float(q3)

    return {
        'acc_opt': acc_opt,
        'matcher': matcher,
        'stats': stats
    }


def get_outfile():
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    filename_in = args.json.split('/')[-1]
    filename_out = f'{args.matcher}_{args.filter}_{args.distance}' \
        if args.comparison \
        else filename_in
    outfile = outdir / f'{filename_out}.csv'
    return outfile


def main():
    pprinter = pprint.PrettyPrinter(indent=2)
    print('\nRunning benchmark with arguments: ', end='')
    pprinter.pprint(vars(args))

    with open(args.json, 'r') as input_json:
        input_dict = json.load(input_json)

    if args.size:
        seed(args.seed)
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
        outfile = get_outfile()
        print_header = not outfile.exists()
        with outfile.open('a') as f:
            stats = results['stats']
            stats.update(vars(args))
            csvw = csv.DictWriter(f, fieldnames=stats.keys())
            if print_header:
                csvw.writeheader()
            csvw.writerow(stats)
    else:
        stats = results['stats']
        others = results['acc_opt']
        matcher = results['matcher']

        errs = others.get('errors', [])
        unmatched = others.get('unmatched', [])
        stats['unmatched'] = len(unmatched) / matcher.n
        stats['mismatched'] = len(errs) / matcher.n

        if isinstance(matcher, graph.MinCostFlow):
            report_flow(matcher, input_dict, errs)

        print('stats: ', end='')
        pprinter.pprint(stats)


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
    elif args.filter == 'si':
        return string.SimpleIndex


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
    argp.add_argument('-f', '--filter', choices=['sa', 'qg', 'si'],
                      default='sa',
                      help='select which filter to use: sa for suffix array;'
                           ' qg for q-gram index; or si for simple index,'
                           ' which uses the same bounds search of sa but just'
                           ' over a sorted list of the input strings.'
                           ' Default: %(default)s')
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
    argp.add_argument('-k', '--size', type=int,
                      help='randomly pick this many entries from input')
    argp.add_argument('-s', '--seed', type=int, default=None,
                      help='seed to initialize RNG. Default: %(default)s')
    argp.add_argument('-o', '--outdir', type=str,
                      help='directory to output time and accuracy as files')
    argp.add_argument('-c', '--comparison', action='store_true',
                      help='switch output of time and accuracy in a file'
                           ' dedicated to the selected matcher instead of'
                           ' given input. Good for comparing different'
                           ' matchers')
    args = argp.parse_args()

    main()
