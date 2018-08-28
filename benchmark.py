import argparse
import json
from time import time

from ortools.graph import pywrapgraph

import Graph
from SuffixArray import SuffixArray


def accuracy_marriage(matching, original_mapping):
    errors = []
    equal_amount = 0
    for k, v in matching.items():
        is_equal = v.label == original_mapping[k.label]
        equal_amount += int(is_equal)
        if not is_equal:
            errors.append(k)
    return equal_amount / len(matching), errors


def accuracy_assignment(assignment, bipartite_matcher, original_mapping):
    errors = []
    equal_amount = 0
    for l in range(assignment.NumNodes()):
        r = assignment.RightMate(l)
        is_equal = bipartite_matcher.right[r].label == \
                   original_mapping[bipartite_matcher.left[l].label]
        equal_amount += int(is_equal)
        if not is_equal:
            errors.append(l)
    return equal_amount / assignment.NumNodes(), errors


def assignment_bench(json_dict):
    keys = list(json_dict.keys())
    values = list(json_dict.values())

    matcher = Graph.BipartiteMatcher(keys, values, SuffixArray)
    time_init = time()
    matcher.set_prefs(Graph.vertex_diff)
    time_end_prefs = time()
    assignment = pywrapgraph.LinearSumAssignment()

    for l in matcher.left:
        for r, cost in l.ratings:
            assignment.AddArcWithCost(l.idx, r.idx, int(cost))

    for r in matcher.right:
        for l, cost in r.ratings:
            assignment.AddArcWithCost(l.idx, r.idx, int(cost))

    solve_status = assignment.Solve()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init
    acc, errs = 0, []
    if solve_status == assignment.OPTIMAL:
        acc, errs = accuracy_assignment(assignment, matcher, json_dict)
        for i in errs:
            print('----------------------> %s\n'
                  '  got mapped to       : %s\n'
                  '  but should have been: %s\n'
                  '  Cost = %d' % (
                      matcher.left[i],
                      matcher.right[assignment.RightMate(i)],
                      matcher.right[i],
                      assignment.AssignmentCost(i)))
        print(f'{len(errs)} wrong assignments')
        print('Accuracy:', acc)
        print('Total cost:', assignment.OptimalCost())
    elif solve_status == assignment.INFEASIBLE:
        print('No assignment is possible.')
    elif solve_status == assignment.POSSIBLE_OVERFLOW:
        print(
            'Some input costs are too large and may cause an integer overflow.')
    print(f'Preferences run time: {prefs_time} seconds'
          f' ({prefs_time / 60} minutes)')
    print(f'Total run time: {total_time} seconds'
          f' ({total_time / 60} minutes)')
    print(matcher.prefs_min, matcher.prefs_max,
          matcher.prefs_mean, matcher.prefs_std, matcher.prefs_qtiles)

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
    matching = G.stable_match()
    time_end = time()

    prefs_time = time_end_prefs - time_init
    total_time = time_end - time_init
    acc, errs = accuracy_marriage(matching, json_dict)

    return {
        'matching': matching,
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
