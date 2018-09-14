import gc
import json
from optparse import OptionParser

import progressbar

from wuerges import GaleShapley as gs
from wuerges.SuffixTree import CreateTree

"""
This module is the driver program that can be used to process the testcases.
"""

parser = OptionParser(usage = "usage %prog [options] json1 json2 ...")
parser.add_option("-l", "--limit", dest="limit", type="int",
                  help="set a LIMIT ot the number of strings from input", metavar="LIMIT")

#parser.add_option("-q", "--quiet",
#                  action="store_false", dest="verbose", default=True,
#                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

if not args:
    parser.print_help()
    exit(0)
result = []
result_match = []

for arg in args:
    with open(arg) as f:
        print("Working on input", arg)
        x = json.load(f)
        if type(x) is dict:
            x = [list(x.keys()), list(x.values())]

        if options.limit:
            x[0] = x[0][:options.limit]
            x[1] = x[1][:options.limit]
        gc.collect()

        print("Creating Tree")
        t1 = CreateTree(x[0])

        g = gs.G(len(x[0]))

        count = 0
        correct = 0
        j = 0

        print("Matching Terms")
        for term in progressbar.progressbar(x[1]):
            count += 1

            # the score of the best match
            best = 0.0

            best_wid = -1
            best_term = term
            best_sz = -1

            for i in range(len(term)-1):
                p, w, wid, sz, lm = t1.search(term[i:])
                #score = lm
                #score = 1/sz
                score = lm/sz

                if score > best:
                    g.grade(j, wid, lm/sz)
                    g.grade(wid, j, lm/sz)

                #if sz < best:
                    best = score
                    best_term = term[i:i+lm]
                    best_sz = sz
                    best_wid = wid

            if j == best_wid:
                correct += 1
            j += 1
            #else:
                #print("-"*10)
                #print(j, best_wid, best_sz, best_term)
                #print("original:", term)
                #print("cut:", best_term)
                #print(x[0][j])
                #print(x[1][j])
                #print(x[0][best_wid])
                #print(x[1][best_wid])

        result.append([count, correct])
        del t1
        gc.collect()

        ps = g.makeprefs()
        #del g

        res = gs.GaleShapley(g.p, ps)
        #print(res)

        c1 = 0
        c2 = 0
        for a, b in res.items():
            c2 += 1
            if a == b:
                c1 += 1

        result_match.append([c2, c1])

        print("Results without matching")
        print(result, sum(a/b for [b,a] in result)/len(result))
        print("Results using crappy matching")
        print(result_match, sum(a/b for [b,a] in result_match)/len(result_match))
