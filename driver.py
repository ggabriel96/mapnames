from optparse import OptionParser
from SuffixTree import CreateTree
import json
import gc
import progressbar

parser = OptionParser()
parser.add_option("-l", "--limit", dest="limit", type="int",
                  help="set a LIMIT ot the number of strings from input", metavar="LIMIT")

#parser.add_option("-q", "--quiet",
#                  action="store_false", dest="verbose", default=True,
#                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()


result = []

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


print(result, sum(a/b for [b,a] in result)/len(result))
