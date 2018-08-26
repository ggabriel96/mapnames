import json
import sys

with open(sys.argv[1]) as f:
    x = json.load(f)

    for k,v in x.items():
        print(k, v)



