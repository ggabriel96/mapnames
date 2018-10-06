import csv
import sys
from collections import defaultdict as dd
from glob import glob

plot_paths = glob(f'{sys.argv[1]}/*.csv')
plot_paths.sort()
values_per_size = dd(lambda: [])
# values_per_case = dd(lambda: [])

for path in plot_paths:
    with open(path, newline='') as csvfile:
        csvr = csv.reader(csvfile)
        next(csvr)  # escape header
        for row in csvr:
            # values_per_case[path].append(float(row[1]))
            values_per_size[int(row[0])].append(float(row[1]))

# for k, v in values_per_case.items():
#     case = k.split('/')[-1][0]
#     with open(f'{sys.argv[1]}/box_per_case_{case}.csv', 'w') as rf:
#         for u in v:
#             print(u, file=rf)

for k, v in values_per_size.items():
    with open(f'{sys.argv[1]}/{k}_per_size.csv', 'w') as rf:
        for i in range(len(v)):
            print(f'{i+1}, {v[i]}', file=rf)
