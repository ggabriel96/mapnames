#!/bin/bash

folder=test_cases # path to folder where .json test cases are
# manually change arguments below, like -m and -o:
for file in $(ls -v ${folder}); do
    # compare filters:
    python benchmark.py ${folder}/${file} -c -m   c   -f si -d qg -o cmpf
    python benchmark.py ${folder}/${file} -c -m   c   -f sa -d qg -o cmpf
    # compare filters with random sample:
    python benchmark.py ${folder}/${file} -c -m   c   -f si -d qg -o cmpf/rand -k 5000 -s 7814958244
    python benchmark.py ${folder}/${file} -c -m   c   -f sa -d qg -o cmpf/rand -k 5000 -s 7814958244
    python benchmark.py ${folder}/${file} -c -m   c   -f qg -d qg -o cmpf/rand -k 5000 -s 7814958244
    # compare matchers:
    python benchmark.py ${folder}/${file} -c -m lgm   -f sa -d qg -o cmpm
    # compare distances
    python benchmark.py ${folder}/${file} -c -m   c   -f sa -d ed -o cmpd -k 5000 -s 7814958244
    python benchmark.py ${folder}/${file} -c -m   c   -f sa -d qg -o cmpd -k 5000 -s 7814958244
done
