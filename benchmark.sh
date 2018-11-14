#!/bin/bash

folder=test_cases # path to folder where .json test cases are
# manually change arguments below, like -m and -o:
for file in $(ls ${folder}); do
    python benchmark.py ${folder}/${file} -c -m c -f si -o sik
done
