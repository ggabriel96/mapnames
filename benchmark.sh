#!/bin/bash

folder=cases_all # path to folder where .json test cases are
# manually change arguments below, like -m and -o:
for file in $(ls ${folder}); do
    python benchmark.py ${folder}/${file} -m igs -o out_bench/ -c
done
