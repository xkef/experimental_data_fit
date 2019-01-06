#!/bin/bash

for d in results/params* ; do
    cat $d  >> ranking.txt
  done
cat ranking.txt | sort -g >> sorted_ranking.txt
rm ranking.txt
