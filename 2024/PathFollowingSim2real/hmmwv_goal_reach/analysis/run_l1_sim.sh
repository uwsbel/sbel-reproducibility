#!/bin/bash

cd /home/harry/projectlets/hmmwv_goal_reach

conda activate tutorial

for i in 1 2 3 4; do
  for j in $(seq 0 99); do
    echo "Running i=$i j=$j"
    python simulation/l2_testing.py $i $j
  done
done